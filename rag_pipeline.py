
from pathlib import Path
from io import BytesIO
import shutil

import faiss
import numpy as np
import ollama
import pymupdf
from PIL import Image
from pypdf import PdfReader
import pytesseract
from sentence_transformers import SentenceTransformer


# Convert text into numerical vectors for semantic search.
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def extract_text(pdf_path: str) -> str:
    """Extract embedded text, using OCR when the PDF is scanned."""
    reader = PdfReader(pdf_path)

    pages = []
    for page in reader.pages:
        text = page.extract_text() or ""
        if text.strip():
            pages.append(text)

    full_text = "\n".join(pages)

    if full_text.strip():
        return full_text

    # Locate the Tesseract executable if it was installed outside PATH.
    if not shutil.which("tesseract"):
        candidates = (
            Path("C:/Program Files/Tesseract-OCR/tesseract.exe"),
            Path("C:/Program Files (x86)/Tesseract-OCR/tesseract.exe"),
        )
        for candidate in candidates:
            if candidate.exists():
                pytesseract.pytesseract.tesseract_cmd = str(candidate)
                break

    try:
        with pymupdf.open(pdf_path) as document:
            ocr_pages = []
            for page in document:
                pixmap = page.get_pixmap(
                    matrix=pymupdf.Matrix(2, 2),
                    alpha=False,
                )
                image = Image.open(BytesIO(pixmap.tobytes("png")))
                text = pytesseract.image_to_string(image)
                if text.strip():
                    ocr_pages.append(text)
    except pytesseract.pytesseract.TesseractNotFoundError as exc:
        raise RuntimeError(
            "This scanned PDF needs OCR, but Tesseract could not be found. "
            "Install Tesseract OCR and restart the app."
        ) from exc

    full_text = "\n".join(ocr_pages)
    if not full_text.strip():
        raise ValueError(
            "OCR could not read text from this PDF. It may be blank, "
            "low-resolution, or written in an unsupported language."
        )
    return full_text


def chunk_text(
    text: str,
    chunk_size: int = 180,
    overlap: int = 40,
) -> list[str]:
    """Split text into overlapping word-based chunks."""
    words = text.split()

    if not words:
        return []

    chunks = []
    start = 0

    while start < len(words):
        end = start + chunk_size
        chunks.append(" ".join(words[start:end]))

        if end >= len(words):
            break

        start = end - overlap

    return chunks


class RAGPipeline:
    """Retrieve document passages relevant to a question."""

    def __init__(self):
        print("Loading embedding model...")
        self.model = SentenceTransformer(MODEL_NAME)
        self.index = None
        self.chunks = []
        self.document_text = ""

    def ingest_pdf(self, pdf_path: str):
        """Read a PDF, split its text, and index its embeddings."""
        text = extract_text(pdf_path)
        self.document_text = text
        self.chunks = chunk_text(text)

        if not self.chunks:
            raise ValueError("The PDF contains no usable text.")

        embeddings = self.model.encode(
            self.chunks,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        embeddings = np.asarray(embeddings, dtype="float32")

        # Normalized vectors + inner product = cosine similarity.
        self.index = faiss.IndexFlatIP(embeddings.shape[1])
        self.index.add(embeddings)

        return len(self.chunks)

    def retrieve(self, question: str, top_k: int = 3) -> list[dict]:
        """Return the most relevant passages for a question."""
        if self.index is None:
            raise ValueError("Please ingest a PDF before searching.")

        if not question.strip():
            raise ValueError("Please enter a question.")

        question_embedding = self.model.encode(
            [question],
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        question_embedding = np.asarray(
            question_embedding,
            dtype="float32",
        )

        k = min(top_k, len(self.chunks))
        scores, indices = self.index.search(question_embedding, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue

            results.append(
                {
                    "text": self.chunks[int(idx)],
                    "score": float(score),
                }
            )

        return results

    def generate_answer(self, question: str, top_k: int = 3) -> dict:
        """Generate an answer grounded in retrieved passages."""
        results = self.retrieve(question, top_k=top_k)
        question_lower = question.lower()
        asks_for_count = any(
            phrase in question_lower
            for phrase in ("how many", "count", "number of", "total")
        )
        if asks_for_count and len(self.document_text) <= 40_000:
            context = f"Full document:\n{self.document_text}"
        else:
            context = "\n\n".join(
                f"Passage {i}:\n{item['text']}"
                for i, item in enumerate(results, start=1)
            )
        response = ollama.chat(
            model="llama3.2:3b",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Answer using only the supplied document passages. "
                        "First check every passage for an explicit answer. "
                        "If a passage identifies the requested person, role, "
                        "date, or fact, answer directly with that information "
                        "and cite its passage number. Copy a person's full "
                        "name exactly as it appears in the passage; do not "
                        "shorten it to only a first or last name. Only say the answer is "
                        "not found if none of the passages contains it. Never "
                        "say both that an answer is missing and then provide "
                        "that answer. For counting questions, use the full "
                        "document when provided. For a question asking how "
                        "many male and female characters there are, count the "
                        "central child characters in the story and reply in "
                        "this format: 'Main child characters: N male "
                        "(names), N female (names).' Do not count parents or "
                        "brief supporting roles in that answer. "
                        "For other counting questions, count each person once "
                        "and ensure any listed items match the total. Do not "
                        "infer or invent facts. Keep the answer concise."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Question: {question}\n\n"
                        f"Document passages:\n{context}"
                    ),
                },
            ],
            options={"temperature": 0},
        )
        return {
            "answer": response["message"]["content"],
            "sources": results,
        }
