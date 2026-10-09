
import streamlit as st
from pathlib import Path
import tempfile

from rag_pipeline import RAGPipeline

st.set_page_config(
    page_title="RAG Document Assistant",
    page_icon="📚",
    layout="wide",
)

st.title("📚 RAG Document Q&A Assistant")
st.write(
    "Upload a PDF document and ask questions about its contents. "
    "The app retrieves relevant passages using semantic search."
)

# Keep the pipeline and indexed document available between interactions.
if "pipeline" not in st.session_state:
    st.session_state.pipeline = None

if "document_name" not in st.session_state:
    st.session_state.document_name = None

uploaded_file = st.file_uploader(
    "Upload a PDF document",
    type=["pdf"],
)

if uploaded_file is not None:
    if st.session_state.document_name != uploaded_file.name:
        with st.spinner("Processing PDF and creating embeddings..."):
            try:
                # Use a temporary file because PdfReader accepts a file path.
                with tempfile.NamedTemporaryFile(
                    delete=False, suffix=".pdf"
                ) as temp_file:
                    temp_file.write(uploaded_file.getvalue())
                    temp_path = temp_file.name

                try:
                    if st.session_state.pipeline is None:
                        st.session_state.pipeline = RAGPipeline()

                    chunk_count = (
                        st.session_state.pipeline.ingest_pdf(temp_path)
                    )
                finally:
                    Path(temp_path).unlink(missing_ok=True)

                st.session_state.document_name = uploaded_file.name
                st.success(
                    f"Processed {uploaded_file.name} "
                    f"into {chunk_count} text chunks."
                )

            except Exception as exc:
                st.error(f"Could not process the PDF: {exc}")

if st.session_state.pipeline is not None:
    st.divider()
    st.subheader("Ask your document")

    with st.form("question_form"):
        question = st.text_input(
            "Enter your question",
            placeholder="What are the main points in this document?",
        )
        submitted = st.form_submit_button(
            "Generate answer",
            type="primary",
        )

    if submitted:
        if not question.strip():
            st.warning("Please enter a question.")
        else:
            try:
                with st.spinner("Searching documents and generating answer..."):
                    result = st.session_state.pipeline.generate_answer(question)

                st.subheader("Answer")
                st.write(result["answer"])
                st.subheader("Supporting passages")
                for i, source in enumerate(result["sources"], start=1):
                    with st.expander(f"Passage {i}"):
                        st.write(source["text"])
                        st.caption(f"Similarity score: {source['score']:.3f}")

            except Exception as exc:
                st.error(
                    f"Could not generate an answer: {exc}. "
                    "Check that Ollama is running and the model is installed."
                )
