import importlib
import os
from pathlib import Path

import streamlit as st

from storage import (
    add_chat_message,
    create_chat_session,
    get_chat_messages,
    initialize_storage,
    list_chat_sessions,
    list_recent_documents,
    mark_document_indexed,
    save_uploaded_pdf,
)

initialize_storage()
st.set_page_config(page_title="Multimodal Document Assistant", layout="wide")

st.title("Multimodal Document Assistant")
st.write(
    "Ask questions about your indexed PDFs. Use `/show graph <question>`, "
    "`/show chart <question>`, or `/show image <question>` to retrieve and "
    "display a matching visual crop."
)

missing_keys = [
    key for key in ("GROQ_API_KEY", "PINECONE_API_KEY")
    if not os.getenv(key)
]
if missing_keys:
    st.warning(
        f"Add {', '.join(missing_keys)} to your local .env file to enable "
        "indexing and chat, then restart Streamlit."
    )

st.sidebar.header("Recent documents")
uploaded_pdf = st.sidebar.file_uploader("Add a PDF", type=["pdf"])
if uploaded_pdf is not None:
    saved_document = save_uploaded_pdf(
        uploaded_pdf.name,
        uploaded_pdf.getvalue(),
    )
    st.sidebar.success(f"Saved: {saved_document['source_name']}")

recent_documents = list_recent_documents()
document_ids = [document["source_id"] for document in recent_documents]
selected_document_id = None
if recent_documents:
    selected_document_id = st.sidebar.selectbox(
        "Recent PDFs",
        options=document_ids,
        format_func=lambda source_id: next(
            document["source_name"]
            + (
                " (indexed)"
                if document["indexed_at"]
                else " (not indexed)"
            )
            for document in recent_documents
            if document["source_id"] == source_id
        ),
    )
else:
    st.sidebar.caption("Uploaded PDFs will appear here.")

if st.sidebar.button(
    "Index selected PDF",
    disabled=selected_document_id is None or bool(missing_keys),
):
    selected_document = next(
        document
        for document in recent_documents
        if document["source_id"] == selected_document_id
    )
    file_path = Path(selected_document["file_path"])
    if not file_path.is_file():
        raise FileNotFoundError(
            f'Saved PDF is missing from recent documents: {file_path}'
        )

    import rag

    rag = importlib.reload(rag)
    with st.spinner(f'Indexing {selected_document["source_name"]}...'):
        chunk_count = rag.ingest_pdf(
            file_path,
            source_name=selected_document["source_name"],
        )
    mark_document_indexed(selected_document_id, chunk_count)
    st.sidebar.success(
        f'Indexed {selected_document["source_name"]} ({chunk_count} chunks).'
    )
    st.rerun()

st.sidebar.divider()
st.sidebar.header("Chat sessions")
if st.sidebar.button("New chat", use_container_width=True):
    st.session_state.active_session_id = create_chat_session()
    st.rerun()

chat_sessions = list_chat_sessions()
if not chat_sessions:
    st.session_state.active_session_id = create_chat_session()
    chat_sessions = list_chat_sessions()

session_ids = [session["session_id"] for session in chat_sessions]
active_session_id = st.session_state.get("active_session_id")
if active_session_id not in session_ids:
    active_session_id = session_ids[0]
    st.session_state.active_session_id = active_session_id

active_index = session_ids.index(active_session_id)
active_session_id = st.sidebar.selectbox(
    "Previous chats",
    options=session_ids,
    index=active_index,
    format_func=lambda session_id: next(
        session["title"]
        for session in chat_sessions
        if session["session_id"] == session_id
    ),
)
st.session_state.active_session_id = active_session_id

messages = get_chat_messages(active_session_id)
for message in messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        for image_path in message["image_paths"]:
            if Path(image_path).is_file():
                st.image(image_path, caption="Matching visual crop")
        if message["sources"]:
            with st.expander(f'Retrieved sources ({len(message["sources"])})'):
                for source in message["sources"]:
                    st.caption(
                        f'{source["document"]} | Page {source["page"]} | '
                        f'{source["modality"]}'
                    )
                    st.write(source["content"])

prompt = st.chat_input(
    "Ask about your indexed documents",
    disabled=bool(missing_keys),
)
if prompt:
    add_chat_message(active_session_id, "user", prompt)
    with st.chat_message("user"):
        st.markdown(prompt)

    import rag

    rag = importlib.reload(rag)

    with st.chat_message("assistant"):
        with st.spinner("Searching your documents..."):
            result = rag.ask_rag(prompt, chat_history=messages)
        st.markdown(result["answer"])

        for image_path in result["image_paths"]:
            if Path(image_path).is_file():
                st.image(image_path, caption="Matching visual crop")

        sources = [
            {
                "page": document.metadata.get("page", "Unknown"),
                "modality": document.metadata.get("modality", "text"),
                "document": document.metadata.get(
                    "source",
                    "Unknown document",
                ),
                "content": document.page_content,
            }
            for document in result["documents"]
        ]
        if sources:
            with st.expander(f"Retrieved sources ({len(sources)})"):
                for source in sources:
                    st.caption(
                        f'{source["document"]} | Page {source["page"]} | '
                        f'{source["modality"]}'
                    )
                    st.write(source["content"])

    add_chat_message(
        active_session_id,
        "assistant",
        result["answer"],
        sources=sources,
        image_paths=result["image_paths"],
    )
