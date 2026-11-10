import hashlib
import os
import re
from pathlib import Path
from datetime import datetime, timezone

from groq import Groq
from pinecone import Pinecone, ServerlessSpec
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore

from config import (
    TEXT_MODEL, VISION_MODEL, EMBEDDING_MODEL,
    PINECONE_INDEX_NAME, PINECONE_NAMESPACE, PINECONE_HISTORY_NAMESPACE,
)
from ingestion import (
    crop_relevant_page_region,
    extract_multimodal_documents,
    image_to_data_uri,
)

HISTORY_RECORD_ID = "latest-indexed-document"


def _clients():
    if not os.getenv("GROQ_API_KEY"):
        raise RuntimeError("GROQ_API_KEY is missing. Add it to your .env file.")
    if not os.getenv("PINECONE_API_KEY"):
        raise RuntimeError("PINECONE_API_KEY is missing. Add it to your .env file.")
    groq_client = Groq(api_key=os.environ["GROQ_API_KEY"])
    text_llm = ChatGroq(model=TEXT_MODEL, temperature=0)
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        encode_kwargs={"normalize_embeddings": True},
    )
    return groq_client, text_llm, embeddings


def _pinecone_index(embeddings):
    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    dimension = len(embeddings.embed_query("dimension check"))
    if not pc.has_index(PINECONE_INDEX_NAME):
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=dimension,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        # Pinecone normally becomes ready shortly after creation.
        import time
        while not pc.describe_index(PINECONE_INDEX_NAME).status["ready"]:
            time.sleep(2)
    return pc, pc.Index(PINECONE_INDEX_NAME)


def ingest_pdf(pdf_path: str | Path, source_name: str | None = None) -> int:
    groq_client, _, embeddings = _clients()
    _, index = _pinecone_index(embeddings)

    pdf_path = Path(pdf_path)
    source_name = source_name or Path(pdf_path).name
    source_id = hashlib.sha256(pdf_path.read_bytes()).hexdigest()
    documents = extract_multimodal_documents(
        pdf_path,
        groq_client,
        source_name,
        source_path=pdf_path,
    )
    if not documents:
        raise RuntimeError("No text, tables, or embedded images were extracted from this PDF.")

    for document in documents:
        document.metadata["source_id"] = source_id
    vector_ids = [
        hashlib.sha256(f"{source_id}:{position}".encode()).hexdigest()
        for position in range(len(documents))
    ]
    vectorstore = PineconeVectorStore(
        index_name=PINECONE_INDEX_NAME,
        embedding=embeddings,
        namespace=PINECONE_NAMESPACE,
    )
    vectorstore.add_documents(documents, ids=vector_ids)

    index.upsert(
        vectors=[{
            "id": HISTORY_RECORD_ID,
            "values": embeddings.embed_query("latest indexed PDF history record"),
            "metadata": {
                "source": source_name,
                "source_id": source_id,
                "indexed_at": datetime.now(timezone.utc).isoformat(),
                "document_count": len(documents),
            },
        }],
        namespace=PINECONE_HISTORY_NAMESPACE,
    )
    return len(documents)


def get_latest_document_history() -> dict | None:
    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        raise RuntimeError("PINECONE_API_KEY is missing. Add it to your .env file.")

    pc = Pinecone(api_key=api_key)
    if not pc.has_index(PINECONE_INDEX_NAME):
        return None

    response = pc.Index(PINECONE_INDEX_NAME).fetch(
        ids=[HISTORY_RECORD_ID],
        namespace=PINECONE_HISTORY_NAMESPACE,
    )
    vector = response.vectors.get(HISTORY_RECORD_ID)
    return vector.metadata if vector else None


def get_index_stats():
    _, index = _pinecone_index(
        HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL,
                              encode_kwargs={"normalize_embeddings": True})
    )
    return index.describe_index_stats()


def _get_retriever():
    _, _, embeddings = _clients()
    vectorstore = PineconeVectorStore(
        index_name=PINECONE_INDEX_NAME,
        embedding=embeddings,
        namespace=PINECONE_NAMESPACE,
    )
    return vectorstore.as_retriever(search_kwargs={"k": 5})


def format_context(retrieved_docs) -> str:
    return "\n\n".join(
        f"[Document: {doc.metadata.get('source', 'Unknown')} | "
        f"Page {doc.metadata.get('page')} | "
        f"{doc.metadata.get('modality', '').upper()}]\n"
        f"{doc.page_content}"
        for doc in retrieved_docs
    )


def _format_chat_history(chat_history: list[dict] | None) -> str:
    if not chat_history:
        return "No earlier conversation."
    return "\n".join(
        f'{message["role"].upper()}: {message["content"]}'
        for message in chat_history[-10:]
    )


def _is_image_request(question: str) -> bool:
    return bool(re.search(
        r"^\s*/show\b|\b(show|display)\b.*\b(graph|chart|image|diagram|visual)\b",
        question,
        re.IGNORECASE,
    ))


def _retrieval_query(question: str) -> str:
    command = re.match(
        r"^\s*/show(?:\s+(graph|chart|image|diagram|visual))?"
        r"(?:\s+(.*))?$",
        question,
        re.IGNORECASE,
    )
    if not command:
        return question
    details = (command.group(2) or "").strip()
    return details or command.group(1) or "graph chart diagram image"


def answer_with_vision(
    groq_client,
    question: str,
    context: str,
    image_paths: list[str],
    chat_history: list[dict] | None = None,
) -> str:
    content = [{
        "type": "text",
        "text": f"""
You answer questions about the uploaded documents.
Use ONLY the retrieved context and attached retrieved visuals.
Synthesize a direct answer, cite document names and page numbers when available,
and say when the retrieved information is insufficient.

CONVERSATION:
{_format_chat_history(chat_history)}

RETRIEVED CONTEXT:
{context}

QUESTION:
{question}
""",
    }]
    for image_path in image_paths[:3]:
        content.append({
            "type": "image_url",
            "image_url": {"url": image_to_data_uri(image_path)},
        })
    response = groq_client.chat.completions.create(
        model=VISION_MODEL,
        messages=[{"role": "user", "content": content}],
        temperature=0,
        max_completion_tokens=900,
    )
    return response.choices[0].message.content.strip()


def ask_rag(question: str, chat_history: list[dict] | None = None) -> dict:
    groq_client, text_llm, _ = _clients()
    retriever = _get_retriever()
    retrieved_docs = retriever.invoke(_retrieval_query(question))
    if not retrieved_docs:
        return {"answer": "I could not find that information in the report.",
                "documents": [], "image_paths": []}

    context = format_context(retrieved_docs)
    image_paths = []
    image_request = _is_image_request(question)
    if image_request:
        page_candidates = []
        seen_pages = set()
        for doc in retrieved_docs:
            source_path = doc.metadata.get("source_path")
            page_number = doc.metadata.get("page")
            source_id = doc.metadata.get("source_id", "")
            candidate_key = (source_path, page_number)
            if (
                source_path
                and page_number
                and candidate_key not in seen_pages
                and Path(source_path).exists()
            ):
                seen_pages.add(candidate_key)
                page_candidates.append((source_path, page_number, source_id))

        for source_path, page_number, source_id in page_candidates[:3]:
            crop_path = crop_relevant_page_region(
                source_path,
                int(page_number),
                question,
                groq_client,
                source_id,
            )
            if crop_path:
                image_paths.append(str(crop_path))

    if not image_paths:
        for doc in retrieved_docs:
            if doc.metadata.get("modality") == "visual":
                path = doc.metadata.get("image_path")
                if path and Path(path).exists() and path not in image_paths:
                    image_paths.append(path)
    image_paths = image_paths[:3]

    if image_paths:
        answer = answer_with_vision(
            groq_client,
            question,
            context,
            image_paths,
            chat_history,
        )
    else:
        chat_history_text = _format_chat_history(chat_history)
        prompt = ChatPromptTemplate.from_template("""
You answer questions about the user's uploaded documents.
Use ONLY the retrieved context. Synthesize a direct answer, cite document names
and page numbers when possible. If the answer is unavailable, say:
"I could not find that information in the report."
Use prior conversation to understand references in the latest question, but
base factual answers only on the retrieved document context.
{image_request_note}

CONVERSATION:
{chat_history}

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
""")
        chain = prompt | text_llm | StrOutputParser()
        answer = chain.invoke({
            "context": context,
            "question": question,
            "chat_history": chat_history_text,
            "image_request_note": (
                "The user requested a graph or image, but no matching image "
                "could be identified. State that clearly."
                if image_request and not image_paths
                else ""
            ),
        })

    return {"answer": answer, "documents": retrieved_docs, "image_paths": image_paths}
