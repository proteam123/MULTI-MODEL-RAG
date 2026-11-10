import base64
import hashlib
import io
import json
import re
from pathlib import Path

import pymupdf
from PIL import Image
from groq import Groq
from langchain_core.documents import Document

from config import IMAGE_DIR, VISION_MODEL


def image_to_data_uri(image_path: str | Path) -> str:
    with Image.open(image_path) as img:
        img = img.convert("RGB")
        img.thumbnail((1600, 1600))
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=85)
    return image_bytes_to_data_uri(buffer.getvalue())


def image_bytes_to_data_uri(image_data: bytes) -> str:
    encoded = base64.b64encode(image_data).decode("utf-8")
    return f"data:image/jpeg;base64,{encoded}"


def summarize_visual(groq_client: Groq, image_path: str | Path, page_number: int) -> str:
    image_data = image_to_data_uri(image_path)
    prompt = f"""
This visual was extracted from page {page_number} of the uploaded report.
Describe useful business information visible in it, concisely and factually.
For a chart, mention important values, high/low values, and the main trend.
For a diagram, identify components and explain their relationships or flow.
"""
    response = groq_client.chat.completions.create(
        model=VISION_MODEL,
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": image_data}},
            ],
        }],
        temperature=0,
        max_completion_tokens=500,
    )
    return response.choices[0].message.content.strip()


def extract_multimodal_documents(
    pdf_path: str | Path,
    groq_client: Groq,
    source_name: str | None = None,
    source_path: str | Path | None = None,
) -> list[Document]:
    pdf_path = Path(pdf_path)
    source_name = source_name or pdf_path.name
    source_path = str(Path(source_path or pdf_path).resolve())
    pdf = pymupdf.open(str(pdf_path))
    documents: list[Document] = []
    extracted_xrefs = set()

    try:
        for page_index in range(len(pdf)):
            page = pdf[page_index]
            page_number = page_index + 1

            text = page.get_text("text").strip()
            if text:
                documents.append(Document(
                    page_content=text,
                    metadata={
                        "page": page_number,
                        "modality": "text",
                        "source": source_name,
                        "source_path": source_path,
                    },
                ))

            try:
                for table_number, table in enumerate(page.find_tables().tables, start=1):
                    df = table.to_pandas()
                    if not df.empty:
                        documents.append(Document(
                            page_content=df.to_markdown(index=False),
                            metadata={
                                "page": page_number, "modality": "table",
                                "table_number": table_number, "source": source_name,
                                "source_path": source_path,
                            },
                        ))
            except Exception as error:
                print(f"Table extraction warning on page {page_number}: {error}")

            for image_number, image_info in enumerate(page.get_images(full=True), start=1):
                xref = image_info[0]
                if xref in extracted_xrefs:
                    continue
                extracted_xrefs.add(xref)

                image_data = pdf.extract_image(xref)
                image_path = IMAGE_DIR / (
                    f"{pdf_path.stem}_page_{page_number}_image_{image_number}."
                    f"{image_data['ext']}"
                )
                image_path.write_bytes(image_data["image"])

                try:
                    summary = summarize_visual(groq_client, image_path, page_number)
                    documents.append(Document(
                        page_content=summary,
                        metadata={
                            "page": page_number, "modality": "visual",
                            "image_path": str(image_path), "source": source_name,
                            "source_path": source_path,
                        },
                    ))
                except Exception as error:
                    print(f"Vision warning on page {page_number}: {error}")
    finally:
        pdf.close()

    return documents


def crop_relevant_page_region(
    pdf_path: str | Path,
    page_number: int,
    question: str,
    groq_client: Groq,
    source_id: str,
) -> Path | None:
    pdf = pymupdf.open(str(pdf_path))
    try:
        if page_number < 1 or page_number > len(pdf):
            return None

        page = pdf[page_number - 1]
        pixmap = page.get_pixmap(matrix=pymupdf.Matrix(1.5, 1.5), alpha=False)
        page_image = Image.open(io.BytesIO(pixmap.tobytes("png"))).convert("RGB")
        page_image.thumbnail((1600, 1600))
        preview = io.BytesIO()
        page_image.save(preview, format="JPEG", quality=85)

        response = groq_client.chat.completions.create(
            model=VISION_MODEL,
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"""
Find the single graph, chart, diagram, or image on this report page that best
matches the user's request. Return only a JSON object with integer coordinates
on a 0-1000 scale: {{"found": true, "x1": 0, "y1": 0, "x2": 1000, "y2": 1000}}.
Coordinates are relative to the displayed image. Include a small margin around
the entire visual, including its labels and legend. If no matching visual is
present, return {{"found": false}}.

User request: {question}
""",
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_bytes_to_data_uri(preview.getvalue())
                        },
                    },
                ],
            }],
            temperature=0,
            max_completion_tokens=200,
        )
        response_text = response.choices[0].message.content or ""
        json_match = re.search(r"\{.*\}", response_text, re.DOTALL)
        if not json_match:
            return None

        try:
            coordinates = json.loads(json_match.group())
            if not coordinates.get("found"):
                return None
            x1, y1, x2, y2 = (
                int(coordinates[key]) for key in ("x1", "y1", "x2", "y2")
            )
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            return None
        x1 = max(0, min(999, x1))
        y1 = max(0, min(999, y1))
        x2 = max(x1 + 1, min(1000, x2))
        y2 = max(y1 + 1, min(1000, y2))

        width, height = page_image.size
        left = max(0, int(width * x1 / 1000) - int(width * 0.015))
        top = max(0, int(height * y1 / 1000) - int(height * 0.015))
        right = min(width, int(width * x2 / 1000) + int(width * 0.015))
        bottom = min(height, int(height * y2 / 1000) + int(height * 0.015))
        crop = page_image.crop((left, top, right, bottom))

        crop_id = hashlib.sha256(
            f"{source_id}:{page_number}:{question}".encode("utf-8")
        ).hexdigest()
        crop_path = IMAGE_DIR / f"{crop_id}_crop.jpg"
        crop.save(crop_path, format="JPEG", quality=90)
        return crop_path
    finally:
        pdf.close()
