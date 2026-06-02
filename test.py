import streamlit as st
import fitz  # PyMuPDF
from rapidocr_onnxruntime import RapidOCR
from PIL import Image, ImageDraw
import numpy as np
import pandas as pd
import tempfile

st.set_page_config(page_title="PDF OCR BBox Viewer", layout="wide")

st.title("PDF OCR BBox Viewer")

ocr = RapidOCR()

uploaded_file = st.sidebar.file_uploader("Upload PDF", type=["pdf"])
page_number = st.sidebar.number_input("Page number", min_value=1, value=1, step=1)
zoom = st.sidebar.slider("Preview zoom", 1.0, 4.0, 2.0, 0.25)

if uploaded_file:
    pdf_bytes = uploaded_file.read()

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(pdf_bytes)
        pdf_path = tmp.name

    doc = fitz.open(pdf_path)

    if page_number > len(doc):
        st.error(f"PDF has only {len(doc)} pages.")
        st.stop()

    page = doc[page_number - 1]

    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, alpha=False)

    image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    image_np = np.array(image)

    result, _ = ocr(image_np)

    draw_img = image.copy()
    draw = ImageDraw.Draw(draw_img)

    rows = []

    if result:
        for item in result:
            bbox, text, score = item

            scaled_bbox = [(int(x), int(y)) for x, y in bbox]

            draw.polygon(
                scaled_bbox,
                outline="red",
                width=2
            )

            x_min = min(p[0] for p in scaled_bbox)
            y_min = min(p[1] for p in scaled_bbox)
            x_max = max(p[0] for p in scaled_bbox)
            y_max = max(p[1] for p in scaled_bbox)

            rows.append({
                "text": text,
                "confidence": round(float(score), 4),
                "x_min": x_min,
                "y_min": y_min,
                "x_max": x_max,
                "y_max": y_max,
                "bbox": scaled_bbox
            })

    left, right = st.columns([1, 1])

    with left:
        st.subheader("PDF Preview with Bounding Boxes")
        st.image(draw_img, use_container_width=True)

    with right:
        st.subheader("Extracted OCR Data")

        if rows:
            df = pd.DataFrame(rows)
            st.dataframe(df, use_container_width=True, height=600)

            st.subheader("Plain Text")
            st.text_area(
                "OCR Text",
                "\n".join(df["text"].tolist()),
                height=300
            )
        else:
            st.warning("No text detected.")
else:
    st.info("Upload a PDF to begin.")