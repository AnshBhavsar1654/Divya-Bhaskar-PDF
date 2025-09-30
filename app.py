import os
import io
import re
import json
import time
import uuid
from typing import List

from fastapi import FastAPI, File, UploadFile, Form, Request
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from PIL import Image
from pdf2image import convert_from_bytes
import pandas as pd
import requests

import google.generativeai as genai
from dotenv import load_dotenv

# ------------------ Config ------------------
load_dotenv()

START_TIME = None

# Prefer .env, otherwise fall back to the provided key (user supplied)
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
POPPLER_PATH = os.getenv("POPPLER_PATH")

if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")
STATIC_DIR = os.path.join(BASE_DIR, "static")
OUTPUT_CSV = os.path.join(BASE_DIR, "epaper_output.csv")

# Ensure folders
os.makedirs(TEMPLATES_DIR, exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)

app = FastAPI(
    title="Gujarati News Extractor",
    description="Extract articles from Gujarati newspaper PDFs using AI",
    version="1.0.0"
)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
templates = Jinja2Templates(directory=TEMPLATES_DIR)


# ------------------ Utility Functions ------------------
def load_image(image_path_or_url: str) -> Image.Image:
    """Load an image from URL or local path and return PIL Image."""
    if image_path_or_url.startswith("http"):
        resp = requests.get(image_path_or_url, timeout=60)
        resp.raise_for_status()
        return Image.open(io.BytesIO(resp.content)).convert("RGB")
    return Image.open(image_path_or_url).convert("RGB")


def extract_articles_with_gemini(image: Image.Image) -> list:
    """Send the page image to Gemini and ask for Gujarati headline/content pairs.
    Returns a list of dicts: [{"headline": str, "content": str, "city": str, "district": str, "sentiment": str}, ...]
    """
    if not GOOGLE_API_KEY:
        print("GOOGLE_API_KEY not set. Set env var or insert your key in the file.")
        return []

    prompt = (
        "You are an expert Gujarati news analyst. Given a full-page Gujarati newspaper image, "
        "visually read the ENTIRE page and identify ALL distinct news articles (ignore advertisements, page headers/footers, and tiny blurbs). "
        "For each real article, return the following in Gujarati where applicable: \n"
        "- headline: concise Gujarati headline \n"
        "- content: short Gujarati summary(40-50 words)\n"
        "- city: the city/location mentioned for the article (Gujarati). If none is visible, return \"\".\n"
        "- district: the district for that city/location (Gujarati). If uncertain, infer reasonably; else return \"\".\n"
        "- sentiment: one of these Gujarati labels based on article tone: \"સકારાત્મક\" (positive), \"તટસ્થ\" (neutral), or \"નકારાત્મક\" (negative).\n"
        "Output only valid JSON with EXACT schema: {\"articles\":[{\"headline\":\"...\",\"content\":\"...\",\"city\":\"...\",\"district\":\"...\",\"sentiment\":\"સકારાત્મક|તટસ્થ|નકારાત્મક\"}, ...]}"
    )

    model = genai.GenerativeModel(GEMINI_MODEL)
    try:
        # Convert PIL image to bytes for upload to Gemini
        img_bytes = io.BytesIO()
        image.save(img_bytes, format="PNG")
        img_bytes.seek(0)
        # google-generativeai supports passing PIL Image directly in many cases,
        # but to be robust we send as a dict with mime_type
        resp = model.generate_content([
            prompt,
            {
                "mime_type": "image/png",
                "data": img_bytes.getvalue(),
            },
        ])
        out_text = getattr(resp, "text", "")
        if not out_text:
            return []

        # Extract JSON
        m = re.search(r"\{.*\}", out_text, re.DOTALL)
        if m:
            out_text = m.group(0)
        data = json.loads(out_text)
        articles = data.get("articles", [])
        # Normalize
        norm = []
        for a in articles:
            h = str(a.get("headline", "")).strip()
            c = str(a.get("content", "")).strip()
            city = str(a.get("city", "")).strip()
            district = str(a.get("district", "")).strip()
            sentiment = str(a.get("sentiment", "")).strip()
            if h or c:
                norm.append({
                    "headline": h,
                    "content": c,
                    "city": city,
                    "district": district,
                    "sentiment": sentiment,
                })
        return norm
    except Exception as e:
        print(f"Gemini error: {e}")
        return []


def pdf_pages_to_images(file_bytes: bytes, dpi: int = 200) -> List[Image.Image]:
    """Convert PDF bytes to a list of PIL Images, one per page."""
    # Note: On Windows, pdf2image requires Poppler. Install it and set POPPLER_PATH if needed.
    # You can set environment variable POPPLER_PATH to the bin folder of poppler.
    if POPPLER_PATH:
        images = convert_from_bytes(file_bytes, dpi=dpi, poppler_path=POPPLER_PATH)
    else:
        images = convert_from_bytes(file_bytes, dpi=dpi)
    out = [im.convert("RGB") for im in images]
    return out


# ------------------ Web Routes ------------------
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/process", response_class=HTMLResponse)
async def process(
    request: Request,
    date: str = Form(...),
    newspaper_name: str = Form(...),
    media_link: str = Form("") ,
    files: List[UploadFile] = File(...),
):
    global START_TIME
    START_TIME = time.time()

    rows = []
    total_pages = 0
    total_articles = 0

    for f in files:
        try:
            content = await f.read()
        finally:
            await f.close()

        # Convert PDF to images
        try:
            page_images = pdf_pages_to_images(content)
        except Exception as e:
            # Skip file on error
            print(f"Failed converting {f.filename}: {e}")
            continue

        # Process each page
        for idx, page_img in enumerate(page_images, start=1):
            total_pages += 1
            articles = extract_articles_with_gemini(page_img)
            # Aggregate rows
            for a in articles:
                total_articles += 1
                rows.append([
                    date,  # date
                    idx,   # page number (within this PDF)
                    a.get("city", ""),
                    a.get("district", ""),
                    media_link,
                    newspaper_name,
                    a.get("headline", ""),
                    a.get("content", ""),
                    a.get("sentiment", ""),
                ])

    # Ensure final DataFrame has requested columns in order
    columns = [
        "date",
        "page number",
        "city",
        "district",
        "media link",
        "news paper name",
        "headline",
        "content",
        "sentiment",
    ]
    df = pd.DataFrame(rows, columns=columns)
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

    elapsed = None
    if START_TIME is not None:
        elapsed = time.time() - START_TIME
        print(f"Total execution time: {elapsed:.2f} seconds")

    print(f"\n✓ Data saved to {OUTPUT_CSV}")
    print(f"Total records: {len(df)}")
    try:
        print(df.head())
    except Exception:
        pass

    # Render result page
    return templates.TemplateResponse(
        "result.html",
        {
            "request": request,
            "records": len(df),
            "pages": total_pages,
            "articles": total_articles,
            "csv_name": os.path.basename(OUTPUT_CSV),
            "elapsed": f"{elapsed:.2f} s" if elapsed is not None else None,
        },
    )


@app.get("/download/{filename}")
async def download_csv(filename: str):
    file_path = os.path.join(BASE_DIR, filename)
    if os.path.isfile(file_path):
        return FileResponse(file_path, filename=filename, media_type="text/csv")
    return HTMLResponse(content="File not found", status_code=404)


# Optional: enable local run via `python app.py`
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)
