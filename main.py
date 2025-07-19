# main.py - Gemini-only AI backend for Automated Underwriting Platform

import os, shutil, tempfile
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import List
import spacy
from dotenv import load_dotenv
import google.generativeai as genai

# Load .env variables
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Initialize Gemini
genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel("gemini-1.5-flash")

# Load SpaCy NLP
nlp = spacy.load("en_core_web_sm")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def extract_text_from_file(filepath: str) -> str:
    # Fallback fake OCR for demo
    return "Appraisal report for a property owned by Jane Doe located in Los Angeles."

def extract_entities(text: str):
    doc = nlp(text)
    entities = {"address": "", "owner": ""}
    for ent in doc.ents:
        if ent.label_ in ("PERSON", "ORG") and not entities["owner"]:
            entities["owner"] = ent.text
        if ent.label_ in ("GPE", "LOC", "FAC") and not entities["address"]:
            entities["address"] = ent.text
    return entities

def analyze_with_gemini(image_path: str):
    with open(image_path, "rb") as f:
        content = f.read()

    try:
        response = gemini_model.generate_content([
            "Describe any visible damage or risk in this property image (e.g., fire, cracks, broken parts, smoke, water damage, etc):",
            {"mime_type": "image/png", "data": content}
        ])
        output = response.text.lower()

        risk = 0.0
        if any(k in output for k in ["fire", "flames", "burn", "charred"]):
            risk += 0.6
        if any(k in output for k in ["smoke", "hazard", "collapse", "electrical issue"]):
            risk += 0.4
        if any(k in output for k in ["crack", "leak", "broken", "water damage", "damaged"]):
            risk += 0.3
        if any(k in output for k in ["good condition", "no visible damage", "intact"]):
            risk -= 0.2

        return min(max(risk, 0.0), 1.0), output
    except Exception as e:
        return 0.0, f"[Gemini error: {str(e)}]"

@app.post("/underwrite")
async def underwrite(
    report: UploadFile = File(...),
    images: List[UploadFile] = File(...),
    address: str = Form(...),
    owner: str = Form(...),
    propertyType: str = Form(...)
):
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Save uploaded report
        report_path = os.path.join(tmp_dir, "report_" + report.filename)
        with open(report_path, "wb") as f:
            shutil.copyfileobj(report.file, f)

        # Simulated OCR + NLP
        ocr_text = extract_text_from_file(report_path)
        entities = extract_entities(ocr_text)
        resolved_address = address or entities["address"]
        resolved_owner = owner or entities["owner"]

        image_analyses = []
        total_risk = 0.0

        for image in images:
            image_path = os.path.join(tmp_dir, image.filename)
            with open(image_path, "wb") as f:
                shutil.copyfileobj(image.file, f)

            gemini_risk, gemini_desc = analyze_with_gemini(image_path)

            total_risk += gemini_risk
            image_analyses.append({
                "filename": image.filename,
                "gemini_description": gemini_desc,
                "gemini_risk": round(gemini_risk, 2)
            })

        # Final risk score based on average
        avg_risk_score = total_risk / len(images) if images else 0.0
        avg_risk_score = min(max(avg_risk_score, 0.0), 1.0)

        risk_level = (
            "Low" if avg_risk_score < 0.33 else
            "Medium" if avg_risk_score < 0.67 else
            "High"
        )

        summary = (
            f"<b>AI Risk Level:</b> {risk_level}<br>"
            f"<b>Address:</b> {resolved_address or '[unknown]'}<br>"
            f"<b>Owner:</b> {resolved_owner or '[unknown]'}"
        )

        return JSONResponse({
            "risk_score": round(avg_risk_score, 2),
            "risk_level": risk_level,
            "property_type": propertyType,
            "address": resolved_address,
            "owner": resolved_owner,
            "image_analysis": image_analyses,
            "analysis_summary": summary
        })

# Run with: python main.py
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
