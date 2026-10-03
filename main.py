
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from nlp_extractor import SymptomExtractor
from risk_engine import assess_risk, validate_rules
from translation_layer import TranslationLayer, ensure_hindi_model_installed
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "best_model.joblib"
SYMPTOMS_PATH = BASE_DIR / "symptom_columns.joblib"
STATIC_DIR = BASE_DIR / "static"

MODEL_VERSION = "logreg_baseline_v1_413classes"
TRAIN_MACRO_F1 = 0.8252  

app = FastAPI(title="AI Health Assistant")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

model = joblib.load(MODEL_PATH)
symptom_cols = joblib.load(SYMPTOMS_PATH)
symptom_set = set(symptom_cols)
extractor = SymptomExtractor(str(SYMPTOMS_PATH))
translator = TranslationLayer()
try:
    ensure_hindi_model_installed()
except Exception as e:
    print(f"WARNING: Hindi model setup failed at startup ({e}). "
          f"Hindi input will pass through untranslated.")

rule_problems = validate_rules(symptom_set)
if rule_problems:
    print("RISK RULE WARNINGS:", rule_problems)

class PredictRequest(BaseModel):
    symptoms: list[str] = Field(..., description="List of symptom names, must match /symptoms exactly")
    top_k: int = Field(5, ge=1, le=20)


class SymptomContribution(BaseModel):
    symptom: str
    weight: float


class PredictionEntry(BaseModel):
    disease: str
    score: float
    contributing_symptoms: list[SymptomContribution]


class PredictResponse(BaseModel):
    predictions: list[PredictionEntry]
    unrecognized_symptoms: list[str]
    model_version: str
    model_macro_f1: float

@app.get("/symptoms")
def get_symptoms():
    """Returns the full symptom vocabulary the model was trained on,
    so the frontend can render a checklist without hardcoding it."""
    return {"symptoms": sorted(symptom_cols)}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if not req.symptoms:
        raise HTTPException(status_code=400, detail="Select at least one symptom.")

    recognized = [s for s in req.symptoms if s in symptom_set]
    unrecognized = [s for s in req.symptoms if s not in symptom_set]

    if not recognized:
        raise HTTPException(
            status_code=400,
            detail="None of the submitted symptoms match the model's vocabulary."
        )
    vec = pd.DataFrame(np.zeros((1, len(symptom_cols))), columns=symptom_cols)
    for s in recognized:
        vec.at[0, s] = 1

    proba = model.predict_proba(vec)[0]
    top_idx = np.argsort(proba)[::-1][: req.top_k]

    input_vec = vec.values[0]
    results = []
    for idx in top_idx:
        class_coefs = model.coef_[idx]
        contributions = class_coefs * input_vec  # zero for absent symptoms
        present_idx = [i for i in range(len(symptom_cols)) if input_vec[i] == 1]
        present_idx.sort(key=lambda i: contributions[i], reverse=True)

        top_contribs = [
            SymptomContribution(symptom=symptom_cols[i], weight=round(float(contributions[i]), 4))
            for i in present_idx[:5]
        ]

        results.append(PredictionEntry(
            disease=model.classes_[idx],
            score=round(float(proba[idx]), 4),
            contributing_symptoms=top_contribs,
        ))

    return PredictResponse(
        predictions=results,
        unrecognized_symptoms=unrecognized,
        model_version=MODEL_VERSION,
        model_macro_f1=TRAIN_MACRO_F1,
    )
class AnalyzeRequest(BaseModel):
    text: str
    top_k: int = Field(5, ge=1, le=20)


@app.post("/analyze")
def analyze(req: AnalyzeRequest):
    translation_result = translator.process(req.text)
    text_for_extraction = translation_result["translated_text"]
    extraction = extractor.extract(text_for_extraction)
    extracted_symptoms= extraction["extracted_symptoms"]
    risk_result = assess_risk(extracted_symptoms)
    predict_result = predict(PredictRequest(symptoms=extracted_symptoms, top_k=req.top_k))

    return {
        "input_text": req.text,
        "detected_language": translation_result["detected_language"],
        "translated_text": translation_result["translated_text"],
        "extracted_symptoms": extracted_symptoms,
        "negated_symptoms": extraction["negated_symptoms"],
        "risk_assessment": risk_result,
        "predictions": predict_result.predictions,
        "model_version": predict_result.model_version,
    }
app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")
