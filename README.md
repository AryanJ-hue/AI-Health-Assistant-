# AI Health Assistant

An explainable, risk-aware symptom-checker prototype built end to end from raw dataset to a working local web app, as a university project.
Not a diagnostic tool. A prototype demonstrating how NLP extraction, machine learning, and an independently-validated safety layer can combine into a transparent clinical decision-support system.

## What it does

- **Free-text or checklist symptom input** — describe symptoms naturally, or select from a searchable list
- **ML-based disease prediction** — ranked probable conditions with real confidence scores
- **Explainability** — shows exactly which symptoms drove each prediction, and by how much (coefficient-based, not a black box)
- **Independent risk engine** — a separate, rule-based safety layer flags emergency symptom patterns regardless of what the ML model predicts
- **Hindi language support** — offline translation layer for Devanagari-script Hindi input

## How it works
```
Text / Checklist Input
        │
        ├──► Translation Layer (Hindi → English, offline)
        │
        ├──► NLP Extraction (spaCy PhraseMatcher + negation handling)
        │
        ├──► ML Prediction (Logistic Regression, 413 disease classes)
        │         │
        │         └──► Explainability (coefficient-based contribution scores)
        │
        └──► Risk Engine (independent rule-based red-flag detection)
```
The risk engine runs in parallel with the ML model, not after it — a confident-but-wrong prediction can never suppress a safety flag, and an uncertain prediction never gets treated as "safe by default."

## Results

Trained and compared three classical baselines on a cleaned, filtered dataset (773 → 413 disease classes, ~184K rows, after removing classes with fewer than 50 samples):
| Model | Test Macro-F1 |
|---|---|
| **Logistic Regression** | **0.825** |
| Decision Tree | 0.732 |
| Random Forest | 0.690 |
5-fold stratified cross-validation on the winning model: **0.8252 ± 0.0007** — stable across folds, not a lucky split.
Failure analysis identified two concrete, explainable error patterns: symptom overlap among dermatological conditions, and precision collapse in low-support disease classes (worth a read in the full project writeup).

## Tech stack

- **Backend:** FastAPI (Python)
- **ML:** scikit-learn
- **NLP:** spaCy (PhraseMatcher-based extraction)
- **Translation:** argos-translate (fully offline after initial model download)
- **Frontend:** vanilla HTML/CSS/JS
- **Dataset:** [Diseases and Symptoms](https://www.kaggle.com/) (Kaggle)
## Running it locally
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# train the baseline model (optional — a trained model is already included)
python disease_symptom_baseline.py

# start the server
uvicorn main:app --reload
```
Then open `http://127.0.0.1:8000`.

## Known limitations

- Risk engine rules are a **draft set**, explicitly flagged as needing clinical/expert review before any real-world use
- NLP extraction is phrase-matching based — sensitive to word order and phrasing variation
- Hindi support currently handles Devanagari script reliably; romanized Hinglish is a known gap
- This is a prototype for academic evaluation, not a validated medical product

## Team
Built as a 6-person college project — dataset & baseline modeling, NLP extraction, risk engine, backend, and frontend split across the team.

Aryan- Integration Owner- lead of the project, worked on all the models included in the project.
Deepika- Backend/API- worked on the backend/api part of the project, integrating everything to make it a runnable model.
Surbhi-Frontend-Worked on the frontend part of the project, creating a structured, simple and well explained interface for the model.
Rohit-Risk Engine-worked on the risk engine, provides red flags or basic assessment levels for the symptoms to know if immediate care is necessary or not.
Sejal- Dataset/Baseline models- Worked on listing out the dataset and sorting it out. Also, ran the baseline models to see whicbh gives the best results for the preferred datasets.
Shivani-NLP Extraction- Worked on the NLP part which extracts the symptoms through phrases and assess them.

*Academic project — not intended for real-world medical use without proper clinical validation.*
