import spacy
from spacy.matcher import PhraseMatcher
import joblib
from pathlib import Path

SYNONYMS = {
    "throwing up": "vomiting",
    "puking": "vomiting",
    "can't breathe": "difficulty breathing",
    "cant breathe": "difficulty breathing",
    "trouble breathing": "difficulty breathing",
    "out of breath": "shortness of breath",
    "high temperature": "fever",
    "chest hurts": "sharp chest pain",
    "chest pain": "sharp chest pain",
    "throat hurts": "sore throat",
    "throat is sore": "sore throat",
    "can't sleep": "insomnia",
    "cant sleep": "insomnia",
    "feeling dizzy": "dizziness",
    "dizzy": "dizziness",
    "feeling weak": "weakness",
    "slurred speech": "slurring words",
}

NEGATION_CUES = {"no", "not", "without", "denies", "never", "n't", "isn't", "aren't", "doesn't", "don't"}
NEGATION_WINDOW = 4  # tokens to look back from a match for a negation cue


class SymptomExtractor:
    def __init__(self, symptom_columns_path: str, spacy_model: str = "en_core_web_sm"):
        try:
            self.nlp = spacy.load(spacy_model)
        except OSError:
            print(f"WARNING: spaCy model '{spacy_model}' not found. "
                  f"Run: python -m spacy download {spacy_model}. "
                  f"Falling back to blank English tokenizer (matching still works, "
                  f"but without deeper linguistic features).")
            self.nlp = spacy.blank("en")

        self.symptom_cols = joblib.load(symptom_columns_path)
        self.vocab_set = set(self.symptom_cols)

        self.matcher = PhraseMatcher(self.nlp.vocab, attr="LOWER")
        self._phrase_to_canonical = {}

        canonical_patterns = [self.nlp.make_doc(s) for s in self.symptom_cols]
        self.matcher.add("SYMPTOM", canonical_patterns)
        for s in self.symptom_cols:
            self._phrase_to_canonical[s] = s

        synonym_patterns = [self.nlp.make_doc(phrase) for phrase in SYNONYMS.keys()]
        self.matcher.add("SYNONYM", synonym_patterns)
        for phrase, canonical in SYNONYMS.items():
            if canonical not in self.vocab_set:
                print(f"WARNING: synonym '{phrase}' maps to '{canonical}', which is "
                      f"NOT in the model's symptom vocabulary — this synonym will be ignored.")
            self._phrase_to_canonical[phrase] = canonical

    def _is_negated(self, doc, start_token_idx: int) -> bool:
        """Check a small window of tokens before the match for a negation cue."""
        window_start = max(0, start_token_idx - NEGATION_WINDOW)
        preceding = doc[window_start:start_token_idx]
        return any(tok.lower_ in NEGATION_CUES for tok in preceding)

    def extract(self, text: str) -> dict:
        doc = self.nlp(text.lower())
        matches = self.matcher(doc)

        found = set()
        negated = set()

        for match_id, start, end in matches:
            span = doc[start:end]
            phrase = span.text
            canonical = self._phrase_to_canonical.get(phrase)
            if canonical is None or canonical not in self.vocab_set:
                continue

            if self._is_negated(doc, start):
                negated.add(canonical)
            else:
                found.add(canonical)

        found -= negated

        return {
            "extracted_symptoms": sorted(found),
            "negated_symptoms": sorted(negated),
            "raw_text": text,
        }


if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    extractor = SymptomExtractor(str(here / "symptom_columns.joblib"))

    test_cases = [
        "I have had a sharp chest pain and shortness of breath since this morning.",
        "I have a sore throat and cough but no fever.",
        "I'm throwing up and feeling really dizzy.",
        "I can't breathe properly and my chest hurts.",
    ]

    for text in test_cases:
        result = extractor.extract(text)
        print(f"\nInput: {text}")
        print(f"  Extracted: {result['extracted_symptoms']}")
        print(f"  Negated (excluded): {result['negated_symptoms']}")
