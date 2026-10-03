import joblib
from pathlib import Path
from typing import Literal
RiskLevel = Literal["emergency", "moderate", "low"]
RISK_RULES = [
    {
        "rule_id": "RF-001",
        "name": "Acute cardiac/respiratory distress",
        "trigger_symptoms": {"sharp chest pain", "shortness of breath"},
        "match_type": "any",  # any ONE of these present triggers it
        "risk_level": "emergency",
        "rationale": "Chest pain combined with breathing difficulty is a "
                      "classic ACS/PE presentation pattern requiring immediate care.",
        "source": "General ER triage convention (needs clinical review)",
        "review_status": "DRAFT - NEEDS CLINICAL REVIEW",
    },
    {
        "rule_id": "RF-002",
        "name": "Breathing difficulty combination",
        "trigger_symptoms": {"difficulty breathing", "breathing fast", "chest tightness"},
        "match_type": "any",
        "risk_level": "emergency",
        "rationale": "Any acute breathing impairment symptom warrants urgent evaluation "
                      "regardless of suspected cause.",
        "source": "General ER triage convention (needs clinical review)",
        "review_status": "DRAFT - NEEDS CLINICAL REVIEW",
    },
    {
        "rule_id": "RF-003",
        "name": "Neurological emergency signs",
        "trigger_symptoms": {"slurring words", "focal weakness", "double vision"},
        "match_type": "any",
        "risk_level": "emergency",
        "rationale": "Sudden focal weakness, slurred speech, or double vision are "
                      "classic stroke (FAST) warning signs.",
        "source": "Stroke FAST criteria concept (needs clinical review — verify "
                   "against actual FAST/BEFAST guideline before relying on this)",
        "review_status": "DRAFT - NEEDS CLINICAL REVIEW",
    },
    {
        "rule_id": "RF-004",
        "name": "Seizure activity",
        "trigger_symptoms": {"seizures"},
        "match_type": "any",
        "risk_level": "emergency",
        "rationale": "Any reported seizure activity requires urgent evaluation.",
        "source": "General clinical convention (needs clinical review)",
        "review_status": "DRAFT - NEEDS CLINICAL REVIEW",
    },
    {
        "rule_id": "RF-005",
        "name": "Significant bleeding",
        "trigger_symptoms": {"vomiting blood", "blood in stool", "rectal bleeding"},
        "match_type": "any",
        "risk_level": "emergency",
        "rationale": "Visible GI bleeding (hematemesis/melena/hematochezia) is a "
                      "recognized emergency presentation.",
        "source": "General clinical convention (needs clinical review)",
        "review_status": "DRAFT - NEEDS CLINICAL REVIEW",
    },
    {
        "rule_id": "RF-006",
        "name": "High fever with systemic signs",
        "trigger_symptoms": {"fever", "vomiting"},
        "match_type": "all", 
        "risk_level": "moderate",
        "rationale": "Fever combined with vomiting suggests possible systemic "
                      "infection needing prompt (not necessarily emergency) care.",
        "source": "General clinical convention (needs clinical review)",
        "review_status": "DRAFT - NEEDS CLINICAL REVIEW",
    },
    {
        "rule_id": "RF-007",
        "name": "Progressive weakness",
        "trigger_symptoms": {"weakness", "muscle weakness"},
        "match_type": "any",
        "risk_level": "moderate",
        "rationale": "General weakness alone is nonspecific but warrants prompt "
                      "professional consultation, distinct from focal weakness (RF-003) "
                      "which is treated as emergency.",
        "source": "General clinical convention (needs clinical review)",
        "review_status": "DRAFT - NEEDS CLINICAL REVIEW",
    },
]

RULES_VERSION = "risk_rules_v0.1_draft"
def validate_rules(symptom_vocab: set[str]) -> list[str]:
    problems = []
    seen_ids = set()
    for rule in RISK_RULES:
        if rule["rule_id"] in seen_ids:
            problems.append(f"Duplicate rule_id: {rule['rule_id']}")
        seen_ids.add(rule["rule_id"])

        unknown = rule["trigger_symptoms"] - symptom_vocab
        if unknown:
            problems.append(
                f"{rule['rule_id']} ({rule['name']}): symptoms not in vocabulary: {unknown}"
            )
    return problems
def assess_risk(symptoms: list[str]) -> dict:
    symptom_set = set(symptoms)
    triggered = []

    level_rank = {"low": 0, "moderate": 1, "emergency": 2}
    highest_level: RiskLevel = "low"

    for rule in RISK_RULES:
        if rule["match_type"] == "any":
            hit = bool(rule["trigger_symptoms"] & symptom_set)
        else:  # "all"
            hit = rule["trigger_symptoms"].issubset(symptom_set)

        if hit:
            triggered.append({
                "rule_id": rule["rule_id"],
                "name": rule["name"],
                "risk_level": rule["risk_level"],
                "rationale": rule["rationale"],
                "review_status": rule["review_status"],
            })
            if level_rank[rule["risk_level"]] > level_rank[highest_level]:
                highest_level = rule["risk_level"]

    guidance = {
        "emergency": "One or more red-flag symptoms were detected. Seek urgent/emergency "
                      "medical care. This assessment does not replace professional judgment.",
        "moderate": "Some concerning symptoms were detected. Prompt professional "
                    "consultation is recommended.",
        "low": "No red-flag patterns detected in this draft rule set. This does NOT "
               "mean the situation is safe — only that it doesn't match the current "
               "(incomplete, unreviewed) rule table. When in doubt, consult a professional.",
    }

    return {
        "risk_level": highest_level,
        "triggered_rules": triggered,
        "guidance": guidance[highest_level],
        "rules_version": RULES_VERSION,
    }


if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    symptom_cols = joblib.load(here / "symptom_columns.joblib")
    problems = validate_rules(set(symptom_cols))
    if problems:
        print("RULE VALIDATION ISSUES:")
        for p in problems:
            print(f"  - {p}")
    else:
        print(f"All {len(RISK_RULES)} rules validated OK against {len(symptom_cols)} symptoms.")

    print("\n--- Test case: chest pain + shortness of breath ---")
    result = assess_risk(["sharp chest pain", "shortness of breath"])
    print(result)

    print("\n--- Test case: mild, no red flags ---")
    result = assess_risk(["sore throat", "cough"])
    print(result)
