"""
Reference-Based Drug Interaction and Prescription Safety Checker.
Performs pairwise interaction analysis across confirmed medicine items.
Identifies severity tiers (Major / Severe, Moderate, Minor / Caution), biological mechanisms,
clinical risks, and action advice, accompanied by rigorous clinical safety disclaimers.

Sources:
- British National Formulary (BNF 86 Appendix 1: Interactions)
- Stockley's Drug Interactions (12th Edition)
- US FDA Drug Safety Communications
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Any


@dataclass(frozen=True)
class InteractionResult:
    drug_a: str
    drug_b: str
    severity: str  # 'Major / Severe', 'Moderate', 'Minor / Caution'
    risk_summary: str
    clinical_mechanism: str
    management_advice: str
    evidence_source: str


# Authoritative pairwise interaction database
INTERACTION_DATABASE: list[InteractionResult] = [
    InteractionResult(
        drug_a="propranolol",
        drug_b="clonazepam",
        severity="Moderate",
        risk_summary="Enhanced central nervous system depression and increased risk of hypotension or severe bradycardia.",
        clinical_mechanism="Additive pharmacodynamic sedative effects on the central nervous system combined with beta-blocker induced reduction in heart rate and systemic blood pressure.",
        management_advice="Monitor blood pressure, heart rate, and level of consciousness. Avoid driving or operating machinery. Dosage adjustment may be required by the physician.",
        evidence_source="British National Formulary (BNF 86), Stockley's Drug Interactions",
    ),
    InteractionResult(
        drug_a="losartan",
        drug_b="propranolol",
        severity="Moderate",
        risk_summary="Enhanced hypotensive response and potential risk of symptomatic orthostatic hypotension or bradycardia.",
        clinical_mechanism="Concurrent dual anti-hypertensive blockade through complementary renin-angiotensin-aldosterone (RAAS) suppression and adrenergic beta-blockade.",
        management_advice="Initiate at lower dosages, monitor standing and sitting blood pressure and serum potassium levels regularly.",
        evidence_source="AHA/ACC Hypertension Guidelines, BNF 86",
    ),
    InteractionResult(
        drug_a="metformin",
        drug_b="propranolol",
        severity="Moderate",
        risk_summary="Masking of hypoglycemia warning symptoms and altered glycemic control.",
        clinical_mechanism="Beta-blockers blunt the adrenergic tachycardia, tremor, and palpitations that normally alert diabetic patients to falling blood glucose, while also inhibiting glycogenolysis.",
        management_advice="Instruct patient that diaphoresis (sweating) typically remains unmasked during hypoglycemia. Frequent blood glucose monitoring advised.",
        evidence_source="American Diabetes Association (ADA) Standards of Care, BNF 86",
    ),
    InteractionResult(
        drug_a="azithromycin",
        drug_b="propranolol",
        severity="Moderate",
        risk_summary="Potential additive cardiovascular stress; risk of QT interval prolongation and conduction disturbance.",
        clinical_mechanism="Azithromycin can prolong cardiac repolarization, while propranolol decreases heart rate, potentially creating favorable electrophysiologic conditions for arrhythmia in susceptible patients.",
        management_advice="Monitor baseline ECG in patients with underlying cardiac conduction disorders or electrolyte imbalances.",
        evidence_source="CredibleMeds QT Database, FDA Drug Safety Communication",
    ),
    InteractionResult(
        drug_a="omeprazole",
        drug_b="clopidogrel",
        severity="Major / Severe",
        risk_summary="Reduced antiplatelet efficacy of clopidogrel, increasing thrombotic event and re-infarction risk.",
        clinical_mechanism="Omeprazole competitively inhibits hepatic CYP2C19, preventing the metabolic bioactivation of clopidogrel prodrug into its active thiol metabolite.",
        management_advice="Avoid combination. Use an alternative gastroprotective agent with lower CYP2C19 inhibitory affinity (e.g., Pantoprazole) under clinician direction.",
        evidence_source="FDA Drug Safety Boxed Warning, European Medicines Agency (EMA)",
    ),
    InteractionResult(
        drug_a="paracetamol",
        drug_b="dexibuprofen",
        severity="Moderate",
        risk_summary="Dual analgesic overlap; increased risk of inadvertent renal or hepatic burden if overdosed.",
        clinical_mechanism="Combining an NSAID (Dexibuprofen) with Paracetamol provides complementary peripheral and central analgesia but increases exposure to combined toxicities without strict dose caps.",
        management_advice="Ensure paracetamol dose does not exceed 4g/day and monitor renal function, especially in elderly or dehydrated patients.",
        evidence_source="BNF 86 Analgesic Guidelines",
    ),
    InteractionResult(
        drug_a="losartan",
        drug_b="dexibuprofen",
        severity="Major / Severe",
        risk_summary="Decreased antihypertensive effect and sharply increased risk of acute kidney injury (AKI) and hyperkalemia.",
        clinical_mechanism="NSAIDs inhibit renal prostaglandin synthesis causing afferent arteriolar vasoconstriction, while ARBs cause efferent arteriolar vasodilation, precipitating a steep drop in glomerular filtration rate.",
        management_advice="Avoid chronic concurrent use. Check serum creatinine, eGFR, and potassium within 1-2 weeks of concurrent therapy.",
        evidence_source="Kidney Disease: Improving Global Outcomes (KDIGO) AKI Guidelines",
    ),
    InteractionResult(
        drug_a="aspirin",
        drug_b="clopidogrel",
        severity="Moderate",
        risk_summary="Increased risk of major hemorrhage and gastrointestinal bleeding with dual antiplatelet therapy (DAPT).",
        clinical_mechanism="Additive inhibition of platelet aggregation via irreversible COX-1 inactivation by aspirin combined with P2Y12 adenosine diphosphate receptor antagonism by clopidogrel.",
        management_advice="Only use concurrently when specifically prescribed for approved post-coronary stent or acute coronary syndrome indications under clinician supervision. Consider gastroprotection.",
        evidence_source="ACC/AHA Guideline on Duration of Dual Antiplatelet Therapy, BNF 86",
    ),
    InteractionResult(
        drug_a="aspirin",
        drug_b="dexibuprofen",
        severity="Major / Severe",
        risk_summary="Loss of aspirin cardioprotective antiplatelet effect and drastically increased gastrointestinal ulceration risk.",
        clinical_mechanism="Dexibuprofen competitively impedes aspirin's access to the Ser529 active site of platelet COX-1, attenuating irreversible platelet inhibition, while compounding gastric mucosal erosion.",
        management_advice="Avoid concurrent administration. If both are necessary, take immediate-release aspirin at least 30 minutes before or 8 hours after the NSAID.",
        evidence_source="US FDA Drug Safety Alert, BNF 86",
    ),
    InteractionResult(
        drug_a="escitalopram",
        drug_b="aspirin",
        severity="Moderate",
        risk_summary="Substantially elevated risk of upper gastrointestinal bleeding.",
        clinical_mechanism="SSRIs deplete platelet intra-cellular serotonin storage necessary for hemostatic plug formation, potentiating aspirin-induced mucosal injury and platelet inhibition.",
        management_advice="Monitor for black tarry stools or unexplained bruising. Consider co-prescribing a proton pump inhibitor for gastroprotection.",
        evidence_source="Stockley's Drug Interactions, BNF 86",
    ),
    InteractionResult(
        drug_a="escitalopram",
        drug_b="clonazepam",
        severity="Moderate",
        risk_summary="Enhanced central nervous system sedation, psychomotor impairment, and dizziness.",
        clinical_mechanism="Additive pharmacodynamic central nervous system depressive activity on GABAergic and serotonergic neurotransmission pathways.",
        management_advice="Advise patient to avoid alcohol and exercise caution when driving or operating machinery. Dosage should be monitored by prescriber.",
        evidence_source="British National Formulary (BNF 86)",
    ),
    InteractionResult(
        drug_a="cefixime",
        drug_b="propranolol",
        severity="Minor / Caution",
        risk_summary="Minor gastrointestinal disturbances; no clinically significant pharmacokinetic interference.",
        clinical_mechanism="Separate elimination pathways; oral cephalosporins may occasionally alter bowel flora causing mild diarrhea.",
        management_advice="Standard clinical administration; take doses at recommended intervals.",
        evidence_source="Lexicomp Drug Information",
    ),
]


def _canonical_name(name: str) -> str:
    """Simplifies brand or generic name to its matching interaction key."""
    clean = name.strip().lower()
    mapping = {
        "napa": "paracetamol",
        "calpol": "paracetamol",
        "tylenol": "paracetamol",
        "acetaminophen": "paracetamol",
        "zimax": "azithromycin",
        "indever": "propranolol",
        "inderen": "propranolol",
        "rivotril": "clonazepam",
        "losita": "losartan",
        "comet": "metformin",
        "exeptim": "cefixime",
        "dexter": "dexibuprofen",
        "ome": "omeprazole",
        "prilosec": "omeprazole",
        "escilex": "escitalopram",
        "nexcital": "escitalopram",
        "asynta": "aspirin",
        "m-lucas": "montelukast",
        "singulair": "montelukast",
    }
    return mapping.get(clean, clean)


def check_drug_interactions(medicine_names: list[str]) -> dict[str, Any]:
    """
    Checks all pairwise combinations of verified prescription medicines against the interaction database.
    Returns detected interactions, severity levels, and comprehensive clinical disclaimers.

    SAFETY MANDATE:
    Never claims a prescription is medically safe solely because no interaction was found in the reference set.
    """
    normalized_list = [_canonical_name(name) for name in medicine_names if name.strip()]
    unique_medicines = list(dict.fromkeys(normalized_list))

    found_interactions: list[dict[str, Any]] = []
    checked_pairs: list[tuple[str, str]] = []

    if len(unique_medicines) >= 2:
        for drug_1, drug_2 in combinations(unique_medicines, 2):
            checked_pairs.append((drug_1.title(), drug_2.title()))
            for item in INTERACTION_DATABASE:
                matches = (
                    (item.drug_a == drug_1 and item.drug_b == drug_2)
                    or (item.drug_a == drug_2 and item.drug_b == drug_1)
                )
                if matches:
                    found_interactions.append({
                        "drug_a": item.drug_a.title(),
                        "drug_b": item.drug_b.title(),
                        "severity": item.severity,
                        "risk_summary": item.risk_summary,
                        "clinical_mechanism": item.clinical_mechanism,
                        "management_advice": item.management_advice,
                        "evidence_source": item.evidence_source,
                    })

    has_major = any(i["severity"].startswith("Major") for i in found_interactions)

    return {
        "checked_medicines": [m.title() for m in unique_medicines],
        "checked_pairs_count": len(checked_pairs),
        "interaction_count": len(found_interactions),
        "interactions": found_interactions,
        "has_major_alert": has_major,
        "overall_status": (
            "Major Alert Detected"
            if has_major
            else ("Interactions Noted" if found_interactions else "No Documented Interactions in Reference Set")
        ),
        "safety_disclaimer": (
            "CRITICAL SAFETY NOTE: This interaction screening is derived from reference clinical databases. "
            "A result of 'No Documented Interactions in Reference Set' DOES NOT prove that this combination is medically safe. "
            "Uncatalogued pharmacodynamic synergies, patient-specific renal/hepatic clearance, age, comorbidities, "
            "and over-the-counter supplements must be evaluated by a qualified doctor or pharmacist before taking multiple medications."
        ),
    }
