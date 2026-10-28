"""
Prescription Information Extraction and Medicine Normalization Engine.
Extracts clinical prescription entities (medicine, strength, form, frequency, duration,
route, instructions, prescriber, date) and matches medicine names against an authoritative
reference dictionary using fuzzy similarity scoring with human-in-the-loop verification.

IMPORTANT SAFETY CONSTRAINTS:
- Never silently replace an unclear medicine name with a guessed name.
- Do NOT infer missing dosages, frequencies, or treatment durations.
- Ambiguous medicine names must display the original OCR text and offer possible matches
  supported by the reference data for manual human verification.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from difflib import SequenceMatcher
from typing import Any


@dataclass
class ExtractedPrescriptionItem:
    raw_text: str
    medicine_name: str
    normalized_name: str | None
    generic_name: str | None
    match_confidence: float
    status: str  # 'Verified Exact', 'Likely Match', 'Uncertain / Review Needed'
    dosage_form: str
    strength: str
    frequency: str
    duration: str
    route: str
    instructions: str
    candidate_matches: list[dict[str, Any]] = field(default_factory=list)
    user_confirmed: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PrescriptionMetadata:
    prescriber_name: str | None = None
    prescriber_credentials: str | None = None
    registration_number: str | None = None
    clinic_or_hospital: str | None = None
    prescriber_contact: str | None = None
    clinic_address: str | None = None
    prescription_date: str | None = None
    patient_name: str | None = None
    patient_sex: str | None = None
    patient_age: str | None = None
    opd_number: str | None = None
    vitals: list[str] = field(default_factory=list)
    diagnosis: str | None = None
    advice: list[str] = field(default_factory=list)
    follow_up: str | None = None
    patient_info: str | None = None


MEDICINE_REFERENCE_DB: dict[str, dict[str, Any]] = {
    # RxHandBD-ML Dataset Medicines
    "napa": {"generic": "Paracetamol", "class": "Analgesic / Antipyretic", "common_forms": ["Tablet", "Syrup", "Suppository"], "common_strengths": ["500mg", "120mg/5ml", "60ml"]},
    "rivotril": {"generic": "Clonazepam", "class": "Anticonvulsant / Anxiolytic (Benzodiazepine)", "common_forms": ["Tablet"], "common_strengths": ["0.5mg", "1mg", "2mg"]},
    "algin": {"generic": "Sodium Alginate + Magnesium Carbonate", "class": "Antacid / Reflux Suppressant", "common_forms": ["Suspension", "Tablet"], "common_strengths": ["200ml", "500mg"]},
    "losita": {"generic": "Losartan Potassium", "class": "Antihypertensive (ARB)", "common_forms": ["Tablet"], "common_strengths": ["25mg", "50mg", "100mg"]},
    "indever": {"generic": "Propranolol Hydrochloride", "class": "Non-selective Beta-blocker", "common_forms": ["Tablet"], "common_strengths": ["10mg", "40mg"]},
    "inderen": {"generic": "Propranolol Hydrochloride", "class": "Non-selective Beta-blocker", "common_forms": ["Tablet"], "common_strengths": ["10mg", "40mg"]},
    "dexter": {"generic": "Dexibuprofen", "class": "NSAID", "common_forms": ["Tablet"], "common_strengths": ["200mg", "300mg", "400mg"]},
    "zimax": {"generic": "Azithromycin", "class": "Macrolide Antibiotic", "common_forms": ["Capsule", "Suspension", "Tablet"], "common_strengths": ["250mg", "500mg", "200mg/5ml"]},
    "econate": {"generic": "Econazole Nitrate", "class": "Antifungal", "common_forms": ["Cream", "Ointment"], "common_strengths": ["1%", "10g", "20g"]},
    "exeptim": {"generic": "Cefixime", "class": "3rd Gen Cephalosporin Antibiotic", "common_forms": ["Capsule", "Suspension"], "common_strengths": ["200mg", "400mg", "100mg/5ml"]},
    "kacin": {"generic": "Amikacin", "class": "Aminoglycoside Antibiotic", "common_forms": ["Injection"], "common_strengths": ["100mg", "250mg", "500mg"]},
    "lubilay": {"generic": "Lubiprostone", "class": "Chloride Channel Activator", "common_forms": ["Capsule"], "common_strengths": ["8mcg", "24mcg"]},
    "hulilax": {"generic": "Lubiprostone", "class": "Chloride Channel Activator", "common_forms": ["Capsule"], "common_strengths": ["24mcg"]},
    "t-cef": {"generic": "Ceftriaxone", "class": "3rd Gen Cephalosporin Antibiotic", "common_forms": ["Injection"], "common_strengths": ["250mg", "500mg", "1g", "2g"]},
    "escilex": {"generic": "Escitalopram", "class": "Antidepressant (SSRI)", "common_forms": ["Tablet"], "common_strengths": ["5mg", "10mg", "20mg"]},
    "traxy": {"generic": "Tranexamic Acid", "class": "Antifibrinolytic Hemostatic", "common_forms": ["Capsule", "Injection"], "common_strengths": ["500mg"]},
    "setra": {"generic": "Sertraline", "class": "Antidepressant (SSRI)", "common_forms": ["Tablet"], "common_strengths": ["25mg", "50mg", "100mg"]},
    "hpr ds": {"generic": "Hydroxyzine Pamoate", "class": "First-generation Antihistamine", "common_forms": ["Tablet", "Syrup"], "common_strengths": ["25mg", "50mg"]},
    "delfian": {"generic": "Deflazacort", "class": "Corticosteroid", "common_forms": ["Tablet"], "common_strengths": ["6mg", "18mg", "30mg"]},
    "clacef": {"generic": "Cefaclor", "class": "2nd Gen Cephalosporin Antibiotic", "common_forms": ["Capsule", "Suspension"], "common_strengths": ["250mg", "500mg", "125mg/5ml"]},
    "paloxi": {"generic": "Palonosetron", "class": "Antiemetic (5-HT3 Antagonist)", "common_forms": ["Tablet", "Injection"], "common_strengths": ["0.5mg", "0.25mg/5ml"]},
    "dominos": {"generic": "Domperidone", "class": "Dopamine D2 Receptor Antagonist (Prokinetic)", "common_forms": ["Tablet", "Suspension"], "common_strengths": ["10mg"]},
    "m-lucas": {"generic": "Montelukast Sodium", "class": "Leukotriene Receptor Antagonist", "common_forms": ["Tablet", "Chewable Tablet"], "common_strengths": ["4mg", "5mg", "10mg"]},
    "clavusef": {"generic": "Cefuroxime Axetil + Clavulanic Acid", "class": "Cephalosporin + Beta-lactamase Inhibitor", "common_forms": ["Tablet"], "common_strengths": ["250mg", "500mg"]},
    "dophyllin": {"generic": "Doxofylline", "class": "Xanthine Bronchodilator", "common_forms": ["Tablet", "Syrup"], "common_strengths": ["400mg", "100mg/5ml"]},
    "uliv": {"generic": "Ursodeoxycholic Acid", "class": "Hepatoprotective / Bile Acid", "common_forms": ["Tablet", "Capsule"], "common_strengths": ["150mg", "300mg"]},
    "comet": {"generic": "Metformin Hydrochloride", "class": "Biguanide Antidiabetic", "common_forms": ["Tablet"], "common_strengths": ["500mg", "850mg", "1000mg"]},
    "nexcital": {"generic": "Escitalopram", "class": "Antidepressant (SSRI)", "common_forms": ["Tablet"], "common_strengths": ["5mg", "10mg", "20mg"]},
    "asynta": {"generic": "Aspirin (Acetylsalicylic Acid)", "class": "Antiplatelet / NSAID", "common_forms": ["Tablet"], "common_strengths": ["75mg", "100mg", "300mg"]},

    # Target Doctor Prescription Medications
    "rozad": {"generic": "Rosuvastatin", "class": "HMG-CoA Reductase Inhibitor (Statin)", "common_forms": ["Capsule", "Tablet"], "common_strengths": ["5mg", "10mg", "20mg", "40mg"]},
    "ambulox": {"generic": "Ambroxol + Amoxicillin", "class": "Mucolytic + Penicillin Antibiotic", "common_forms": ["Tablet", "Capsule"], "common_strengths": ["15mg/250mg", "30mg/500mg"]},
    "potil": {"generic": "Clonazepam", "class": "Benzodiazepine Anxiolytic", "common_forms": ["Tablet"], "common_strengths": ["0.25mg", "0.5mg", "1mg"]},
    "potil pm": {"generic": "Clonazepam + Paroxetine", "class": "Anxiolytic / Antidepressant", "common_forms": ["Tablet"], "common_strengths": ["0.5mg/10mg", "0.5mg/12.5mg"]},
    "placida": {"generic": "Flupentixol + Melitracen", "class": "Anxiolytic / Psychotropic", "common_forms": ["Tablet"], "common_strengths": ["0.5mg/10mg"]},
    "esotop": {"generic": "Esomeprazole", "class": "Proton Pump Inhibitor (PPI)", "common_forms": ["Tablet", "Capsule"], "common_strengths": ["20mg", "40mg"]},
    "esotop hc": {"generic": "Esomeprazole", "class": "Proton Pump Inhibitor (PPI)", "common_forms": ["Tablet", "Capsule"], "common_strengths": ["40mg"]},

    # Common Core Medical Prescription Formulations
    "paracetamol": {"generic": "Paracetamol", "class": "Analgesic / Antipyretic", "common_forms": ["Tablet", "Syrup", "IV Infusion"], "common_strengths": ["500mg", "650mg", "120mg/5ml"]},
    "amoxicillin": {"generic": "Amoxicillin", "class": "Aminopenicillin Antibiotic", "common_forms": ["Capsule", "Suspension"], "common_strengths": ["250mg", "500mg", "875mg"]},
    "augmentin": {"generic": "Amoxicillin + Clavulanic Acid", "class": "Beta-lactam + Beta-lactamase Inhibitor", "common_forms": ["Tablet", "Suspension", "Injection"], "common_strengths": ["375mg", "625mg", "1g"]},
    "omeprazole": {"generic": "Omeprazole", "class": "Proton Pump Inhibitor (PPI)", "common_forms": ["Capsule", "Injection"], "common_strengths": ["20mg", "40mg"]},
    "pantoprazole": {"generic": "Pantoprazole", "class": "Proton Pump Inhibitor (PPI)", "common_forms": ["Tablet", "Injection"], "common_strengths": ["20mg", "40mg"]},
    "esomeprazole": {"generic": "Esomeprazole", "class": "Proton Pump Inhibitor (PPI)", "common_forms": ["Tablet", "Capsule", "Injection"], "common_strengths": ["20mg", "40mg"]},
    "metformin": {"generic": "Metformin Hydrochloride", "class": "Biguanide Antidiabetic", "common_forms": ["Tablet"], "common_strengths": ["500mg", "850mg", "1000mg"]},
    "azithromycin": {"generic": "Azithromycin", "class": "Macrolide Antibiotic", "common_forms": ["Tablet", "Suspension"], "common_strengths": ["250mg", "500mg"]},
    "ciprofloxacin": {"generic": "Ciprofloxacin", "class": "Fluoroquinolone Antibiotic", "common_forms": ["Tablet", "Eye Drops", "IV Infusion"], "common_strengths": ["250mg", "500mg", "750mg"]},
    "ceftriaxone": {"generic": "Ceftriaxone", "class": "3rd Gen Cephalosporin Antibiotic", "common_forms": ["Injection (IV/IM)"], "common_strengths": ["500mg", "1g", "2g"]},
    "cefixime": {"generic": "Cefixime", "class": "3rd Gen Cephalosporin Antibiotic", "common_forms": ["Capsule", "Suspension"], "common_strengths": ["200mg", "400mg"]},
    "ibuprofen": {"generic": "Ibuprofen", "class": "NSAID", "common_forms": ["Tablet", "Syrup"], "common_strengths": ["200mg", "400mg", "600mg"]},
    "cetirizine": {"generic": "Cetirizine Hydrochloride", "class": "Second-generation Antihistamine", "common_forms": ["Tablet", "Syrup"], "common_strengths": ["10mg", "5mg/5ml"]},
    "fexofenadine": {"generic": "Fexofenadine Hydrochloride", "class": "Second-generation Antihistamine", "common_forms": ["Tablet", "Suspension"], "common_strengths": ["60mg", "120mg", "180mg"]},
    "salbutamol": {"generic": "Salbutamol", "class": "Short-acting Beta-2 Agonist Bronchodilator", "common_forms": ["Inhaler", "Syrup", "Nebulizer Solution"], "common_strengths": ["100mcg/puff", "2mg/5ml"]},
    "atorvastatin": {"generic": "Atorvastatin", "class": "HMG-CoA Reductase Inhibitor (Statin)", "common_forms": ["Tablet"], "common_strengths": ["10mg", "20mg", "40mg", "80mg"]},
    "amlodipine": {"generic": "Amlodipine Besylate", "class": "Dihydropyridine Calcium Channel Blocker", "common_forms": ["Tablet"], "common_strengths": ["2.5mg", "5mg", "10mg"]},
    "losartan": {"generic": "Losartan Potassium", "class": "Angiotensin II Receptor Blocker (ARB)", "common_forms": ["Tablet"], "common_strengths": ["25mg", "50mg", "100mg"]},
    "clopidogrel": {"generic": "Clopidogrel Bisulfate", "class": "P2Y12 Platelet Inhibitor", "common_forms": ["Tablet"], "common_strengths": ["75mg"]},
    "aspirin": {"generic": "Aspirin (Acetylsalicylic Acid)", "class": "Antiplatelet / NSAID", "common_forms": ["Tablet"], "common_strengths": ["75mg", "100mg", "300mg"]},
    "doxycycline": {"generic": "Doxycycline Hyclate", "class": "Tetracycline Antibiotic", "common_forms": ["Capsule", "Tablet"], "common_strengths": ["100mg"]},
    "metronidazole": {"generic": "Metronidazole", "class": "Nitroimidazole Antimicrobial", "common_forms": ["Tablet", "Suspension", "IV Infusion"], "common_strengths": ["200mg", "400mg", "500mg"]},
    "propranolol": {"generic": "Propranolol Hydrochloride", "class": "Non-selective Beta-blocker", "common_forms": ["Tablet"], "common_strengths": ["10mg", "40mg"]},
    "clonazepam": {"generic": "Clonazepam", "class": "Benzodiazepine Anticonvulsant / Anxiolytic", "common_forms": ["Tablet"], "common_strengths": ["0.5mg", "1mg", "2mg"]},
    "montelukast": {"generic": "Montelukast Sodium", "class": "Leukotriene Receptor Antagonist", "common_forms": ["Tablet", "Chewable"], "common_strengths": ["4mg", "5mg", "10mg"]},
}


def normalize_medicine_name(candidate_text: str, similarity_threshold: float = 0.65) -> dict[str, Any]:
    """
    Compares candidate OCR text against the medicine reference database.
    Computes fuzzy similarity ratios, compiles top candidate matches, and classifies status.

    Rules:
    - If similarity >= 0.90: 'Verified Exact'
    - If 0.65 <= similarity < 0.90: 'Likely Match'
    - If similarity < 0.65: 'Uncertain / Review Needed' (matched_name is NOT guessed)
    """
    clean_text = candidate_text.strip().lower()
    clean_text = re.sub(r"^(tab|cap|inj|syr|susp|drop|drp|ointment|crm|tablet|capsule|t|c)\.?\s*", "", clean_text)
    clean_medicine_word = re.sub(r"\s+\d+.*$", "", clean_text).strip()

    if not clean_medicine_word:
        clean_medicine_word = clean_text

    # Calculate similarity scores across all database entries
    scored_candidates: list[tuple[str, float]] = []
    for db_key in MEDICINE_REFERENCE_DB:
        if clean_medicine_word == db_key:
            scored_candidates.append((db_key, 1.0))
        else:
            score = SequenceMatcher(None, clean_medicine_word, db_key).ratio()
            scored_candidates.append((db_key, score))

    scored_candidates.sort(key=lambda x: x[1], reverse=True)

    best_match_key = None
    best_score = 0.0
    if scored_candidates:
        best_match_key, best_score = scored_candidates[0]

    # Collect top candidates above 0.45 threshold to present to the user
    candidate_matches = []
    for cand_key, score in scored_candidates[:4]:
        if score >= 0.45:
            candidate_matches.append({
                "candidate_name": cand_key.title(),
                "generic_name": MEDICINE_REFERENCE_DB[cand_key]["generic"],
                "therapeutic_class": MEDICINE_REFERENCE_DB[cand_key]["class"],
                "similarity_score": round(score, 3),
            })

    if best_score >= 0.90:
        status = "Verified Exact"
        final_matched_name = best_match_key.title() if best_match_key else None
        final_generic = MEDICINE_REFERENCE_DB[best_match_key]["generic"] if best_match_key else None
        final_class = MEDICINE_REFERENCE_DB[best_match_key]["class"] if best_match_key else None
    elif best_score >= similarity_threshold:
        status = "Likely Match"
        final_matched_name = best_match_key.title() if best_match_key else None
        final_generic = MEDICINE_REFERENCE_DB[best_match_key]["generic"] if best_match_key else None
        final_class = MEDICINE_REFERENCE_DB[best_match_key]["class"] if best_match_key else None
    else:
        # Never silently replace an unclear medicine name with a guessed name
        status = "Uncertain / Review Needed"
        final_matched_name = None
        final_generic = None
        final_class = None

    return {
        "candidate": candidate_text,
        "clean_query": clean_medicine_word,
        "matched_name": final_matched_name,
        "generic_name": final_generic,
        "therapeutic_class": final_class,
        "similarity_score": round(best_score, 3),
        "status": status,
        "candidate_matches": candidate_matches,
        "common_forms": MEDICINE_REFERENCE_DB[best_match_key]["common_forms"] if (best_match_key and status != "Uncertain / Review Needed") else [],
        "common_strengths": MEDICINE_REFERENCE_DB[best_match_key]["common_strengths"] if (best_match_key and status != "Uncertain / Review Needed") else [],
    }


def extract_prescription_fields(ocr_text: str) -> tuple[PrescriptionMetadata, list[ExtractedPrescriptionItem]]:
    """
    Parses OCR transcription lines into structured prescription entities.

    SAFETY CONSTRAINT:
    Does NOT infer missing dosages, frequencies, routes, instructions, or durations.
    Missing fields are marked explicitly as 'Not specified'.
    """
    lines = [line.strip() for line in ocr_text.splitlines() if line.strip()]
    metadata = PrescriptionMetadata()
    items: list[ExtractedPrescriptionItem] = []

    date_pattern = re.compile(r"\b(\d{1,2}[-/.](?:\d{1,2}|[A-Za-z]{3})[-/.](?:\d{2,4}))\b", re.IGNORECASE)
    prescriber_pattern = re.compile(r"\b(Dr\.?|Doctor|Prof\.?)\b", re.IGNORECASE)
    credentials_pattern = re.compile(r"\b(M\.?B\.?B\.?S\.?|S\.?M\.?O\.?|F\.?C\.?P\.?S\.?|M\.?D\.?|M\.?S\.?|F\.?R\.?C\.?P\.?)\b", re.IGNORECASE)
    reg_pattern = re.compile(r"\b(Reg(?:\.|istration)?\s*(?:No\.?)?[:\s]*[A-Z0-9-]+)\b", re.IGNORECASE)
    hospital_pattern = re.compile(r"\b(Hospital|Clinic|Healthcare|Dispensary)\b", re.IGNORECASE)
    title_pattern = re.compile(r"\b(Senior\s+Medical\s+Officer|Medical\s+Officer|Consultant|Surgeon|Physician|Specialist)\b", re.IGNORECASE)
    contact_pattern = re.compile(r"\b(Mob(?:ile)?\.?|Phone|Tel|Cell)[:\s]*([0-9\-\+\s]{7,15})\b", re.IGNORECASE)

    patient_name_pat = re.compile(r"^Name[:\s]+(.*)$", re.IGNORECASE)
    sex_pat = re.compile(r"^Sex[:\s]+(.*)$", re.IGNORECASE)
    age_pat = re.compile(r"^Age[:\s]+(.*)$", re.IGNORECASE)
    opd_pat = re.compile(r"^OPD\s*(?:No\.?)?[:\s]+(.*)$", re.IGNORECASE)

    diag_pat = re.compile(r"^Diagnosis[:\s]+(.*)$", re.IGNORECASE)
    vitals_header = re.compile(r"^Vitals[:\s]*$", re.IGNORECASE)
    vital_line_pat = re.compile(r"^(?:-\s*)?(BP|SpO[2₂]|Pulse|Temp|Weight|Wt)[:\s]+(.*)$", re.IGNORECASE)

    rx_header = re.compile(r"^(?:Prescription\s*\(Rx\)|Rx|Prescription)[:\s]*$", re.IGNORECASE)
    advice_header = re.compile(r"^Advice[:\s]*$", re.IGNORECASE)
    advice_line_pat = re.compile(r"^(?:-\s*)?(Diet|Walk(?:\s*\/\s*Exercise)?|Exercise|Rest|Plenty\s+of\s+fluids|Avoid\s+oily\s+food)\b", re.IGNORECASE)
    followup_pat = re.compile(r"^(?:Follow-?up|Review)[:\s]+(.*)$", re.IGNORECASE)
    address_pat = re.compile(r"^(?:Address[:\s]+)?([0-9A-Za-z\-\/,\s]+(?:Nagar|Road|Street|Kota|Raj|Delhi|Mumbai|Colony|Lane|\d{6}).*)$", re.IGNORECASE)

    strength_pattern = re.compile(r"\b(\d+(?:\.\d+)?\s*(?:mg|g|mcg|ml|iu|%|mg\/5ml|mcg\/puff))\b", re.IGNORECASE)
    form_pattern = re.compile(r"\b(Tab(?:let)?|Cap(?:sule)?|Syr(?:up)?|Susp(?:ension)?|Inj(?:ection)?|Drops?|Ointment|Cream|Inhaler|Suppository|T|Cap|Tab)\b\.?", re.IGNORECASE)
    frequency_pattern = re.compile(r"\b(\d-\d-\d|\d\+\d\+\d|once\s+daily|twice\s+daily|three\s+times\s+daily|b\.?i\.?d\.?|t\.?i\.?d\.?|q\.?i\.?d\.?|o\.?d\.?|q8h|q12h|prn|sos|stat|at\s+bedtime|1\s*BD|1\s*OD(?:\s*AC|\s*HS)?|10P)\b", re.IGNORECASE)
    duration_pattern = re.compile(r"\b(\d+\s*(?:days?|weeks?|months?|d|wks?|mths?))\b", re.IGNORECASE)
    route_pattern = re.compile(r"\b(oral|topical|sublingual|ophthalmic|otic|inhalation|iv|im|intravenous|intramuscular)\b", re.IGNORECASE)
    instructions_pattern = re.compile(r"\b(before\s+food|after\s+food|with\s+food|empty\s+stomach|a\.?c\.?|p\.?c\.?|at\s+night|morning|\d{1,2}\s*(?:AM|PM))\b", re.IGNORECASE)

    form_norm_map = {
        "tab": "Tablet",
        "tablet": "Tablet",
        "t": "Tablet",
        "cap": "Capsule",
        "capsule": "Capsule",
        "syr": "Syrup",
        "syrup": "Syrup",
        "susp": "Suspension",
        "suspension": "Suspension",
        "inj": "Injection",
        "injection": "Injection",
        "drops": "Drops",
        "drop": "Drops",
        "ointment": "Ointment",
        "cream": "Cream",
        "inhaler": "Inhaler",
        "suppository": "Suppository",
    }

    in_advice_section = False

    for line in lines:
        # Check section headers
        if vitals_header.match(line):
            continue
        if rx_header.match(line):
            in_advice_section = False
            continue
        if advice_header.match(line):
            in_advice_section = True
            continue

        # Prescriber info
        if prescriber_pattern.search(line) and not metadata.prescriber_name:
            metadata.prescriber_name = line
            continue
        if credentials_pattern.search(line) and not form_pattern.search(line):
            if not metadata.prescriber_credentials:
                metadata.prescriber_credentials = line
            continue
        if title_pattern.search(line) and not form_pattern.search(line):
            if metadata.prescriber_credentials:
                metadata.prescriber_credentials += f", {line}"
            else:
                metadata.prescriber_credentials = line
            continue
        if reg_pattern.search(line):
            reg_match = reg_pattern.search(line)
            if reg_match:
                metadata.registration_number = reg_match.group(1)
            continue
        if hospital_pattern.search(line) and not form_pattern.search(line):
            if not metadata.clinic_or_hospital:
                metadata.clinic_or_hospital = line
            continue
        if contact_pattern.search(line):
            c_match = contact_pattern.search(line)
            if c_match:
                metadata.prescriber_contact = c_match.group(0)
            continue

        # Patient Info
        if patient_name_pat.match(line):
            metadata.patient_name = patient_name_pat.match(line).group(1).strip()
            continue
        if sex_pat.match(line):
            metadata.patient_sex = sex_pat.match(line).group(1).strip()
            continue
        if age_pat.match(line):
            metadata.patient_age = age_pat.match(line).group(1).strip()
            continue
        if opd_pat.match(line):
            metadata.opd_number = opd_pat.match(line).group(1).strip()
            continue
        if date_pattern.search(line):
            d_match = date_pattern.search(line)
            if not metadata.prescription_date:
                metadata.prescription_date = d_match.group(1)
            if line.lower().startswith("date") or len(line) < 20:
                continue

        # Clinical Assessment
        if diag_pat.match(line):
            metadata.diagnosis = diag_pat.match(line).group(1).strip()
            continue

        # Vitals
        if vital_line_pat.match(line):
            metadata.vitals.append(line.lstrip("- ").strip())
            continue

        # Advice & Follow-up
        if followup_pat.match(line):
            metadata.follow_up = followup_pat.match(line).group(1).strip()
            continue
        if in_advice_section and (line.startswith("-") or advice_line_pat.search(line)):
            metadata.advice.append(line.lstrip("- ").strip())
            continue
        if not in_advice_section and advice_line_pat.search(line) and not form_pattern.search(line):
            metadata.advice.append(line.lstrip("- ").strip())
            continue

        # Address
        if line.lower().startswith("address:") or ("vigyan nagar" in line.lower() and "kota" in line.lower()):
            addr_text = re.sub(r"^Address[:\s]+", "", line).strip()
            metadata.clinic_address = addr_text
            continue

        # Extract Item fields - DO NOT INFER MISSING VALUES
        form_match = form_pattern.search(line)
        if form_match:
            raw_form = form_match.group(1).lower()
            dosage_form = form_norm_map.get(raw_form, form_match.group(1).title())
        else:
            dosage_form = "Not specified"

        strength_match = strength_pattern.search(line)
        strength = strength_match.group(1) if strength_match else "Not specified"

        freq_match = frequency_pattern.search(line)
        frequency = freq_match.group(1) if freq_match else "Not specified"

        duration_match = duration_pattern.search(line)
        duration = duration_match.group(1) if duration_match else "Not specified"

        route_match = route_pattern.search(line)
        route = route_match.group(1).title() if route_match else "Not specified"

        inst_match = instructions_pattern.search(line)
        instructions = inst_match.group(1).title() if inst_match else "Not specified"

        # Candidate medicine word: strip recognized tokens
        candidate_words = line
        for pat in (form_pattern, strength_pattern, frequency_pattern, duration_pattern, route_pattern, instructions_pattern):
            candidate_words = pat.sub("", candidate_words)
        cleaned_candidate = re.sub(r"^[0-9\.\-\)\s–]+", "", candidate_words).strip()
        cleaned_candidate = re.sub(r"[,\;:\–\-]", " ", cleaned_candidate).strip()
        cleaned_candidate = re.sub(r"\s+", " ", cleaned_candidate)

        if len(cleaned_candidate) < 2:
            cleaned_candidate = line.split()[0] if line.split() else "Unknown"

        norm_res = normalize_medicine_name(cleaned_candidate)

        item = ExtractedPrescriptionItem(
            raw_text=line,
            medicine_name=cleaned_candidate,
            normalized_name=norm_res["matched_name"],
            generic_name=norm_res["generic_name"],
            match_confidence=norm_res["similarity_score"],
            status=norm_res["status"],
            dosage_form=dosage_form,
            strength=strength,
            frequency=frequency,
            duration=duration,
            route=route,
            instructions=instructions,
            candidate_matches=norm_res.get("candidate_matches", []),
            user_confirmed=False,
        )
        items.append(item)

    # Maintain patient_info string for backwards compatibility
    parts = []
    if metadata.patient_name:
        parts.append(metadata.patient_name)
    if metadata.patient_sex:
        parts.append(metadata.patient_sex)
    if metadata.patient_age:
        parts.append(metadata.patient_age)
    if parts:
        metadata.patient_info = ", ".join(parts)

    return metadata, items
