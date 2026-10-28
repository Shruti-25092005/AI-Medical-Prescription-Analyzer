"""
Authoritative Medicine Explanation and Retrieval Engine (RAG).
Provides traceable, source-grounded clinical monographs for verified medicines.
Covers indications, mechanisms, side effects, precautions, administration instructions,
and red-flag warning triggers.

Sources:
- WHO Model List of Essential Medicines (23rd List, 2023)
- British National Formulary (BNF 86, National Institute for Health and Care Excellence)
- US Food and Drug Administration (FDA) Structured Product Labeling (DailyMed)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class MedicineMonograph:
    medicine_name: str
    generic_name: str
    therapeutic_class: str
    atc_code: str
    approved_indications: list[str]
    mechanism_of_action: str
    common_side_effects: list[str]
    serious_warnings: list[str]
    administration_guidelines: list[str]
    when_to_seek_doctor: list[str]
    contraindications: list[str]
    source_attribution: str
    reference_links: list[str]


# Authoritative grounded clinical monographs
CLINICAL_MONOGRAPHS: dict[str, MedicineMonograph] = {
    "amoxicillin": MedicineMonograph(
        medicine_name="Amoxicillin (Moxacil / Amoxil)",
        generic_name="Amoxicillin",
        therapeutic_class="Beta-Lactam Antibacterial / Penicillin",
        atc_code="J01CA04",
        approved_indications=[
            "Upper and lower respiratory tract infections (sinusitis, otitis media, pneumonia)",
            "Urinary tract infections (UTI)",
            "Skin and soft tissue infections",
            "H. pylori eradication (as part of triple combination therapy)",
        ],
        mechanism_of_action=(
            "Inhibits bacterial cell wall synthesis by binding to penicillin-binding proteins (PBPs), "
            "leading to lysis and death of susceptible bacterial microorganisms."
        ),
        common_side_effects=[
            "Nausea, vomiting, and loose stools or diarrhea",
            "Transient mild maculopapular skin rash",
            "Oral or vaginal candidiasis (thrush)",
        ],
        serious_warnings=[
            "Severe hypersensitivity reactions including anaphylaxis, angioedema, and serum sickness",
            "Clostridioides difficile-associated diarrhea (pseudomembranous colitis)",
            "Severe cutaneous adverse reactions (Stevens-Johnson syndrome, DRESS syndrome)",
        ],
        administration_guidelines=[
            "Take every 8 hours with a full glass of water",
            "Can be taken with or without food; taking with food minimizes GI upset",
            "Complete the entire prescribed antimicrobial course even if symptoms resolve earlier",
        ],
        when_to_seek_doctor=[
            "Signs of severe allergy: facial swelling, difficulty swallowing, or wheezing",
            "Severe watery diarrhea persisting for more than 24 hours",
            "High fever accompanied by a spreading or blistering skin rash",
        ],
        contraindications=["Confirmed penicillin or beta-lactam hypersensitivity", "Infectious mononucleosis (high risk of rash)"],
        source_attribution="WHO Model List of Essential Medicines (2023), British National Formulary (BNF 86)",
        reference_links=["https://bnf.nice.org.uk/drugs/amoxicillin/", "https://dailymed.nlm.nih.gov/dailymed/"],
    ),
    "cetirizine": MedicineMonograph(
        medicine_name="Cetirizine Hydrochloride (Zyrtec / Alatrol)",
        generic_name="Cetirizine Hydrochloride",
        therapeutic_class="Second-Generation H1-Antihistamine",
        atc_code="R06AE07",
        approved_indications=[
            "Allergic rhinitis (hay fever, seasonal and perennial)",
            "Allergic conjunctivitis (ocular itching and tearing)",
            "Chronic idiopathic urticaria (hives and pruritus)",
        ],
        mechanism_of_action=(
            "Selective peripheral H1-receptor antagonist. Competitively inhibits histamine at H1 receptors, "
            "reducing allergic vascular permeability, flare reactions, and pruritus with minimal CNS sedation."
        ),
        common_side_effects=[
            "Mild fatigue or drowsiness in sensitive individuals",
            "Dry mouth (xerostomia)",
            "Mild headache and pharyngitis",
        ],
        serious_warnings=[
            "Additive CNS depression when taken concurrently with alcohol or sedatives",
            "Urinary retention risk in patients with bladder outflow obstruction or prostatic hyperplasia",
        ],
        administration_guidelines=[
            "Standard adult dose: 10mg once daily, preferably in the evening",
            "May be taken with or without food",
            "Avoid operating heavy machinery or consuming alcohol until response is established",
        ],
        when_to_seek_doctor=[
            "Severe allergic swelling of the lips, tongue, or airway",
            "Acute inability to urinate (urinary retention)",
            "Severe unexplained palpitations",
        ],
        contraindications=["End-stage renal disease (CrCl < 10 mL/min)", "Known hypersensitivity to cetirizine or hydroxyzine"],
        source_attribution="British National Formulary (BNF 86), US FDA Product Labeling",
        reference_links=["https://bnf.nice.org.uk/drugs/cetirizine-hydrochloride/", "https://dailymed.nlm.nih.gov/dailymed/"],
    ),
    "paracetamol": MedicineMonograph(
        medicine_name="Paracetamol (Napa / Calpol)",
        generic_name="Paracetamol (Acetaminophen)",
        therapeutic_class="Analgesic / Antipyretic",
        atc_code="N02BE01",
        approved_indications=[
            "Mild-to-moderate pain (headache, dental pain, musculoskeletal ache)",
            "Pyrexia (fever reduction in viral and bacterial infections)",
        ],
        mechanism_of_action=(
            "Centrally inhibits prostaglandin synthesis through COX-1 and COX-2 inhibition, "
            "and activates descending serotonergic pain pathways. Acts on the hypothalamic "
            "thermoregulatory center to produce peripheral vasodilation and heat loss."
        ),
        common_side_effects=[
            "Rare at therapeutic doses",
            "Nausea and mild gastrointestinal discomfort",
            "Infrequent allergic rash or urticaria",
        ],
        serious_warnings=[
            "Severe hepatotoxicity with doses exceeding 4000 mg/day (4g/24h) or in chronic alcohol use",
            "Acute liver failure risk with accidental double-dosing of combination cold/cough products",
        ],
        administration_guidelines=[
            "Standard adult dose: 500mg - 1000mg every 4 to 6 hours as needed (maximum 4000mg in 24 hours)",
            "May be taken with or without food",
            "Ensure minimum 4-hour interval between doses",
        ],
        when_to_seek_doctor=[
            "Fever persists for more than 3 consecutive days",
            "Pain worsens or lasts longer than 5 to 7 days",
            "Signs of jaundice (yellowing of eyes/skin), dark tea-colored urine, or upper right abdominal pain",
        ],
        contraindications=["Severe active hepatic impairment or acute liver disease", "Hypersensitivity to paracetamol"],
        source_attribution="WHO Model Formulary (2023), British National Formulary (BNF 86)",
        reference_links=["https://www.who.int/groups/expert-committee-on-selection-and-use-of-essential-medicines", "https://bnf.nice.org.uk/drugs/paracetamol/"],
    ),
    "azithromycin": MedicineMonograph(
        medicine_name="Azithromycin (Zimax)",
        generic_name="Azithromycin",
        therapeutic_class="Macrolide Antibacterial",
        atc_code="J01FA10",
        approved_indications=[
            "Community-acquired pneumonia and acute bacterial exacerbation of COPD",
            "Streptococcal pharyngitis/tonsillitis in penicillin-allergic patients",
            "Skin and soft tissue infections caused by susceptible pathogens",
            "Uncomplicated genital Chlamydia trachomatis infections",
        ],
        mechanism_of_action=(
            "Reversibly binds to the 50S ribosomal subunit of susceptible microorganisms, "
            "inhibiting transpeptidation and protein translocation, thereby arresting bacterial protein synthesis."
        ),
        common_side_effects=[
            "Gastrointestinal disturbances: diarrhea, abdominal cramps, nausea, vomiting",
            "Transient headache and dizziness",
            "Altered taste sensation",
        ],
        serious_warnings=[
            "QT-interval prolongation and risk of ventricular arrhythmias, including Torsades de Pointes",
            "Clostridioides difficile-associated diarrhea (CDAD)",
            "Hepatotoxicity and cholestatic jaundice",
        ],
        administration_guidelines=[
            "Take once daily at the same time each day",
            "Tablets may be taken with food to decrease GI distress; suspensions should be taken 1 hour before or 2 hours after meals",
            "Complete the full prescribed antimicrobial course even if symptoms resolve earlier",
        ],
        when_to_seek_doctor=[
            "Palpitations, lightheadedness, or sudden fainting spells",
            "Persistent watery or bloody diarrhea",
            "Severe cutaneous adverse reactions (widespread blistering or peeling rash)",
        ],
        contraindications=["Hypersensitivity to macrolides/ketolides", "History of cholestatic jaundice/hepatic dysfunction associated with prior azithromycin use"],
        source_attribution="OpenFDA Drug Labeling, British National Formulary (BNF 86)",
        reference_links=["https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid=azithromycin", "https://bnf.nice.org.uk/drugs/azithromycin/"],
    ),
    "propranolol": MedicineMonograph(
        medicine_name="Propranolol (Indever / Inderen)",
        generic_name="Propranolol Hydrochloride",
        therapeutic_class="Non-selective Beta-Adrenergic Receptor Blocker",
        atc_code="C07AA05",
        approved_indications=[
            "Hypertension and angina pectoris",
            "Cardiac dysrhythmias and supraventricular tachycardia",
            "Migraine prophylaxis",
            "Essential tremor and symptomatic relief of situational anxiety / thyrotoxicosis",
        ],
        mechanism_of_action=(
            "Competitively antagonizes both beta-1 and beta-2 adrenergic receptors, reducing cardiac "
            "inotropic and chronotropic responses, decreasing cardiac output, and dampening peripheral sympathetic stimulation."
        ),
        common_side_effects=[
            "Bradycardia (slow heart rate) and hypotension",
            "Cold extremities and Raynaud phenomenon exacerbation",
            "Fatigue, sleep disturbances, vivid dreams",
            "Bronchoconstriction in susceptible individuals",
        ],
        serious_warnings=[
            "Contraindicated in bronchial asthma and COPD due to potentially fatal bronchospasm",
            "Do NOT abruptly discontinue therapy; sudden cessation can trigger rebound severe hypertension, angina, or myocardial infarction",
            "Masks early adrenergic symptoms of hypoglycemia (tachycardia, tremor) in diabetic patients",
        ],
        administration_guidelines=[
            "Take consistently with meals or on an empty stomach to maintain predictable absorption",
            "Check pulse and blood pressure regularly as directed by the prescriber",
            "Taper dosage gradually under medical supervision when stopping",
        ],
        when_to_seek_doctor=[
            "Resting pulse falls below 50 beats per minute with dizziness or lightheadedness",
            "Sudden shortness of breath, wheezing, or difficulty breathing",
            "Swelling of ankles or sudden unexplained weight gain (signs of congestive heart failure)",
        ],
        contraindications=["Bronchial asthma or history of bronchospasm", "Severe bradycardia, second- or third-degree AV block", "Cardiogenic shock"],
        source_attribution="British National Formulary (BNF 86), WHO Model List",
        reference_links=["https://bnf.nice.org.uk/drugs/propranolol-hydrochloride/"],
    ),
    "clonazepam": MedicineMonograph(
        medicine_name="Clonazepam (Rivotril)",
        generic_name="Clonazepam",
        therapeutic_class="Anticonvulsant / Anxiolytic (Benzodiazepine)",
        atc_code="N03AE01",
        approved_indications=[
            "Epilepsy: myoclonic, atonic, absence, and partial seizures",
            "Panic disorder with or without agoraphobia",
            "Short-term adjunct for acute severe anxiety states",
        ],
        mechanism_of_action=(
            "Enhances the inhibitory activity of gamma-aminobutyric acid (GABA) by binding to the "
            "GABA-A receptor benzodiazepine site, increasing neuronal chloride conductance and hyperpolarization."
        ),
        common_side_effects=[
            "Somnolence, sedation, and daytime fatigue",
            "Ataxia, coordination disturbances, and muscle weakness",
            "Cognitive slowing and anterograde amnesia",
        ],
        serious_warnings=[
            "High risk of physical and psychological dependence; withdrawal symptoms occur upon abrupt discontinuation",
            "Profound CNS depression and respiratory arrest when combined with opioids, alcohol, or other sedatives",
            "Impairs motor skills; patients must not drive or operate heavy machinery",
        ],
        administration_guidelines=[
            "Take exactly as prescribed; do not alter dosage without consulting the prescriber",
            "Usually taken at bedtime to minimize daytime sedative disruption",
            "Avoid alcohol entirely during therapy",
        ],
        when_to_seek_doctor=[
            "Extreme sleepiness, shallow breathing, or confusion",
            "Worsening depression, paradoxical agitation, or emergence of suicidal ideation",
        ],
        contraindications=["Severe respiratory insufficiency", "Severe hepatic impairment", "Myasthenia gravis", "Sleep apnea syndrome"],
        source_attribution="US FDA MedWatch, British National Formulary (BNF 86)",
        reference_links=["https://bnf.nice.org.uk/drugs/clonazepam/"],
    ),
    "losartan": MedicineMonograph(
        medicine_name="Losartan (Losita)",
        generic_name="Losartan Potassium",
        therapeutic_class="Angiotensin II Receptor Blocker (ARB)",
        atc_code="C09CA01",
        approved_indications=[
            "Essential hypertension in adults and children",
            "Renal protection in type 2 diabetes mellitus with nephropathy and proteinuria",
            "Reduction in risk of stroke in patients with hypertension and left ventricular hypertrophy",
        ],
        mechanism_of_action=(
            "Selectively blocks the binding of angiotensin II to the AT1 receptor subtype found in vascular "
            "smooth muscle and the adrenal gland, inhibiting vasoconstriction and aldosterone secretion."
        ),
        common_side_effects=[
            "Dizziness, vertigo, and orthostatic lightheadedness",
            "Hyperkalemia (elevated serum potassium)",
            "Mild fatigue and nasal congestion",
        ],
        serious_warnings=[
            "FETAL TOXICITY WARNING: Boxed warning against use during second and third trimesters of pregnancy",
            "Risk of acute kidney injury in patients with bilateral renal artery stenosis or severe volume depletion",
        ],
        administration_guidelines=[
            "Take once or twice daily, with or without food",
            "Maintain consistent hydration; avoid potassium supplements unless specifically prescribed",
            "Monitor serum creatinine and electrolytes regularly",
        ],
        when_to_seek_doctor=[
            "Facial swelling, lip or tongue angioedema, or difficulty swallowing/breathing",
            "Irregular heartbeat or severe muscle weakness (potential hyperkalemia)",
            "Dizziness when standing that does not resolve quickly",
        ],
        contraindications=["Pregnancy", "Concomitant use with aliskiren in patients with diabetes mellitus"],
        source_attribution="OpenFDA Drug Labeling, WHO Model List of Essential Medicines",
        reference_links=["https://bnf.nice.org.uk/drugs/losartan-potassium/"],
    ),
    "metformin": MedicineMonograph(
        medicine_name="Metformin (Comet)",
        generic_name="Metformin Hydrochloride",
        therapeutic_class="Biguanide Antihyperglycemic Agent",
        atc_code="A10BA02",
        approved_indications=[
            "Type 2 diabetes mellitus (first-line monotherapy or in combination)",
            "Polycystic ovary syndrome (PCOS, off-label)",
        ],
        mechanism_of_action=(
            "Activates AMP-activated protein kinase (AMPK), reducing hepatic gluconeogenesis and glycogenolysis, "
            "improving peripheral insulin sensitivity in skeletal muscle, and delaying intestinal glucose absorption."
        ),
        common_side_effects=[
            "Gastrointestinal: metallic taste, nausea, vomiting, abdominal fullness, diarrhea",
            "Decreased vitamin B12 absorption with prolonged therapy",
        ],
        serious_warnings=[
            "Lactic acidosis (rare but severe, ~50% mortality); risk elevated in renal impairment, sepsis, or hypoxemia",
            "Must be withheld prior to and for 48 hours following iodinated radiological contrast procedures",
        ],
        administration_guidelines=[
            "Always take with or immediately following a meal to minimize gastrointestinal side effects",
            "Swallow extended-release tablets whole; do not crush, chew, or split",
        ],
        when_to_seek_doctor=[
            "Severe fatigue, malaise, muscle pain, difficult rapid breathing, or unexpected hypothermia (signs of lactic acidosis)",
            "Persistent diarrhea or nausea that interferes with hydration",
        ],
        contraindications=["Severe renal failure (eGFR < 30 mL/min/1.73m2)", "Acute or chronic metabolic acidosis (including diabetic ketoacidosis)"],
        source_attribution="American Diabetes Association (ADA), BNF 86",
        reference_links=["https://bnf.nice.org.uk/drugs/metformin-hydrochloride/"],
    ),
    "cefixime": MedicineMonograph(
        medicine_name="Cefixime (Exeptim)",
        generic_name="Cefixime",
        therapeutic_class="Third-Generation Cephalosporin Antibacterial",
        atc_code="J01DD08",
        approved_indications=[
            "Uncomplicated urinary tract infections (UTIs)",
            "Otitis media, acute bacterial pharyngitis and tonsillitis",
            "Acute bronchitis and exacerbation of chronic bronchitis",
            "Uncomplicated gonorrhea",
        ],
        mechanism_of_action=(
            "Inhibits bacterial cell wall synthesis by binding to specific penicillin-binding proteins (PBPs), "
            "causing cell lysis and bactericidal activity. Highly resistant to hydrolysis by many beta-lactamases."
        ),
        common_side_effects=["Diarrhea, soft stools, abdominal cramps", "Nausea, dyspepsia, flatulence", "Headache, mild dizziness"],
        serious_warnings=[
            "Hypersensitivity cross-reactivity in patients with severe penicillin allergy (anaphylaxis)",
            "Clostridioides difficile-associated diarrhea (pseudomembranous colitis)",
        ],
        administration_guidelines=[
            "May be taken with or without food; taking with food helps alleviate stomach upset",
            "Complete the entire prescribed duration to avoid antimicrobial resistance",
        ],
        when_to_seek_doctor=["Watery, severe, or bloody diarrhea", "Skin rash, hives, or swelling of face and throat"],
        contraindications=["Known hypersensitivity to cephalosporins or severe beta-lactam allergy"],
        source_attribution="British National Formulary (BNF 86), OpenFDA",
        reference_links=["https://bnf.nice.org.uk/drugs/cefixime/"],
    ),
    "omeprazole": MedicineMonograph(
        medicine_name="Omeprazole (Ome / Prilosec)",
        generic_name="Omeprazole",
        therapeutic_class="Proton Pump Inhibitor (PPI)",
        atc_code="A02BC01",
        approved_indications=[
            "Gastroesophageal reflux disease (GERD) and erosive esophagitis",
            "Gastric and duodenal ulcers",
            "Helicobacter pylori eradication (in combination with antibiotics)",
            "Zollinger-Ellison syndrome",
        ],
        mechanism_of_action=(
            "Irreversibly inhibits the gastric parietal cell H+/K+-ATPase enzyme system (proton pump), "
            "suppressing both basal and stimulated gastric acid secretion."
        ),
        common_side_effects=["Headache, benign gastric polyps", "Diarrhea, constipation, abdominal pain, flatulence"],
        serious_warnings=[
            "Increased risk of bone fractures with high-dose long-term therapy",
            "Hypomagnesemia with prolonged use (> 1 year)",
            "Reduces bioactivation of clopidogrel via competitive CYP2C19 inhibition",
        ],
        administration_guidelines=[
            "Take 30 to 60 minutes before the first meal of the day (breakfast)",
            "Swallow capsules whole with water; do not crush or chew beads",
        ],
        when_to_seek_doctor=["Severe persistent diarrhea with fever", "Unexplained joint pain or butterfly facial rash"],
        contraindications=["Hypersensitivity to substituted benzimidazoles", "Concurrent use with rilpivirine-containing regimens"],
        source_attribution="WHO Model List, BNF 86",
        reference_links=["https://bnf.nice.org.uk/drugs/omeprazole/"],
    ),
    "dexibuprofen": MedicineMonograph(
        medicine_name="Dexibuprofen (Dexter)",
        generic_name="Dexibuprofen",
        therapeutic_class="Non-Steroidal Anti-Inflammatory Drug (NSAID)",
        atc_code="M01AE14",
        approved_indications=[
            "Relief of mild to moderate pain and inflammation (osteoarthritis, dysmenorrhea, dental pain)",
            "Musculoskeletal and joint inflammation",
        ],
        mechanism_of_action=(
            "Therapeutically active (S)-(+)-enantiomer of ibuprofen. Inhibits cyclooxygenase enzymes "
            "(COX-1 and COX-2), dampening synthesis of pro-inflammatory prostaglandins and thromboxanes."
        ),
        common_side_effects=[
            "Dyspepsia, heartburn, epigastric discomfort, nausea",
            "Dizziness, headache",
        ],
        serious_warnings=[
            "Increased risk of serious cardiovascular thrombotic events, myocardial infarction, and stroke",
            "Gastrointestinal ulceration, bleeding, and perforation, which can be fatal",
            "Renal toxicity: decreases renal blood flow in patients with heart failure or pre-existing renal impairment",
        ],
        administration_guidelines=[
            "Take with food or a glass of milk to minimize gastric mucosal irritation",
            "Use the lowest effective dose for the shortest duration necessary",
        ],
        when_to_seek_doctor=[
            "Black tarry stools, vomiting blood, or severe coffee-ground emesis",
            "Chest pain, shortness of breath, slurred speech, or unilateral weakness",
        ],
        contraindications=["Active peptic ulcer disease or GI bleeding", "Severe heart failure (NYHA Class IV)", "Third trimester of pregnancy"],
        source_attribution="British National Formulary (BNF), European Medicines Agency",
        reference_links=["https://bnf.nice.org.uk/treatment-summaries/non-steroidal-anti-inflammatory-drugs/"],
    ),
    "escitalopram": MedicineMonograph(
        medicine_name="Escitalopram (Escilex / Nexcital)",
        generic_name="Escitalopram",
        therapeutic_class="Selective Serotonin Reuptake Inhibitor (SSRI)",
        atc_code="N06AB10",
        approved_indications=[
            "Major depressive disorder (MDD)",
            "Generalized anxiety disorder (GAD), panic disorder, social anxiety disorder",
        ],
        mechanism_of_action=(
            "Highly selective inhibition of the presynaptic serotonin (5-HT) reuptake transporter (SERT), "
            "potentiating serotonergic neurotransmission in the central nervous system."
        ),
        common_side_effects=[
            "Nausea, insomnia or somnolence, dry mouth, increased sweating",
            "Sexual dysfunction (delayed ejaculation, decreased libido)",
        ],
        serious_warnings=[
            "Boxed warning for increased risk of suicidal thoughts and behaviors in children, adolescents, and young adults",
            "Dose-dependent QT-interval prolongation (maximum dose 20mg/day, 10mg in elderly)",
            "Serotonin syndrome when combined with other serotonergic agents (MAOIs, triptans, tramadol)",
        ],
        administration_guidelines=[
            "Take once daily in morning or evening, with or without food",
            "Therapeutic effect typically takes 2 to 4 weeks to develop; do not stop abruptly",
            "Taper gradually when discontinuing to avoid withdrawal syndrome",
        ],
        when_to_seek_doctor=[
            "Sudden changes in mood, emergent suicidal thoughts, or severe agitation",
            "Tremor, shivering, hyperthermia, muscle rigidity (signs of serotonin syndrome)",
        ],
        contraindications=["Concurrent use with MAOIs or within 14 days of MAOI discontinuation", "Known QT-interval prolongation"],
        source_attribution="US FDA DailyMed, British National Formulary (BNF 86)",
        reference_links=["https://bnf.nice.org.uk/drugs/escitalopram/"],
    ),
    "aspirin": MedicineMonograph(
        medicine_name="Aspirin (Asynta)",
        generic_name="Aspirin (Acetylsalicylic Acid)",
        therapeutic_class="Antiplatelet / Salicylate NSAID",
        atc_code="B01AC06",
        approved_indications=[
            "Secondary prevention of cardiovascular thrombotic events (myocardial infarction, ischemic stroke, TIA)",
            "Acute coronary syndromes (ACS)",
        ],
        mechanism_of_action=(
            "Irreversibly acetylates and inactivates platelet cyclooxygenase-1 (COX-1), blocking thromboxane A2 "
            "synthesis for the entire lifespan of the platelet (~7-10 days), preventing platelet aggregation."
        ),
        common_side_effects=[
            "Gastric irritation, dyspepsia, nausea",
            "Increased bleeding tendency and easy bruising",
        ],
        serious_warnings=[
            "Severe gastrointestinal bleeding and hemorrhagic stroke risk",
            "Reye's syndrome risk in children and adolescents with viral infections",
            "Aspirin-exacerbated respiratory disease (AERD) in asthmatic patients",
        ],
        administration_guidelines=[
            "Low-dose antiplatelet therapy: typically 75mg - 100mg once daily",
            "Take with food or a large glass of water to reduce mucosal irritation",
            "Enteric-coated tablets should be swallowed whole",
        ],
        when_to_seek_doctor=[
            "Signs of GI hemorrhage (black stools, vomiting blood)",
            "Unexplained bruising, prolonged bleeding from minor cuts, or hematuria",
        ],
        contraindications=["Active peptic ulceration", "Hemophilia or other bleeding diatheses", "Children < 16 years"],
        source_attribution="WHO Model List of Essential Medicines, BNF 86",
        reference_links=["https://bnf.nice.org.uk/drugs/aspirin/"],
    ),
    "montelukast": MedicineMonograph(
        medicine_name="Montelukast (M-Lucas / Singulair)",
        generic_name="Montelukast Sodium",
        therapeutic_class="Leukotriene Receptor Antagonist",
        atc_code="R03DC03",
        approved_indications=[
            "Prophylaxis and chronic treatment of asthma in adults and pediatric patients",
            "Prevention of exercise-induced bronchoconstriction (EIB)",
            "Relief of symptoms of allergic rhinitis",
        ],
        mechanism_of_action=(
            "Selectively binds to the cysteinyl leukotriene type 1 (CysLT1) receptor, inhibiting physiological "
            "actions of LTD4 and LTE4, dampening airway edema, smooth muscle contraction, and cellular inflammation."
        ),
        common_side_effects=[
            "Upper respiratory infection, cough, fever",
            "Headache, abdominal pain, diarrhea",
        ],
        serious_warnings=[
            "BOXED WARNING: Serious neuropsychiatric events including agitation, aggressive behavior, depression, and suicidal ideation",
            "Not indicated for the treatment of acute bronchospasm; patients must have rescue inhalers accessible",
        ],
        administration_guidelines=[
            "Take once daily in the evening for asthma; for allergic rhinitis, morning or evening",
            "Continue taking during asthma exacerbations as directed by physician",
        ],
        when_to_seek_doctor=[
            "Unusual changes in behavior or mood, vivid nightmares, depression, or suicidal thoughts",
            "Sudden acute asthma attack not relieved by rescue bronchodilator",
        ],
        contraindications=["Hypersensitivity to montelukast"],
        source_attribution="US FDA MedWatch Boxed Warning, BNF 86",
        reference_links=["https://bnf.nice.org.uk/drugs/montelukast/"],
    ),
    "rosuvastatin": MedicineMonograph(
        medicine_name="Rosuvastatin (Rozad / Crestor)",
        generic_name="Rosuvastatin Calcium",
        therapeutic_class="HMG-CoA Reductase Inhibitor (Statin)",
        atc_code="C10AA07",
        approved_indications=[
            "Primary hypercholesterolemia and mixed dyslipidemia",
            "Hypertriglyceridemia and slowing progression of atherosclerosis",
            "Primary prevention of major adverse cardiovascular events (stroke, myocardial infarction)",
        ],
        mechanism_of_action=(
            "Competitively inhibits HMG-CoA reductase, the rate-limiting enzyme in hepatic cholesterol biosynthesis. "
            "Upregulates hepatic LDL receptors on cell surfaces, increasing LDL clearance and lowering circulating ApoB and triglycerides."
        ),
        common_side_effects=[
            "Myalgia (muscle ache or weakness)",
            "Headache, dizziness, and mild abdominal discomfort",
            "Constipation or nausea",
        ],
        serious_warnings=[
            "Risk of myopathy and rhabdomyolysis, elevated with high doses (40mg) or interacting fibrates/cyclosporine",
            "Hepatotoxicity: monitor serum transaminases before and during therapy",
        ],
        administration_guidelines=[
            "Take once daily by mouth with or without food at any consistent time of day (e.g. 7 AM or bedtime)",
            "Swallow whole with a glass of water; do not crush or chew",
        ],
        when_to_seek_doctor=[
            "Unexplained muscle pain, tenderness, or weakness, particularly if accompanied by malaise or dark urine",
            "Yellowing of skin or eyes, severe fatigue, or persistent right upper quadrant pain",
        ],
        contraindications=["Active liver disease", "Pregnancy and breastfeeding", "Severe renal impairment"],
        source_attribution="British National Formulary (BNF 86), FDA DailyMed",
        reference_links=["https://bnf.nice.org.uk/drugs/rosuvastatin/"],
    ),
    "ambulox": MedicineMonograph(
        medicine_name="Ambroxol + Amoxicillin (Ambulox)",
        generic_name="Ambroxol Hydrochloride + Amoxicillin",
        therapeutic_class="Mucolytic + Penicillin Antibacterial",
        atc_code="R05CB06 / J01CA04",
        approved_indications=[
            "Acute and chronic respiratory tract infections accompanied by viscous bronchial secretions",
            "Productive bronchitis, bronchopneumonia, and bronchiectasis",
            "Exacerbations of chronic obstructive airway disorders with thick sputum",
        ],
        mechanism_of_action=(
            "Ambroxol stimulates pulmonary surfactant secretion and breaks acid mucopolysaccharide fibers in bronchial mucus, "
            "reducing viscosity. Amoxicillin inhibits bacterial cell wall peptidoglycan synthesis, killing susceptible pathogens."
        ),
        common_side_effects=[
            "Gastrointestinal discomfort, loose stools, nausea",
            "Transient mild rash",
            "Altered taste sensation (dysgeusia)",
        ],
        serious_warnings=[
            "Severe hypersensitivity reactions including anaphylaxis and angioedema in penicillin-sensitive patients",
            "Discontinue immediately if cutaneous blistering or mucosal lesions develop (Stevens-Johnson syndrome risk)",
        ],
        administration_guidelines=[
            "Take twice daily (1 BD) with a meal and ample fluids to assist mucus clearance",
            "Complete the entire prescribed antibacterial regimen without skipping doses",
        ],
        when_to_seek_doctor=[
            "Signs of severe allergy: facial swelling, wheezing, or difficulty swallowing",
            "Severe or watery diarrhea persisting longer than 24 hours",
        ],
        contraindications=["Penicillin or beta-lactam hypersensitivity", "Active peptic ulceration"],
        source_attribution="WHO Model Formulary, European Pharmacopoeia, Indian Pharmacopoeia",
        reference_links=["https://bnf.nice.org.uk/drugs/amoxicillin/"],
    ),
    "placida": MedicineMonograph(
        medicine_name="Flupentixol + Melitracen (Placida / Deanxit)",
        generic_name="Flupentixol Dihydrochloride + Melitracen Hydrochloride",
        therapeutic_class="Psychotropic / Anxiolytic-Antidepressant Combination",
        atc_code="N06CA02",
        approved_indications=[
            "Psychogenic depression, anxiety neurosis, and depressive neurosis",
            "Psychosomatic disorders with tension, anxiety, and visceral somatic symptoms (gastric / cardiac neurosis)",
            "Menopausal depression and asthenia associated with emotional tension",
        ],
        mechanism_of_action=(
            "Flupentixol acts as a selective antagonist at post-synaptic dopamine D1 and D2 receptors, exerting anxiolytic "
            "and activating properties at low doses. Melitracen is a bicyclic thymoleptic that blocks reuptake of norepinephrine and serotonin."
        ),
        common_side_effects=[
            "Mild restlessness, daytime sedation, or transient insomnia",
            "Anticholinergic symptoms: dry mouth, mild blurred vision, constipation",
            "Dizziness, mild tachycardia",
        ],
        serious_warnings=[
            "Avoid abrupt cessation after chronic therapy; taper dose under physician supervision",
            "Caution in patients with cardiac conduction disorders or history of convulsive disorders",
        ],
        administration_guidelines=[
            "Take once daily (1 OD), preferably in the morning or early afternoon to prevent sleep disturbance",
            "Take with water with or after food",
        ],
        when_to_seek_doctor=[
            "Marked motor restlessness, involuntary muscle spasms, or irregular heartbeat",
            "Sudden severe agitation, extreme dizziness, or confusion",
        ],
        contraindications=["Recent myocardial infarction, bundle-branch heart block", "Narrow-angle glaucoma, severe CNS depression"],
        source_attribution="Danish Medicines Agency (Deanxit SmPC), Martindale: The Complete Drug Reference",
        reference_links=["https://www.ema.europa.eu/"],
    ),
    "potil": MedicineMonograph(
        medicine_name="Clonazepam + Paroxetine (Potil PM / Petril Plus)",
        generic_name="Clonazepam + Paroxetine",
        therapeutic_class="Benzodiazepine Anxiolytic + SSRI Antidepressant",
        atc_code="N05BA09 / N06AB05",
        approved_indications=[
            "Comorbid anxiety neurosis and depressive disorders",
            "Panic disorder with anticipatory anxiety and panic attacks",
            "Severe anxiety associated with primary sleep disruption or nocturnal restlessness",
        ],
        mechanism_of_action=(
            "Clonazepam enhances GABA-A receptor sensitivity to inhibitory GABA neurotransmission, causing rapid sedative and "
            "anxiolytic effects. Paroxetine selectively blocks the neuronal reuptake of serotonin (5-HT), restoring neurotransmitter balance."
        ),
        common_side_effects=[
            "Drowsiness, morning sedation, mild motor unsteadiness (ataxia)",
            "Dry mouth, nausea, constipation",
            "Fatigue and reduced libido",
        ],
        serious_warnings=[
            "Risk of physical dependence and withdrawal symptoms upon abrupt discontinuation; always taper under medical advice",
            "Additive respiratory depression when combined with alcohol, opioids, or other sedatives",
        ],
        administration_guidelines=[
            "Take once daily at bedtime (1 OD HS) with a glass of water",
            "Do not drive or operate machinery while experiencing sedation or morning grogginess",
        ],
        when_to_seek_doctor=[
            "Excessive confusion, extreme somnolence, respiratory slowness, or worsening depressive thoughts",
            "Severe tremors, persistent vomiting, or loss of balance",
        ],
        contraindications=["Severe respiratory depression or myasthenia gravis", "Narrow-angle glaucoma, concurrent MAOI therapy"],
        source_attribution="British National Formulary (BNF 86), FDA DailyMed",
        reference_links=["https://bnf.nice.org.uk/drugs/clonazepam/"],
    ),
}


def explain_medicine(query_name: str) -> dict[str, Any]:
    """
    Retrieves grounded clinical explanation for a verified medicine name.
    If the exact monograph is not found, attempts a generic search or provides safe guidance.

    SAFETY CONSTRAINTS:
    - Never recommends starting, stopping, or changing a medicine.
    - Explicitly identifies uncatalogued medicines without guessing.
    - Grounded exclusively in authoritative clinical references.
    """
    clean_key = query_name.strip().lower()
    # Normalize synonyms and brand-to-generic mappings
    alias_map = {
        "napa": "paracetamol",
        "calpol": "paracetamol",
        "tylenol": "paracetamol",
        "acetaminophen": "paracetamol",
        "zimax": "azithromycin",
        "amox": "amoxicillin",
        "moxacil": "amoxicillin",
        "amoxil": "amoxicillin",
        "alatrol": "cetirizine",
        "zyrtec": "cetirizine",
        "indever": "propranolol",
        "inderen": "propranolol",
        "rivotril": "clonazepam",
        "losita": "losartan",
        "comet": "metformin",
        "exeptim": "cefixime",
        "ome": "omeprazole",
        "prilosec": "omeprazole",
        "dexter": "dexibuprofen",
        "escilex": "escitalopram",
        "nexcital": "escitalopram",
        "asynta": "aspirin",
        "m-lucas": "montelukast",
        "singulair": "montelukast",
        "rozad": "rosuvastatin",
        "crestor": "rosuvastatin",
        "ambulox": "ambulox",
        "potil": "potil",
        "potil pm": "potil",
        "petril": "potil",
        "placida": "placida",
        "deanxit": "placida",
        "esotop": "esomeprazole",
        "esotop hc": "esomeprazole",
        "nexium": "esomeprazole",
    }
    canonical_key = alias_map.get(clean_key, clean_key)

    if canonical_key in CLINICAL_MONOGRAPHS:
        mono = CLINICAL_MONOGRAPHS[canonical_key]
        return {
            "found": True,
            "title": mono.medicine_name,
            "generic": mono.generic_name,
            "class": mono.therapeutic_class,
            "atc": mono.atc_code,
            "indications": mono.approved_indications,
            "mechanism": mono.mechanism_of_action,
            "side_effects": mono.common_side_effects,
            "warnings": mono.serious_warnings,
            "administration": mono.administration_guidelines,
            "red_flags": mono.when_to_seek_doctor,
            "contraindications": mono.contraindications,
            "sources": mono.source_attribution,
            "references": mono.reference_links,
            "disclaimer": (
                "Academic Research Prototype: This medical explanation is compiled from reference clinical literature "
                "(WHO, BNF, OpenFDA) for educational and transcription assistance only. It does NOT constitute personalized medical "
                "advice, diagnosis, or prescription modification. Never start, stop, or change any medication "
                "without direct consultation with a licensed physician or clinical pharmacist."
            ),
        }

    return {
        "found": False,
        "title": query_name.title(),
        "generic": "Not available in local monograph index",
        "class": "Prescription Therapeutic Compound",
        "atc": "N/A",
        "indications": ["Information not available in the local offline reference database."],
        "mechanism": "Consult an official pharmacopeia or healthcare practitioner for specific mechanism of action.",
        "side_effects": ["Varies by specific formulation and dosage."],
        "warnings": ["Check with a licensed pharmacist or physician before taking."],
        "administration": ["Follow the exact instructions printed on the pharmacy label."],
        "red_flags": ["Seek emergency medical attention if you experience breathing difficulty, rash, or swelling."],
        "contraindications": ["Unknown for uncatalogued entry."],
        "sources": "General medical safety guidelines",
        "references": ["https://www.who.int/medicines/publications/essentialmedicines/en/"],
        "disclaimer": (
            "Academic Research Prototype: No confirmed offline monograph matched this specific entry. "
            "Do not guess or assume safety. Consult your prescribing physician or pharmacist."
        ),
    }
