"""
Generate a sample medical knowledge PDF for RAG ingestion.
Run this script once to create data/medical_book.pdf if you don't already
have a real medical reference PDF to use.

In production, replace this with a real medical textbook or dataset —
just make sure the final file is saved as data/medical_book.pdf, which is
the single filename the whole app (API, RAG pipeline, Docker, README) expects.
"""

import os
from pathlib import Path

# Single source of truth for the knowledge-base PDF path, matching
# app.py's DEFAULT_PDF_PATH and RAGPipeline's __main__ usage.
DEFAULT_OUTPUT_PATH = os.getenv("MEDICAL_PDF_PATH", "data/medical_book.pdf")


def create_sample_medical_pdf(output_path: str = DEFAULT_OUTPUT_PATH):
    """Create a sample medical knowledge PDF using reportlab."""
    if os.path.exists(output_path):
        print(
            f"'{output_path}' already exists — skipping sample generation "
            "so a real medical reference PDF is never overwritten. "
            "Delete the file first if you really want to regenerate the sample."
        )
        return

    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import inch
        from reportlab.lib import colors
    except ImportError:
        print("reportlab not installed. Run: pip install reportlab")
        return

    Path("data").mkdir(exist_ok=True)

    doc = SimpleDocTemplate(output_path, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []

    title_style = ParagraphStyle("title", parent=styles["Title"], fontSize=18, spaceAfter=12)
    h2_style = ParagraphStyle("h2", parent=styles["Heading2"], fontSize=14, spaceAfter=8, textColor=colors.darkblue)
    body_style = styles["BodyText"]
    body_style.spaceAfter = 6
    body_style.leading = 16

    medical_content = [
        ("MedAI Medical Knowledge Base", None),

        ("Chapter 1: Diabetes Mellitus", "h2"),
        ("""Diabetes mellitus is a group of metabolic diseases characterized by hyperglycemia
        resulting from defects in insulin secretion, insulin action, or both. The chronic hyperglycemia
        of diabetes is associated with long-term damage, dysfunction, and failure of various organs,
        especially the eyes, kidneys, nerves, heart, and blood vessels.

        Type 1 diabetes (T1D) results from cellular-mediated autoimmune destruction of the β-cells
        of the pancreas. Markers of immune destruction include islet cell autoantibodies, autoantibodies
        to insulin, autoantibodies to GAD (GAD65), and autoantibodies to the tyrosine phosphatases.

        Type 2 diabetes (T2D) is characterized by insulin resistance and relative (rather than absolute)
        insulin deficiency. Risk factors include obesity (BMI ≥ 30 kg/m²), physical inactivity, family
        history of diabetes, impaired glucose tolerance (fasting glucose 100-125 mg/dL), hypertension
        (≥140/90 mmHg), dyslipidemia (HDL < 35 mg/dL or triglycerides > 250 mg/dL), and history of
        gestational diabetes.

        Diagnostic criteria: Fasting plasma glucose ≥ 126 mg/dL (7.0 mmol/L), or 2-hour plasma glucose
        ≥ 200 mg/dL during an oral glucose tolerance test, or HbA1c ≥ 6.5%, or random plasma glucose
        ≥ 200 mg/dL with symptoms of hyperglycemia.""", "body"),

        ("Diabetes Management", "h2"),
        ("""Management of type 2 diabetes involves lifestyle modification (diet and exercise) and
        pharmacological therapy. Metformin is the preferred initial pharmacological agent for type 2
        diabetes. It works by decreasing hepatic glucose production and improving insulin sensitivity.
        
        Target glycemic goals: HbA1c < 7% for most patients, fasting glucose 80-130 mg/dL,
        postprandial glucose < 180 mg/dL. Blood pressure target: < 140/90 mmHg.
        Lipid management: LDL < 100 mg/dL (< 70 mg/dL for high cardiovascular risk).
        
        Complications of diabetes include: Diabetic retinopathy (leading cause of blindness in adults),
        diabetic nephropathy (leading cause of end-stage renal disease), diabetic neuropathy
        (most common complication, affects 60-70% of patients), and cardiovascular disease
        (accounts for 50% of deaths in diabetic patients).""", "body"),

        ("Chapter 2: Cardiovascular Disease", "h2"),
        ("""Cardiovascular disease (CVD) is the leading cause of death globally, accounting for
        17.9 million deaths per year. It includes coronary artery disease (CAD), heart failure,
        stroke, hypertension, and peripheral artery disease.

        Coronary artery disease is caused by atherosclerosis — buildup of plaques in coronary arteries.
        Risk factors include: hypertension, hyperlipidemia, smoking, diabetes, obesity, family history,
        age (men > 45, women > 55), and physical inactivity.

        Symptoms of CAD: Chest pain (angina) — pressure, squeezing, fullness in chest lasting minutes.
        Stable angina is predictable, triggered by exertion. Unstable angina is unpredictable, may occur
        at rest and represents an emergency. Myocardial infarction (MI) symptoms include severe chest
        pain radiating to arm/jaw, shortness of breath, nausea, diaphoresis.""", "body"),

        ("Heart Failure", "h2"),
        ("""Heart failure (HF) is a clinical syndrome characterized by the heart's inability to pump
        sufficient blood to meet the body's needs. Classified by ejection fraction: HFrEF (EF < 40%),
        HFmrEF (EF 40-49%), HFpEF (EF ≥ 50%).
        
        Common causes: CAD (most common), hypertension, cardiomyopathy, valvular heart disease.
        NYHA Classification: Class I (no limitation), Class II (slight limitation), 
        Class III (marked limitation), Class IV (symptoms at rest).
        
        BNP > 100 pg/mL or NT-proBNP > 300 pg/mL supports diagnosis.
        Management: ACE inhibitors/ARBs, beta-blockers, diuretics, aldosterone antagonists,
        SGLT2 inhibitors (now Class I recommendation in HFrEF).""", "body"),

        ("Chapter 3: Hypertension", "h2"),
        ("""Hypertension (HTN) is defined as systolic BP ≥ 130 mmHg and/or diastolic BP ≥ 80 mmHg
        (ACC/AHA 2017 guidelines). It affects approximately 1.28 billion adults worldwide.

        Classification (ACC/AHA 2017):
        - Normal: SBP < 120 and DBP < 80 mmHg
        - Elevated: SBP 120-129 and DBP < 80 mmHg  
        - Stage 1 HTN: SBP 130-139 or DBP 80-89 mmHg
        - Stage 2 HTN: SBP ≥ 140 or DBP ≥ 90 mmHg
        - Hypertensive crisis: SBP > 180 and/or DBP > 120 mmHg

        Primary (essential) hypertension accounts for 90-95% of cases. Secondary hypertension causes
        include renal artery stenosis, primary hyperaldosteronism, pheochromocytoma, sleep apnea.
        
        Target BP: < 130/80 mmHg for most patients. First-line agents: thiazide diuretics, 
        ACE inhibitors/ARBs, calcium channel blockers.""", "body"),

        ("Chapter 4: Common Symptoms and Red Flags", "h2"),
        ("""Chest Pain: Evaluate character (sharp, dull, pressure), location, radiation, onset,
        duration, alleviating/aggravating factors. Cardiac causes: ACS, pericarditis, aortic dissection.
        Non-cardiac: GERD, musculoskeletal, pleuritis, pneumothorax.
        
        Shortness of Breath (Dyspnea): Cardiac causes — heart failure, ACS, arrhythmia.
        Pulmonary causes — pneumonia, COPD exacerbation, pulmonary embolism, asthma.
        
        Syncope: Vasovagal (most common), orthostatic hypotension, cardiac arrhythmia, structural
        heart disease. Cardiac syncope carries higher mortality risk.
        
        Headache Red Flags (SNOOP): Systemic symptoms (fever, weight loss), Neurological symptoms,
        Onset sudden ("thunderclap"), Older (new onset > 50), Positional changes.""", "body"),

        ("Chapter 5: Respiratory Diseases", "h2"),
        ("""Pneumonia: Infection causing inflammation of lung parenchyma. Community-acquired pneumonia
        (CAP) most commonly caused by Streptococcus pneumoniae, Haemophilus influenzae, Mycoplasma
        pneumoniae, and respiratory viruses.
        
        Symptoms: Fever, productive cough, pleuritic chest pain, dyspnea. Examination: decreased breath
        sounds, crackles, dullness to percussion. Diagnosis: Chest X-ray (consolidation), CBC, cultures.
        PSI/PORT score and CURB-65 guide hospitalization decisions.
        
        COPD: Chronic obstructive pulmonary disease characterized by persistent respiratory symptoms
        and airflow limitation. Primary risk factor: cigarette smoking (>90% of cases).
        Diagnosis: Spirometry FEV1/FVC < 0.70 post-bronchodilator. GOLD staging based on FEV1.
        
        Asthma: Chronic inflammatory disease of airways. Characterized by variable airflow obstruction
        and airway hyperresponsiveness. Triggers: allergens, exercise, cold air, respiratory infections.""", "body"),

        ("Chapter 6: Neurological Conditions", "h2"),
        ("""Stroke: Sudden neurological deficit caused by focal brain ischemia or hemorrhage.
        Ischemic stroke (85%): thrombotic or embolic occlusion of cerebral artery.
        Hemorrhagic stroke (15%): intracerebral or subarachnoid hemorrhage.
        
        FAST acronym: Face drooping, Arm weakness, Speech difficulty, Time to call emergency.
        Time-sensitive treatment: IV tPA within 4.5 hours of symptom onset for ischemic stroke.
        Mechanical thrombectomy within 24 hours for large vessel occlusion.
        
        Migraine: Recurrent headache disorder with episodes lasting 4-72 hours. 
        Characterized by unilateral location, pulsating quality, moderate-to-severe intensity,
        aggravation by routine activity, and associated nausea/photophobia/phonophobia.
        Prophylaxis: topiramate, valproate, beta-blockers, amitriptyline, CGRP antagonists.""", "body"),

        ("Medical Disclaimer", "h2"),
        ("""This knowledge base is intended for educational and informational purposes only.
        It does not constitute medical advice and should not be used as a substitute for professional
        medical consultation, diagnosis, or treatment. Always seek the advice of your physician or
        other qualified health provider with any questions you may have regarding a medical condition.
        
        Never disregard professional medical advice or delay seeking it because of information
        provided in this knowledge base. If you think you may have a medical emergency, call your
        doctor or emergency services immediately.""", "body"),
    ]

    for content, content_type in medical_content:
        if content_type is None:
            story.append(Paragraph(content, title_style))
        elif content_type == "h2":
            story.append(Spacer(1, 0.2 * inch))
            story.append(Paragraph(content, h2_style))
        else:
            story.append(Paragraph(content.replace("\n", " "), body_style))
        story.append(Spacer(1, 0.1 * inch))

    doc.build(story)
    print(f"✅ Medical knowledge PDF created: {output_path}")
    print("   Run the /api/ingest endpoint to load it into Pinecone.")


if __name__ == "__main__":
    create_sample_medical_pdf()
