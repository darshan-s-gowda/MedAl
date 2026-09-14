"""
MedAI Patient Medical Report Generator

Creates a PDF containing:
- Patient information
- Diabetes prediction
- Heart disease prediction
- Cardiovascular risk score
- Medical disclaimer

This is a report of AI-generated screening results,
not a medical diagnosis.
"""

from io import BytesIO
from datetime import datetime

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)


def generate_patient_report(patient_data, results):
    """
    Generate a patient assessment PDF in memory.

    Args:
        patient_data: Dictionary containing patient information.
        results: Dictionary containing prediction results.

    Returns:
        BytesIO containing the generated PDF.
    """

    buffer = BytesIO()

    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=18 * mm,
        leftMargin=18 * mm,
        topMargin=18 * mm,
        bottomMargin=18 * mm,
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "ReportTitle",
        parent=styles["Title"],
        fontSize=20,
        leading=24,
        alignment=TA_CENTER,
        spaceAfter=8,
    )

    subtitle_style = ParagraphStyle(
        "Subtitle",
        parent=styles["Normal"],
        fontSize=9,
        alignment=TA_CENTER,
        spaceAfter=18,
    )

    heading_style = ParagraphStyle(
        "Heading",
        parent=styles["Heading2"],
        fontSize=13,
        leading=16,
        spaceBefore=10,
        spaceAfter=8,
    )

    normal_style = ParagraphStyle(
        "NormalReport",
        parent=styles["Normal"],
        fontSize=9.5,
        leading=14,
    )

    disclaimer_style = ParagraphStyle(
        "Disclaimer",
        parent=styles["Normal"],
        fontSize=8,
        leading=11,
    )

    story = []

    # --------------------------------------------------
    # Header
    # --------------------------------------------------

    story.append(Paragraph("MedAI", title_style))
    story.append(
        Paragraph(
            "AI-Assisted Patient Risk Assessment Report",
            subtitle_style,
        )
    )

    story.append(
        Paragraph(
            f"Generated: {datetime.now().strftime('%d %B %Y, %I:%M %p')}",
            normal_style,
        )
    )

    story.append(Spacer(1, 8))

    # --------------------------------------------------
    # Patient Information
    # --------------------------------------------------

    story.append(
        Paragraph("Patient Information", heading_style)
    )

    patient_rows = [
        ["Field", "Value"],
        ["Name", str(patient_data.get("name", "Not provided"))],
        ["Age", str(patient_data.get("age", "Not provided"))],
        ["Sex", str(patient_data.get("sex", "Not provided"))],
        ["BMI", str(patient_data.get("bmi", "Not provided"))],
    ]

    patient_table = Table(
        patient_rows,
        colWidths=[55 * mm, 105 * mm],
    )

    patient_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#263238")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1),
                 [colors.whitesmoke, colors.HexColor("#eeeeee")]),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )

    story.append(patient_table)

    # --------------------------------------------------
    # Assessment Results
    # --------------------------------------------------

    story.append(
        Paragraph("AI Assessment Results", heading_style)
    )

    diabetes = results.get("diabetes", {})
    heart = results.get("heart_disease", {})
    cardiovascular = results.get("cardiovascular", {})

    result_rows = [
        ["Assessment", "Result", "Probability"],
        [
            "Diabetes",
            str(diabetes.get("prediction", "N/A")),
            f"{diabetes.get('probability', 'N/A')}%",
        ],
        [
            "Heart Disease",
            str(heart.get("prediction", "N/A")),
            f"{heart.get('probability', 'N/A')}%",
        ],
        [
            "Cardiovascular Risk",
            str(cardiovascular.get("risk_level", "N/A")),
            f"{cardiovascular.get('risk_percentage', 'N/A')}%",
        ],
    ]

    result_table = Table(
        result_rows,
        colWidths=[65 * mm, 50 * mm, 45 * mm],
    )

    result_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#263238")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("ALIGN", (2, 1), (2, -1), "CENTER"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 7),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
            ]
        )
    )

    story.append(result_table)

    # --------------------------------------------------
    # Risk Factors
    # --------------------------------------------------

    factors = cardiovascular.get("factors", [])

    if factors:
        story.append(
            Paragraph("Identified Risk Factors", heading_style)
        )

        for factor in factors:
            story.append(
                Paragraph(
                    f"• {factor}",
                    normal_style,
                )
            )

    # --------------------------------------------------
    # Disclaimer
    # --------------------------------------------------

    story.append(Spacer(1, 18))

    story.append(
        Paragraph(
            "<b>Important Medical Disclaimer</b>",
            heading_style,
        )
    )

    story.append(
        Paragraph(
            "This report contains AI-generated screening results for "
            "educational and informational purposes. These results are "
            "not a medical diagnosis and should not be used as a substitute "
            "for examination, diagnosis, or treatment by a qualified "
            "healthcare professional. Please consult an appropriate "
            "healthcare professional for interpretation of these results.",
            disclaimer_style,
        )
    )

    doc.build(story)

    buffer.seek(0)

    return buffer