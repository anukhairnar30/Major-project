import os
from io import BytesIO

import numpy as np
import torch
import streamlit as st
import matplotlib.pyplot as plt

from PIL import Image
from torchvision import transforms

from model import load_trained_model
from utils.explainability import generate_explanation


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Dermatological Image Analysis",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ============================================================
# CONSTANTS
# ============================================================

CLASS_NAMES = [
    "nv",
    "mel",
    "bkl",
    "bcc",
    "akiec",
    "vasc",
    "df"
]

CLASS_DESCRIPTIONS = {
    "nv": "Melanocytic Nevi",
    "mel": "Melanoma",
    "bkl": "Benign Keratosis-like Lesions",
    "bcc": "Basal Cell Carcinoma",
    "akiec": "Actinic Keratoses / Intraepithelial Carcinoma",
    "vasc": "Vascular Lesions",
    "df": "Dermatofibroma"
}

LOCALIZATIONS = [
    "abdomen",
    "acral",
    "back",
    "chest",
    "ear",
    "face",
    "foot",
    "genital",
    "hand",
    "lower extremity",
    "neck",
    "scalp",
    "trunk",
    "unknown",
    "upper extremity"
]

# Current inference values used in predict.py
AGE_MEAN = 50.05
AGE_STD = 16.42


# ============================================================
# IMAGE TRANSFORM
# ============================================================

image_transform = transforms.Compose([
    transforms.Resize((384, 384)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ============================================================
# SESSION STATE
# ============================================================

if "analysis_result" not in st.session_state:
    st.session_state.analysis_result = None

if "current_file" not in st.session_state:
    st.session_state.current_file = None


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model_path = os.path.join(
        os.path.dirname(__file__),
        "models",
        "multimodal_vit_bilstm_skin_cancer.pth"
    )

    model = load_trained_model(
        model_path,
        device
    )

    return model, device


model, device = load_model()


# ============================================================
# METADATA CREATION
# ============================================================

def create_metadata(
    age,
    sex,
    localization
):

    features = []

    # --------------------------------------------------------
    # AGE
    # --------------------------------------------------------

    normalized_age = (
        age - AGE_MEAN
    ) / AGE_STD

    features.append(
        normalized_age
    )

    # --------------------------------------------------------
    # SEX
    # --------------------------------------------------------

    features.append(
        1.0 if sex == "Female" else 0.0
    )

    features.append(
        1.0 if sex == "Male" else 0.0
    )

    features.append(
        1.0 if sex == "Unknown" else 0.0
    )

    # --------------------------------------------------------
    # LOCALIZATION
    # --------------------------------------------------------

    for location in LOCALIZATIONS:

        features.append(
            1.0
            if localization == location
            else 0.0
        )

    return torch.tensor(
        [features],
        dtype=torch.float32
    )


# ============================================================
# PDF REPORT
# ============================================================

def create_pdf_report(
    original_image,
    heatmap_image,
    predicted_class,
    confidence,
    probabilities,
    age,
    sex,
    localization
):

    try:

        from reportlab.lib import colors
        from reportlab.lib.enums import TA_CENTER
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import (
            getSampleStyleSheet,
            ParagraphStyle
        )
        from reportlab.lib.units import inch
        from reportlab.platypus import (
            SimpleDocTemplate,
            Paragraph,
            Spacer,
            Table,
            TableStyle,
            Image as ReportLabImage
        )

    except ImportError:

        raise ImportError(
            "reportlab is not installed."
        )

    # --------------------------------------------------------
    # PDF BUFFER
    # --------------------------------------------------------

    pdf_buffer = BytesIO()

    document = SimpleDocTemplate(
        pdf_buffer,
        pagesize=A4,
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=40
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "ReportTitle",
        parent=styles["Title"],
        alignment=TA_CENTER,
        fontSize=18,
        leading=22,
        spaceAfter=18
    )

    heading_style = ParagraphStyle(
        "ReportHeading",
        parent=styles["Heading2"],
        fontSize=13,
        leading=16,
        spaceBefore=12,
        spaceAfter=8
    )

    normal_style = ParagraphStyle(
        "ReportNormal",
        parent=styles["Normal"],
        fontSize=9.5,
        leading=13
    )

    story = []

    # --------------------------------------------------------
    # TITLE
    # --------------------------------------------------------

    story.append(
        Paragraph(
            "Deep Learning Solutions for Dermatological Image Analysis",
            title_style
        )
    )

    story.append(
        Paragraph(
            "Skin Lesion Analysis Report",
            styles["Heading2"]
        )
    )

    story.append(
        Spacer(1, 10)
    )

    # --------------------------------------------------------
    # ANALYSIS RESULT
    # --------------------------------------------------------

    story.append(
        Paragraph(
            "Analysis Result",
            heading_style
        )
    )

    result_data = [
        ["Parameter", "Result"],
        [
            "Predicted Class",
            predicted_class.upper()
        ],
        [
            "Description",
            CLASS_DESCRIPTIONS[predicted_class]
        ],
        [
            "Confidence",
            f"{confidence:.2f}%"
        ]
    ]

    result_table = Table(
        result_data,
        colWidths=[
            2.1 * inch,
            3.7 * inch
        ]
    )

    result_table.setStyle(
        TableStyle([
            (
                "BACKGROUND",
                (0, 0),
                (-1, 0),
                colors.HexColor("#e8eef7")
            ),
            (
                "TEXTCOLOR",
                (0, 0),
                (-1, 0),
                colors.HexColor("#1e293b")
            ),
            (
                "FONTNAME",
                (0, 0),
                (-1, 0),
                "Helvetica-Bold"
            ),
            (
                "FONTNAME",
                (0, 1),
                (0, -1),
                "Helvetica-Bold"
            ),
            (
                "GRID",
                (0, 0),
                (-1, -1),
                0.5,
                colors.HexColor("#cbd5e1")
            ),
            (
                "VALIGN",
                (0, 0),
                (-1, -1),
                "MIDDLE"
            ),
            (
                "PADDING",
                (0, 0),
                (-1, -1),
                7
            )
        ])
    )

    story.append(
        result_table
    )

    # --------------------------------------------------------
    # PATIENT INFORMATION
    # --------------------------------------------------------

    story.append(
        Paragraph(
            "Patient Information",
            heading_style
        )
    )

    input_data = [
        ["Parameter", "Value"],
        [
            "Age",
            f"{age} years"
        ],
        [
            "Sex",
            sex
        ],
        [
            "Lesion Localization",
            localization.title()
        ]
    ]

    input_table = Table(
        input_data,
        colWidths=[
            2.1 * inch,
            3.7 * inch
        ]
    )

    input_table.setStyle(
        TableStyle([
            (
                "BACKGROUND",
                (0, 0),
                (-1, 0),
                colors.HexColor("#e8eef7")
            ),
            (
                "FONTNAME",
                (0, 0),
                (-1, 0),
                "Helvetica-Bold"
            ),
            (
                "GRID",
                (0, 0),
                (-1, -1),
                0.5,
                colors.HexColor("#cbd5e1")
            ),
            (
                "PADDING",
                (0, 0),
                (-1, -1),
                7
            )
        ])
    )

    story.append(
        input_table
    )

    # --------------------------------------------------------
    # VISUAL EXPLANATION
    # --------------------------------------------------------

    story.append(
        Paragraph(
            "Visual Explanation",
            heading_style
        )
    )

    original_buffer = BytesIO()

    original_image.save(
        original_buffer,
        format="PNG"
    )

    original_buffer.seek(0)

    original_report_image = ReportLabImage(
        original_buffer,
        width=2.6 * inch,
        height=2.6 * inch
    )

    image_cells = [
        original_report_image
    ]

    if heatmap_image is not None:

        heatmap_buffer = BytesIO()

        heatmap_image.save(
            heatmap_buffer,
            format="PNG"
        )

        heatmap_buffer.seek(0)

        heatmap_report_image = ReportLabImage(
            heatmap_buffer,
            width=2.6 * inch,
            height=2.6 * inch
        )

        image_cells.append(
            heatmap_report_image
        )

    image_table = Table(
        [image_cells],
        colWidths=[
            2.8 * inch
        ] * len(image_cells)
    )

    image_table.setStyle(
        TableStyle([
            (
                "VALIGN",
                (0, 0),
                (-1, -1),
                "MIDDLE"
            ),
            (
                "ALIGN",
                (0, 0),
                (-1, -1),
                "CENTER"
            )
        ])
    )

    story.append(
        image_table
    )

    image_labels = [
        "Original Image"
    ]

    if heatmap_image is not None:

        image_labels.append(
            "Attention Heatmap"
        )

    label_table = Table(
        [image_labels],
        colWidths=[
            2.8 * inch
        ] * len(image_labels)
    )

    label_table.setStyle(
        TableStyle([
            (
                "ALIGN",
                (0, 0),
                (-1, -1),
                "CENTER"
            ),
            (
                "FONTNAME",
                (0, 0),
                (-1, -1),
                "Helvetica-Bold"
            ),
            (
                "FONTSIZE",
                (0, 0),
                (-1, -1),
                9
            )
        ])
    )

    story.append(
        label_table
    )

    # --------------------------------------------------------
    # CLASS PROBABILITIES
    # --------------------------------------------------------

    story.append(
        Paragraph(
            "Class Probabilities",
            heading_style
        )
    )

    probability_data = [
        ["Class", "Probability"]
    ]

    for class_name, probability in zip(
        CLASS_NAMES,
        probabilities
    ):

        probability_data.append([
            class_name,
            f"{probability * 100:.2f}%"
        ])

    probability_table = Table(
        probability_data,
        colWidths=[
            2.1 * inch,
            3.7 * inch
        ]
    )

    probability_table.setStyle(
        TableStyle([
            (
                "BACKGROUND",
                (0, 0),
                (-1, 0),
                colors.HexColor("#e8eef7")
            ),
            (
                "FONTNAME",
                (0, 0),
                (-1, 0),
                "Helvetica-Bold"
            ),
            (
                "GRID",
                (0, 0),
                (-1, -1),
                0.5,
                colors.HexColor("#cbd5e1")
            ),
            (
                "PADDING",
                (0, 0),
                (-1, -1),
                6
            )
        ])
    )

    story.append(
        probability_table
    )

    # --------------------------------------------------------
    # DISCLAIMER
    # --------------------------------------------------------

    story.append(
        Spacer(1, 20)
    )

    story.append(
        Paragraph(
            "For research and educational purposes only. "
            "This system is not intended to provide a medical diagnosis.",
            normal_style
        )
    )

    # --------------------------------------------------------
    # BUILD
    # --------------------------------------------------------

    document.build(
        story
    )

    pdf_buffer.seek(0)

    return pdf_buffer.getvalue()


# ============================================================
# HEADER
# ============================================================

st.title(
    "Deep Learning Solutions for Dermatological Image Analysis"
)

st.caption(
    "Skin lesion classification using image and patient metadata"
)


# ============================================================
# IMAGE UPLOAD
# ============================================================

uploaded_file = st.file_uploader(
    "Upload Skin Lesion Image",
    type=[
        "jpg",
        "jpeg",
        "png"
    ]
)


# ============================================================
# RESET RESULT WHEN NEW IMAGE IS UPLOADED
# ============================================================

if uploaded_file is not None:

    file_identifier = (
        uploaded_file.name,
        uploaded_file.size
    )

    if (
        st.session_state.current_file
        != file_identifier
    ):

        st.session_state.current_file = (
            file_identifier
        )

        st.session_state.analysis_result = None


# ============================================================
# MAIN
# ============================================================

if uploaded_file is not None:

    original_image = Image.open(
        uploaded_file
    ).convert("RGB")

    # ========================================================
    # PATIENT INFORMATION
    # ========================================================

    st.subheader(
        "Patient Information"
    )

    info_col1, info_col2, info_col3 = st.columns(3)

    with info_col1:

        age = st.number_input(
            "Age",
            min_value=1,
            max_value=100,
            value=50,
            step=1
        )

    with info_col2:

        sex = st.selectbox(
            "Sex",
            [
                "Male",
                "Female",
                "Unknown"
            ]
        )

    with info_col3:

        localization = st.selectbox(
            "Lesion Localization",
            LOCALIZATIONS
        )

    # ========================================================
    # INPUT IMAGE
    # ========================================================

    st.subheader(
        "Input Image"
    )

    image_col1, image_col2, image_col3 = st.columns(
        [1, 2, 1]
    )

    with image_col2:

        st.image(
            original_image,
            use_container_width=True
        )

    # ========================================================
    # ANALYZE
    # ========================================================

    analyze = st.button(
        "Analyze",
        type="primary",
        use_container_width=True
    )

    if analyze:

        with st.spinner(
            "Analyzing image..."
        ):

            try:

                # --------------------------------------------
                # IMAGE PREPROCESSING
                # --------------------------------------------

                image_tensor = image_transform(
                    original_image
                ).unsqueeze(0).to(device)

                # --------------------------------------------
                # METADATA
                # --------------------------------------------

                meta_tensor = create_metadata(
                    age,
                    sex,
                    localization
                ).to(device)

                # --------------------------------------------
                # PREDICTION
                # --------------------------------------------

                with torch.no_grad():

                    logits = model(
                        image_tensor,
                        meta_tensor
                    )

                    probabilities_tensor = torch.softmax(
                        logits,
                        dim=1
                    )[0]

                    predicted_index = torch.argmax(
                        probabilities_tensor
                    ).item()

                    predicted_class = (
                        CLASS_NAMES[
                            predicted_index
                        ]
                    )

                    confidence = (
                        probabilities_tensor[
                            predicted_index
                        ].item()
                        * 100
                    )

                    probabilities = (
                        probabilities_tensor
                        .detach()
                        .cpu()
                        .numpy()
                    )

                # --------------------------------------------
                # ATTENTION HEATMAP
                # --------------------------------------------

                heatmap_overlay = None

                try:

                    _, heatmap_overlay = (
                        generate_explanation(
                            model=model,
                            image_tensor=image_tensor,
                            meta_features=meta_tensor,
                            original_image=np.array(
                                original_image
                            )
                        )
                    )

                    if heatmap_overlay is not None:

                        heatmap_overlay = (
                            Image.fromarray(
                                heatmap_overlay
                            )
                        )

                except Exception as heatmap_error:

                    print(
                        "Heatmap error:",
                        heatmap_error
                    )

                    heatmap_overlay = None

                # --------------------------------------------
                # SAVE RESULT
                # --------------------------------------------

                st.session_state.analysis_result = {
                    "predicted_class": predicted_class,
                    "confidence": confidence,
                    "probabilities": probabilities,
                    "original_image": original_image.copy(),
                    "heatmap": heatmap_overlay,
                    "age": age,
                    "sex": sex,
                    "localization": localization
                }

            except Exception as error:

                st.error(
                    f"Analysis failed: {error}"
                )


# ============================================================
# DISPLAY RESULT
# ============================================================

result = (
    st.session_state.analysis_result
)


if result is not None:

    st.divider()

    # ========================================================
    # ANALYSIS RESULT
    # ========================================================

    st.subheader(
        "Analysis Result"
    )

    result_col1, result_col2 = st.columns(2)

    with result_col1:

        st.metric(
            "Predicted Class",
            result[
                "predicted_class"
            ].upper()
        )

        st.caption(
            CLASS_DESCRIPTIONS[
                result["predicted_class"]
            ]
        )

    with result_col2:

        st.metric(
            "Confidence",
            f'{result["confidence"]:.2f}%'
        )

    # ========================================================
    # VISUAL EXPLANATION
    # ========================================================

    st.subheader(
        "Visual Explanation"
    )

    visual_col1, visual_col2 = st.columns(2)

    with visual_col1:

        st.write(
            "Original Image"
        )

        st.image(
            result["original_image"],
            use_container_width=True
        )

    with visual_col2:

        st.write(
            "Attention Heatmap"
        )

        if result["heatmap"] is not None:

            st.image(
                result["heatmap"],
                use_container_width=True
            )

        else:

            st.info(
                "Attention heatmap is not available."
            )

    # ========================================================
    # CLASS PROBABILITY
    # ========================================================

    st.subheader(
        "Class Probability"
    )

    probabilities = (
        result["probabilities"]
    )

    # Highest probability first
    sorted_indices = np.argsort(
        probabilities
    )[::-1]

    sorted_classes = [
        CLASS_NAMES[i]
        for i in sorted_indices
    ]

    sorted_probabilities = (
        probabilities[
            sorted_indices
        ] * 100
    )

    fig, ax = plt.subplots(
        figsize=(9, 4.5)
    )

    bars = ax.barh(
        sorted_classes,
        sorted_probabilities
    )

    # Highest at top
    ax.invert_yaxis()

    ax.set_xlabel(
        "Probability (%)"
    )

    ax.set_ylabel(
        "Class"
    )

    ax.set_xlim(
        0,
        100
    )

    ax.grid(
        axis="x",
        alpha=0.2
    )

    # Percentage labels
    for bar, value in zip(
        bars,
        sorted_probabilities
    ):

        ax.text(
            min(
                value + 1,
                96
            ),
            bar.get_y()
            + bar.get_height() / 2,
            f"{value:.1f}%",
            va="center",
            fontsize=9
        )

    plt.tight_layout()

    st.pyplot(
        fig,
        use_container_width=True
    )

    plt.close(
        fig
    )

    # ========================================================
    # REPORT
    # ========================================================

    st.subheader(
        "Report"
    )

    try:

        pdf_data = create_pdf_report(
            original_image=result[
                "original_image"
            ],
            heatmap_image=result[
                "heatmap"
            ],
            predicted_class=result[
                "predicted_class"
            ],
            confidence=result[
                "confidence"
            ],
            probabilities=result[
                "probabilities"
            ],
            age=result[
                "age"
            ],
            sex=result[
                "sex"
            ],
            localization=result[
                "localization"
            ]
        )

        st.download_button(
            label="Download Report",
            data=pdf_data,
            file_name=(
                "dermatological_analysis_report.pdf"
            ),
            mime="application/pdf",
            type="primary",
            use_container_width=True
        )

    except ImportError:

        st.error(
            "PDF report requires reportlab."
        )

        st.code(
            "python -m pip install reportlab"
        )

    except Exception as error:

        st.error(
            f"Report generation failed: {error}"
        )

    # ========================================================
    # DISCLAIMER
    # ========================================================

    st.caption(
        "For research and educational purposes only. "
        "This system is not intended to provide a medical diagnosis."
    )