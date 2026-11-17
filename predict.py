import os

import numpy as np
import torch
from PIL import Image
import torchvision.transforms as transforms

from model import load_trained_model


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "models",
    "multimodal_vit_bilstm_skin_cancer.pth"
)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# CLASS NAMES
# Exact order from training notebook
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


# ============================================================
# IMAGE PREPROCESSING
# Exact validation/test preprocessing
# ============================================================

transform = transforms.Compose([
    transforms.Resize((384, 384)),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[
            0.485,
            0.456,
            0.406
        ],
        std=[
            0.229,
            0.224,
            0.225
        ]
    )
])


# ============================================================
# METADATA COLUMNS
# Exact order used during training
# ============================================================

META_COLUMNS = [
    "age",

    "sex_female",
    "sex_male",
    "sex_unknown",

    "localization_abdomen",
    "localization_acral",
    "localization_back",
    "localization_chest",
    "localization_ear",
    "localization_face",
    "localization_foot",
    "localization_genital",
    "localization_hand",
    "localization_lower extremity",
    "localization_neck",
    "localization_scalp",
    "localization_trunk",
    "localization_unknown",
    "localization_upper extremity"
]


# ============================================================
# AGE NORMALIZATION
# ============================================================

AGE_MEAN = 50.05
AGE_STD = 16.42


def normalize_age(age):

    return (
        float(age) - AGE_MEAN
    ) / AGE_STD


# ============================================================
# CREATE METADATA VECTOR
# ============================================================

def create_metadata(
    age,
    sex,
    localization
):

    # Start with zeros
    metadata = {
        column: 0.0
        for column in META_COLUMNS
    }

    # --------------------------------------------------------
    # AGE
    # --------------------------------------------------------

    metadata["age"] = normalize_age(
        age
    )

    # --------------------------------------------------------
    # SEX
    # --------------------------------------------------------

    sex = sex.lower().strip()

    if sex == "female":

        metadata["sex_female"] = 1.0

    elif sex == "male":

        metadata["sex_male"] = 1.0

    else:

        metadata["sex_unknown"] = 1.0

    # --------------------------------------------------------
    # LOCALIZATION
    # --------------------------------------------------------

    localization_map = {

        "Abdomen":
            "localization_abdomen",

        "Acral":
            "localization_acral",

        "Back":
            "localization_back",

        "Chest":
            "localization_chest",

        "Ear":
            "localization_ear",

        "Face":
            "localization_face",

        "Foot":
            "localization_foot",

        "Genital":
            "localization_genital",

        "Hand":
            "localization_hand",

        "Lower extremity":
            "localization_lower extremity",

        "Neck":
            "localization_neck",

        "Scalp":
            "localization_scalp",

        "Trunk":
            "localization_trunk",

        "Unknown":
            "localization_unknown",

        "Upper extremity":
            "localization_upper extremity"
    }

    if localization in localization_map:

        column = localization_map[
            localization
        ]

        metadata[column] = 1.0

    else:

        metadata[
            "localization_unknown"
        ] = 1.0

    # --------------------------------------------------------
    # NUMPY ARRAY
    # Exact training order
    # --------------------------------------------------------

    metadata_array = np.array(
        [
            metadata[column]
            for column in META_COLUMNS
        ],
        dtype=np.float32
    )

    return metadata_array


# ============================================================
# LOAD TRAINED MODEL
# ============================================================

print("==========================================")
print("Loading Dermatological AI Model")
print("==========================================")

print(
    "Device:",
    DEVICE
)

print(
    "Loading trained model..."
)

model = load_trained_model(
    MODEL_PATH,
    DEVICE
)

print(
    "Model loaded successfully."
)

print(
    "Classes:",
    CLASS_NAMES
)

print(
    "Metadata dimension:",
    len(META_COLUMNS)
)

print("==========================================")


# ============================================================
# PREDICTION FUNCTION
# ============================================================

def predict(
    image,
    age,
    sex,
    localization
):

    # --------------------------------------------------------
    # 1. IMAGE PREPROCESSING
    # --------------------------------------------------------

    image_tensor = transform(
        image
    ).unsqueeze(0).to(DEVICE)

    # --------------------------------------------------------
    # 2. METADATA
    # --------------------------------------------------------

    metadata_array = create_metadata(
        age,
        sex,
        localization
    )

    metadata_tensor = torch.tensor(
        metadata_array,
        dtype=torch.float32
    ).unsqueeze(0).to(DEVICE)

    # --------------------------------------------------------
    # 3. MODEL INFERENCE
    # --------------------------------------------------------

    with torch.no_grad():

        logits = model(
            image_tensor,
            metadata_tensor
        )

        probabilities = torch.softmax(
            logits,
            dim=1
        )

    # --------------------------------------------------------
    # 4. PREDICTED CLASS
    # --------------------------------------------------------

    predicted_index = torch.argmax(
        probabilities,
        dim=1
    ).item()

    # --------------------------------------------------------
    # 5. CONFIDENCE
    # --------------------------------------------------------

    confidence = probabilities[
        0,
        predicted_index
    ].item()

    # --------------------------------------------------------
    # 6. ALL PROBABILITIES
    # --------------------------------------------------------

    probability_array = (
        probabilities[0]
        .cpu()
        .numpy()
    )

    return (
        predicted_index,
        CLASS_NAMES[predicted_index],
        confidence,
        probability_array
    )


# ============================================================
# DIRECT TEST
# ============================================================

if __name__ == "__main__":

    print()
    print("predict.py is ready.")
    print("Model: Multimodal ViT-BiLSTM")
    print("Device:", DEVICE)
    print()
    print(
        "Use predict() from backend/main.py."
    )