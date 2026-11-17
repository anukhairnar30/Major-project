import torch

from model import load_trained_model


MODEL_PATH = "models/multimodal_vit_bilstm_skin_cancer.pth"

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("Device:", device)
print("Loading model...")

model = load_trained_model(
    MODEL_PATH,
    device
)

print()
print("✅ MODEL LOADED SUCCESSFULLY")
print()
print("Backbone:", model.backbone.__class__.__name__)
print("Feature dimension:", model.backbone.num_features)
print("Metadata dimension: 19")
print("Number of classes: 7")