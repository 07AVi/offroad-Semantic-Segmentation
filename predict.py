import os
import cv2
import torch
import numpy as np
import segmentation_models_pytorch as smp

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ==== MODEL (MUST MATCH TRAIN.PY EXACTLY) ====
model = smp.DeepLabV3Plus(
    encoder_name="resnet34",
    encoder_weights=None,
    in_channels=3,
    classes=6,
).to(DEVICE)

model.load_state_dict(torch.load("best_model.pth", map_location=DEVICE))
model.eval()

# ==== PATHS ====
test_dir = "test/images"
output_dir = "predictions_overlay"
os.makedirs(output_dir, exist_ok=True)

# ==== COLOR MAP FOR 6 CLASSES ====
COLORS = np.array([
    [0, 0, 0],        # Class 0 - Black
    [255, 0, 0],      # Class 1 - Blue
    [0, 255, 0],      # Class 2 - Green
    [0, 0, 255],      # Class 3 - Red
    [255, 255, 0],    # Class 4 - Cyan
    [255, 0, 255],    # Class 5 - Pink
], dtype=np.uint8)

print("Starting Overlay Prediction...")

for img_name in os.listdir(test_dir):

    img_path = os.path.join(test_dir, img_name)
    image = cv2.imread(img_path)
    original = image.copy()

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (960, 544))  # divisible by 16
    image = image / 255.0

    image = torch.tensor(image).permute(2, 0, 1).unsqueeze(0).float().to(DEVICE)

    with torch.no_grad():
        output = model(image)
        pred = torch.argmax(output, dim=1).squeeze().cpu().numpy()

    # convert to color mask
    color_mask = COLORS[pred]

    # resize mask back to original size
    color_mask = cv2.resize(color_mask, (original.shape[1], original.shape[0]))

    # overlay
    overlay = cv2.addWeighted(original, 0.6, color_mask, 0.4, 0)

    save_path = os.path.join(output_dir, img_name)
    cv2.imwrite(save_path, overlay)

    print("Saved:", img_name)

print("Overlay Prediction Completed!")