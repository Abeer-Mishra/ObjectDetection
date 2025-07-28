import torch
from torchvision import models, transforms
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
from PIL import Image, ImageDraw, ImageFont
import requests
from io import BytesIO
import os

# -------------------------------
# 1. Load pre-trained model
# -------------------------------
weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
model = models.detection.fasterrcnn_resnet50_fpn(weights=weights)
model.eval()

# -------------------------------
# 2. Load image from URL or local path
# -------------------------------
def load_image(source):
    if source.startswith("http://") or source.startswith("https://"):
        try:
            response = requests.get(source)
            if response.status_code == 200 and "image" in response.headers.get("Content-Type", ""):
                return Image.open(BytesIO(response.content)).convert("RGB")
            else:
                raise Exception("❌ Invalid image URL or content type.")
        except Exception as e:
            raise Exception(f"Failed to load image from URL: {e}")
    elif os.path.exists(source):
        return Image.open(source).convert("RGB")
    else:
        raise FileNotFoundError("❌ The provided path or URL is not valid.")

# ✅ Set this to a local path or image URL
image_source = "<https://miro.medium.com/v2/resize:fit:1400/1*v0Bm-HQxWtpbQ0Yq463uqw.jpeg>"
# image_source = "<IMAGE FROM YOUR PC/DISK>"

# Load the image
image = load_image(image_source)

# -------------------------------
# 3. Transform image for model
# -------------------------------
transform = transforms.Compose([
    transforms.ToTensor()
])
img_tensor = transform(image).unsqueeze(0)  # Add batch dimension

# -------------------------------
# 4. Run object detection
# -------------------------------
with torch.no_grad():
    outputs = model(img_tensor)

# -------------------------------
# 5. COCO class labels
# -------------------------------
COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella',
    'N/A', 'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
    'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
    'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass', 'cup', 'fork',
    'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
    'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'N/A', 'dining table', 'N/A', 'N/A', 'toilet',
    'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

# -------------------------------
# 6. Draw boxes and labels
# -------------------------------
draw = ImageDraw.Draw(image)
font = ImageFont.load_default()
threshold = 0.5  # lower threshold to show more detections

detected = 0

for idx, box in enumerate(outputs[0]['boxes']):
    score = outputs[0]['scores'][idx].item()
    if score < threshold:
        continue

    label_idx = outputs[0]['labels'][idx].item()
    label = COCO_INSTANCE_CATEGORY_NAMES[label_idx]

    x1, y1, x2, y2 = box
    draw.rectangle(((x1, y1), (x2, y2)), outline="red", width=3)
    draw.text((x1, y1), f"{label} ({score:.2f})", fill="red", font=font)
    detected += 1

print(f"✅ Detected {detected} object(s)")

# -------------------------------
# 7. Show and save result
# -------------------------------
image.show()  # Show image in preview window
image.save("output_detected.jpg")  # Save output