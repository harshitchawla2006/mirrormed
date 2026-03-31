import torch
import torch.nn as nn
import timm
import json
import numpy as np
import base64
import io
from torchvision import transforms
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

with open('class_map.json', 'r') as f:
    class_map = json.load(f)
    class_map = {int(k): v for k, v in class_map.items()}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Running on: {device}")

model = timm.create_model(
    'efficientnet_b3',
    pretrained  = False,
    num_classes = 7,
    drop_rate   = 0.3
)
model.load_state_dict(
    torch.load('best_model.pt', map_location=device)
)
model = model.to(device)
model.eval()
print("Model loaded!")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

def enable_dropout(model):
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()

def mc_predict(image_bytes, n_passes=10):
    image  = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    tensor = transform(image).unsqueeze(0).to(device)

    model.eval()
    enable_dropout(model)

    with torch.no_grad():
        preds = torch.stack([
            torch.softmax(model(tensor), dim=1)
            for _ in range(n_passes)
        ])

    mean        = preds.mean(0)
    variance    = preds.var(0)
    pred_class  = mean.argmax().item()
    confidence  = round(mean.max().item(), 4)
    uncertainty = round(variance.mean().item(), 6)

    heatmap_b64 = generate_gradcam(image, tensor, pred_class)

    return {
        "class"      : class_map[pred_class],
        "confidence" : confidence,
        "uncertainty": uncertainty,
        "heatmap"    : heatmap_b64
    }

def generate_gradcam(pil_image, tensor, pred_class):
    try:
        target_layer = [model.conv_head]
        cam = GradCAM(model=model, target_layers=target_layer)

        grayscale_cam = cam(input_tensor=tensor, targets=None)

        # resize original image to 224x224 and convert to float32 numpy
        img_resized = pil_image.resize((224, 224))
        img_np = np.array(img_resized, dtype=np.float32) / 255.0

        # overlay heatmap
        visualization = show_cam_on_image(img_np, grayscale_cam[0], use_rgb=True)

        # encode to base64
        vis_pil = Image.fromarray(visualization)
        buffer  = io.BytesIO()
        vis_pil.save(buffer, format='PNG')
        return base64.b64encode(buffer.getvalue()).decode('utf-8')

    except Exception as e:
        print(f"Grad-CAM error: {e}")
        return None