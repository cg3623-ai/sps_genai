"""Image classifier for the /classify API endpoint.

Loads the CNN trained in Part 2 (app/cifar10_cnn.pt) and predicts
the CIFAR10 class of an uploaded image.
"""

import io
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from app.cnn_model import CNNClassifier

CIFAR10_CLASSES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck',
]

# same preprocessing as training (Part 2)
PREPROCESS = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2023, 0.1994, 0.2010)),
])


class ImageClassifier:
    def __init__(self, weights_path=None):
        if weights_path is None:
            weights_path = Path(__file__).parent / 'cifar10_cnn.pt'
        self.model = CNNClassifier(num_classes=10)
        self.model.load_state_dict(torch.load(weights_path, map_location='cpu'))
        self.model.eval()

    def predict(self, image_bytes: bytes) -> dict:
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        x = PREPROCESS(image).unsqueeze(0)  # add batch dim
        with torch.no_grad():
            probs = torch.softmax(self.model(x), dim=1)
            confidence, idx = probs.max(dim=1)
        class_id = int(idx.item())
        return {
            'predicted_class': class_id,
            'class_name': CIFAR10_CLASSES[class_id],
            'confidence': round(float(confidence.item()), 4),
        }
