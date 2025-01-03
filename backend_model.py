import base64
import os
from io import BytesIO
from pprint import pprint

import cv2
import numpy as np
import torch
from imutils import paths
from PIL import Image, ImageEnhance
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from skimage.transform import resize
from torchsr.models import ninasr_b0, ninasr_b2  # or import rcan if needed
from torchvision.transforms.functional import to_pil_image, to_tensor

model_arch = "ninasr_b2"
dataset_type = "bwpa"
scale = 8
model_name = f"{model_arch}_x{scale}_{dataset_type}"


def load_model(model_name=model_name, scale=scale):
    model_path = f"./{model_name}_model.pt"
    model = ninasr_b2(
        scale=int(scale)
    )  # Use the correct scale for your model (8 in this case)
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()  # Set the model to evaluation mode
    return model


def base64_to_image(img_base64):
    """Converts a base64 string to a PIL Image object."""
    img_data = base64.b64decode(img_base64)
    return Image.open(BytesIO(img_data))


def image_to_base64(image):
    """Converts a PIL Image object to a base64 string."""
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def enhance(image):
    img = np.array(image)
    print("Enhancing SR")
    # A 5x5 kernel means that the morphological operation will look at a 5x5 pixel area around each pixel
    # (its "neighborhood") when applying the opening or closing operation.
    # This is a moderate kernel size, which is typically strong enough
    # to remove small noise or fill in small holes without being overly aggressive.
    kernel = np.ones((5, 5), np.uint8)

    img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    img_obj = Image.fromarray(img)
    return img_obj


def run_inference(img_base64, model):
    print("Applying SR")
    image = base64_to_image(img_base64)
    lr_tensor = to_tensor(image).unsqueeze(0)  # Add batch dimension
    with torch.no_grad():  # Disable gradient calculation for inference
        sr_tensor = model(lr_tensor)  # Pass the LR tensor through the model
    sr_image = to_pil_image(sr_tensor.squeeze(0))  # Remove the batch dimension

    return image_to_base64(enhance(sr_image))


MODEL = load_model()
