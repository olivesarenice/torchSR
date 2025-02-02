import os
import shutil

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

DATASET = "bwpa"
MIN_BLUR_SCORE = 0.52
EXPOSURE_SCORE_RANGE = [0.2, 0.6]
INPUT_FOLDER = f"user_datasets/2_curate/{DATASET}/original"
BLUR_FOLDER = f"user_datasets/2_curate/{DATASET}/autocurate/blur"
EXPOSURE_FOLDER = f"user_datasets/2_curate/{DATASET}/autocurate/exposure"
PASS_FOLDER = f"user_datasets/2_curate/{DATASET}/autocurate/pass"

# Ensure folders exist
os.makedirs(BLUR_FOLDER, exist_ok=True)
os.makedirs(EXPOSURE_FOLDER, exist_ok=True)
os.makedirs(PASS_FOLDER, exist_ok=True)


def calculate_blur(img):
    # Convert the image to grayscale
    grayscale_img = img.convert("L")

    # Convert image to numpy array
    img_array = np.array(grayscale_img)

    # Perform FFT (Fast Fourier Transform) and shift zero frequency to the center
    f = np.fft.fft2(img_array)
    fshift = np.fft.fftshift(f)

    # Calculate the magnitude spectrum
    magnitude_spectrum = np.abs(fshift)

    # We can calculate the sum of high frequencies by excluding the center (low frequencies)
    rows, cols = magnitude_spectrum.shape
    center_x, center_y = rows // 2, cols // 2

    # Create a mask that excludes low-frequency components (center)
    mask_radius = 30  # You can experiment with this radius for tuning
    mask = np.ones((rows, cols))
    mask[
        center_x - mask_radius : center_x + mask_radius,
        center_y - mask_radius : center_y + mask_radius,
    ] = 0

    # Apply the mask to isolate high frequencies
    high_freq_spectrum = magnitude_spectrum * mask

    # The blur score is inversely proportional to the amount of high-frequency content
    high_freq_energy = np.sum(high_freq_spectrum)
    blur_score = high_freq_energy / np.sum(
        magnitude_spectrum
    )  # Normalize to get a score between 0 and 1

    return blur_score


def calculate_exposure(img):
    # Convert the image to grayscale
    grayscale_img = img.convert("L")

    # Convert to numpy array for easier manipulation
    img_array = np.array(grayscale_img)

    # Calculate brightness (average pixel intensity)
    brightness = np.mean(img_array) / 255.0  # Normalize to [0, 1]

    # Calculate contrast (standard deviation of pixel values)
    contrast = np.std(img_array) / 255.0  # Normalize to [0, 1]

    # Define weights for brightness and contrast (can be adjusted as needed)
    brightness_weight = 0.8
    contrast_weight = 0.2

    # Calculate final exposure score as a weighted sum of brightness and contrast
    exposure_score = (brightness_weight * brightness) + (contrast_weight * contrast)

    return exposure_score


def copy_to(file_name, folder):
    shutil.copy(file_name, folder)


def filter(file_name):
    img_path = os.path.join(INPUT_FOLDER, file_name)
    img = Image.open(img_path)
    exposure = calculate_exposure(img)
    blur = calculate_blur(img)

    if exposure > EXPOSURE_SCORE_RANGE[1] or exposure < EXPOSURE_SCORE_RANGE[0]:
        folder = "exposure"
        copy_to(img_path, EXPOSURE_FOLDER)
    elif blur < MIN_BLUR_SCORE:
        folder = "blur"
        copy_to(img_path, BLUR_FOLDER)
    else:
        folder = "pass"
        copy_to(img_path, PASS_FOLDER)

    details = {
        "file_name": file_name,
        "blur": blur,
        "exposure": exposure,
        "folder": folder,
    }
    return details


def apply_curation():
    to_blur = []
    to_exposure = []
    to_pass = []

    for file_name in os.listdir(INPUT_FOLDER):
        if file_name.lower().endswith(
            (".png", ".jpg", ".jpeg")
        ):  # Filter for image files
            details = filter(file_name)
            if details["folder"] == "blur":
                to_blur.append(details["blur"])
            elif details["folder"] == "exposure":
                to_exposure.append(details["exposure"])
            else:
                to_pass.append(details)

    print(f"Blur: {len(to_blur)}, Exposure: {len(to_exposure)}, Passed: {len(to_pass)}")


def get_distributions():
    blur_ls = []
    exposure_ls = []

    for file_name in os.listdir(INPUT_FOLDER):
        if file_name.lower().endswith(
            (".png", ".jpg", ".jpeg")
        ):  # Filter for image files
            img_path = os.path.join(INPUT_FOLDER, file_name)
            img = Image.open(img_path)
            blur_ls.append(calculate_blur(img))
            exposure_ls.append(calculate_exposure(img))

    # Plot histograms
    plt.figure(figsize=(12, 6))

    # Blur distribution
    plt.subplot(1, 2, 1)
    plt.hist(blur_ls, bins=20, color="blue", alpha=0.7)
    plt.title("Blur Score Distribution")
    plt.xlabel("Blur Score")
    plt.ylabel("Frequency")

    # Exposure distribution
    plt.subplot(1, 2, 2)
    plt.hist(exposure_ls, bins=20, color="orange", alpha=0.7)
    plt.title("Exposure Score Distribution")
    plt.xlabel("Exposure Score")
    plt.ylabel("Frequency")

    plt.tight_layout()
    plt.show()


# get_distributions()
apply_curation()
