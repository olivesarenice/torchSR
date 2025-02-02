# This script will describe the statistics of your source data. Meant to be run from the repo root `torchSR/`
# It is useful for deciding the threshold at which to define hi-res vs lo-res images for the curation of super-res training dataset.

import math
import os
import shutil
from collections import Counter

import matplotlib.pyplot as plt
from PIL import Image

USER_FOLDER = "bwpa"  # Directory containing the images
DUMP_FOLDER = "original"
HI_RES_THRESHOLD = 600  # max(x,y)
LOWEST_ALLOWED_PX = 50  # Discard any images below this px
MAX_ALLOWED_PX = 1080  # Discard any images above this px
IMAGE_DIR = f"user_datasets/1_source/{USER_FOLDER}"  # constant
DUMP_DIR = f"{IMAGE_DIR}/{DUMP_FOLDER}"
DISCARD_DIR = f"{IMAGE_DIR}/profiled/discard"
HI_DIR = f"{IMAGE_DIR}/profiled/hi_res"
LO_DIR = f"{IMAGE_DIR}/profiled/low_res"

# Initialize lists
discard_files = []
discard_dimensions = []
os.makedirs(DISCARD_DIR, exist_ok=True)

lo_res_files = []
lo_res_dimensions = []
os.makedirs(HI_DIR, exist_ok=True)

hi_res_files = []
hi_res_dimensions = []
os.makedirs(LO_DIR, exist_ok=True)

all_res_dimensions = []

for filename in os.listdir(DUMP_DIR):
    if filename.endswith((".png", ".jpg", ".jpeg")):  # Add other formats as needed
        with Image.open(os.path.join(DUMP_DIR, filename)) as img:
            x, y = img.size
            length = max((x, y))
            all_res_dimensions.append(length)

            if LOWEST_ALLOWED_PX < length < HI_RES_THRESHOLD:
                lo_res_dimensions.append(length)
                lo_res_files.append(filename)
                shutil.copy(
                    os.path.join(DUMP_DIR, filename),
                    os.path.join(LO_DIR, filename),
                )

            elif HI_RES_THRESHOLD <= length <= MAX_ALLOWED_PX:
                hi_res_dimensions.append(length)
                hi_res_files.append(filename)
                shutil.copy(
                    os.path.join(DUMP_DIR, filename),
                    os.path.join(HI_DIR, filename),
                )

            # Discard if norm_length is outside range
            else:
                discard_dimensions.append(length)
                discard_files.append(filename)
                shutil.copy(
                    os.path.join(DUMP_DIR, filename),
                    os.path.join(DISCARD_DIR, filename),
                )

all_distribution = Counter(all_res_dimensions)
lo_distribution = Counter(lo_res_dimensions)
hi_distribution = Counter(hi_res_dimensions)
discard_distribution = Counter(discard_dimensions)
import numpy as np

binwidth = 50
bins = np.arange(min(all_distribution), max(all_distribution) + binwidth, binwidth)


# Create the plot
plt.figure(figsize=(10, 6))

# Low resolutions
plt.hist(
    lo_distribution.keys(),
    bins=bins,
    weights=lo_distribution.values(),
    alpha=0.7,
    color="orange",
    label="Low Resolutions",
)

# High resolutions
plt.hist(
    hi_distribution.keys(),
    bins=bins,
    weights=hi_distribution.values(),
    alpha=0.7,
    color="green",
    label="High Resolutions",
)

# Discarded resolutions
plt.hist(
    discard_distribution.keys(),
    bins=bins,
    weights=discard_distribution.values(),
    alpha=0.7,
    color="red",
    label="Discarded Resolutions",
)

# Add labels, legend, and limits
plt.title("Image Resolution Distribution")
plt.xlabel("Resolution (pixels)")
plt.ylabel("Frequency")
plt.legend(loc="upper right")

plt.grid(alpha=0.3)
plt.tight_layout()

# Show the plot
plt.show()
