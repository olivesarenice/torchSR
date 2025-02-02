import os
import random

import cv2
import numpy as np
from imutils import paths
from PIL import Image


def downsample_folder(input_dir, scales, output_dir):
    # Get all image paths
    img_paths = list(paths.list_images(input_dir))

    # Shuffle the image paths for a random split
    # random.seed(42)
    # random.shuffle(img_paths)

    # Split the dataset into 80% train, 10% validation, 10% test
    num_images = len(img_paths)
    train_split = int(0.8 * num_images)
    val_split = int(0.9 * num_images)

    # Create lists for each split
    train_images = img_paths[:train_split]
    val_images = img_paths[train_split:val_split]
    test_images = img_paths[val_split:]
    print(
        f"train: {len(train_images)} | val: {len(val_images)} | test: {len(test_images)}"
    )
    # Process each split and save HR images in folders
    for split, image_paths in zip(
        ["train", "val", "test"], [train_images, val_images, test_images]
    ):
        # Create directories for high-resolution (HR) images
        hr_split_dir = os.path.join(output_dir, f"UserSupplied_{split}_HR")
        os.makedirs(hr_split_dir, exist_ok=True)

        for img_path in image_paths:
            # Load the high-resolution image
            image = cv2.imread(img_path)
            (h, w) = image.shape[:2]

            # Crop the image to ensure dimensions are divisible by the largest scale factor
            w = w - (w % max(scales))
            h = h - (h % max(scales))
            image = image[0:h, 0:w]
            img_name = os.path.basename(img_path)
            hr_img_path = os.path.join(hr_split_dir, img_name)
            cv2.imwrite(hr_img_path, image)

    # Process each split and save LR images in folders for each scale
    for scale in scales:
        for split, image_paths in zip(
            ["train", "val", "test"], [train_images, val_images, test_images]
        ):
            # Create directories for low-resolution (LR) images at the specified scale
            lr_split_dir = os.path.join(
                output_dir,
                f"UserSupplied_{split}_LR_bicubic",
                f"X{scale}",
            )
            os.makedirs(lr_split_dir, exist_ok=True)

            for img_path in image_paths:
                # Load the high-resolution image
                image = cv2.imread(img_path)
                (h, w) = image.shape[:2]

                # Crop the image to be divisible by the scale
                w -= int(w % scale)
                h -= int(h % scale)
                image = image[0:h, 0:w]

                # Downscale the image by the scale factor
                lowW = int(w * (1.0 / scale))
                lowH = int(h * (1.0 / scale))
                highW = int(lowW * scale)
                highH = int(lowH * scale)

                # Perform the actual scaling (downscale and then upscale back)
                scaled = np.array(
                    Image.fromarray(image).resize((lowW, lowH), resample=Image.BICUBIC)
                )
                # scaled = np.array(
                #     Image.fromarray(scaled).resize(
                #         (highW, highH), resample=Image.BICUBIC
                #     )
                # )

                # Save the downscaled LR image
                img_name = os.path.basename(img_path)
                lr_img_path = os.path.join(lr_split_dir, img_name)
                cv2.imwrite(lr_img_path, scaled)


if __name__ == "__main__":
    DATASET = "bwpa"
    SCALE = [4]  # You can adjust the scale value here
    OUTPUT_DIR = f"user_datasets/3_model_usage/{DATASET}/UserSupplied"  # Replace with your desired output directory
    INPUT_DIR = f"user_datasets/2_curate/{DATASET}/autocurate/pass"

    # Copies over the HR images, splits them into train test val, then creates a downsampled version of each HR image in another folder.
    downsample_folder(INPUT_DIR, SCALE, OUTPUT_DIR)
