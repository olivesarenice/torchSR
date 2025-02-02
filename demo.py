import os
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


def resize_image_to_match(image1, image2, resample_method=Image.BICUBIC):
    """
    Resize image1 (LR image) to match the size of image2 (HR image).

    Parameters:
    - image1: The low-resolution image to resize (PIL Image object).
    - image2: The high-resolution image to match the size (PIL Image object).
    - resample_method: Interpolation method to use (default: BICUBIC).

    Returns:
    - Resized low-resolution image (PIL Image object).
    """

    # Get the size (width, height) of the second image (HR image)
    target_size = image2.size

    # Resize the first image (LR image) to match the target size
    resized_image = image1.resize(target_size, resample=resample_method)

    return resized_image


def calculate_psnr_ssim(folder_in, folder_hr, upscale_type):

    # folder_in can be of
    # LR or HR

    psnr_list = []
    ssim_list = []

    # Get list of image files in both folders
    files_hr = sorted(os.listdir(folder_hr))  # Assuming identical filenames
    files_in = sorted(os.listdir(folder_in))
    print(upscale_type)
    for file_in, file_hr in zip(files_in, files_hr):
        print(file_in, file_hr)
        # Check if both files are images and have the same filename
        if file_in.endswith(("jpg", "jpeg", "png")) and file_hr.endswith(
            ("jpg", "jpeg", "png")
        ):
            hr_img = Image.open(os.path.join(folder_hr, file_hr)).convert("RGB")
            hr_img_np = np.array(hr_img)

            in_img = Image.open(os.path.join(folder_in, file_in)).convert("RGB")

            match upscale_type:  # This is only for LR folders
                case "lr_nn":
                    in_img_resized = np.array(
                        resize_image_to_match(in_img, hr_img, Image.NEAREST)
                    )
                case "lr_lanczos":
                    in_img_resized = np.array(
                        resize_image_to_match(in_img, hr_img, Image.LANCZOS)
                    )
                case "sr":  # This is for the already SR upscaled folder
                    in_img = resize_image_to_match(in_img, hr_img)
                    in_img_resized = np.array(in_img)

                case "sr_enhanced":
                    in_img = resize_image_to_match(in_img, hr_img)
                    in_img_resized = np.array(enhance(in_img))
            psnr_value = peak_signal_noise_ratio(
                hr_img_np, in_img_resized, data_range=255
            )  # PSNR between the resized LR and SR images
            ssim_value = structural_similarity(
                hr_img_np, in_img_resized, multichannel=True, win_size=3, data_range=255
            )  # SSIM comparison

            psnr_list.append(psnr_value)
            ssim_list.append(ssim_value)
    psnr = sum(psnr_list) / len(psnr_list)
    ssim = sum(ssim_list) / len(ssim_list)
    return {"psnr": float(round(psnr, 2)), "ssim": float(round(ssim, 3))}


def enhance(image):
    img = np.array(image)

    # A 5x5 kernel means that the morphological operation will look at a 5x5 pixel area around each pixel
    # (its "neighborhood") when applying the opening or closing operation.
    # This is a moderate kernel size, which is typically strong enough
    # to remove small noise or fill in small holes without being overly aggressive.
    kernel = np.ones((5, 5), np.uint8)

    img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    img_obj = Image.fromarray(img)
    return img_obj


def create_visualization(input_dir, output_dir, hr_dir, visualization_dir):
    # Create visualization folder if it does not exist
    if not os.path.exists(visualization_dir):
        os.makedirs(visualization_dir)

    # List all files in the input directory (assuming all folders have the same file names)
    filenames = os.listdir(input_dir)

    # Loop through each image file
    for filename in filenames:
        # Construct the full file paths
        input_image_path = os.path.join(input_dir, filename)
        output_image_path = os.path.join(output_dir, filename)
        hr_image_path = os.path.join(hr_dir, filename)

        try:
            # Open the images
            input_image = Image.open(input_image_path)
            output_image = Image.open(output_image_path)
            hr_image = Image.open(hr_image_path)

            # Ensure that all images have the same height (resize if necessary)
            height = max(input_image.height, output_image.height, hr_image.height)

            # Resize images to match the maximum height (preserving aspect ratio)
            input_nn_image = input_image.resize(
                (int(input_image.width * height / input_image.height), height),
                resample=Image.NEAREST,
            )
            input_bi_image = input_image.resize(
                (int(input_image.width * height / input_image.height), height)
            )
            output_image = output_image.resize(
                (int(output_image.width * height / output_image.height), height)
            )
            output_image_enhanced = enhance(output_image)
            hr_image = hr_image.resize(
                (int(hr_image.width * height / hr_image.height), height)
            )

            # Concatenate the images horizontally (side by side)
            combined_image = Image.new(
                "RGB",
                (
                    input_nn_image.width
                    + input_bi_image.width
                    + output_image.width
                    + output_image_enhanced.width
                    + hr_image.width,
                    height,
                ),
            )
            combined_image.paste(input_nn_image, (0, 0))
            combined_image.paste(input_bi_image, (input_nn_image.width, 0))
            combined_image.paste(
                output_image, (input_bi_image.width + input_nn_image.width, 0)
            )
            combined_image.paste(
                output_image_enhanced,
                (
                    output_image_enhanced.width
                    + input_bi_image.width
                    + input_nn_image.width,
                    0,
                ),
            )
            combined_image.paste(
                hr_image,
                (
                    input_bi_image.width
                    + input_nn_image.width
                    + output_image.width
                    + output_image_enhanced.width,
                    0,
                ),
            )

            # Save the combined image in the visualization folder
            combined_image.save(os.path.join(visualization_dir, filename))

            # print(f"Saved visualization for {filename}")

        except Exception as e:
            print(f"Error processing {filename}: {e}")


def load_model(model_name, scale):
    model_path = f"user_models/{model_name}_model.pt"
    model = ninasr_b2(
        scale=int(scale)
    )  # Use the correct scale for your model (8 in this case)
    model.load_state_dict(torch.load(model_path))
    model.eval()  # Set the model to evaluation mode
    return model


def resize_to_target(image, target_size=600):
    """
    Resizes the image based on its dimensions:
    - If smaller than target_size, upscale using Lanczos.
    - If larger than target_size, downscale using Bicubic.

    :param image: PIL Image object to resize.
    :param target_size: Threshold size for resizing.
    :return: Resized PIL Image object.
    """
    if max(image.size) < target_size:
        # Upscale using Lanczos
        new_size = (
            int(image.width * (target_size / max(image.size))),
            int(image.height * (target_size / max(image.size))),
        )
        return image.resize(new_size, Image.LANCZOS)
    elif max(image.size) > target_size:
        # Downscale using Bicubic
        new_size = (
            int(image.width * (target_size / max(image.size))),
            int(image.height * (target_size / max(image.size))),
        )
        return image.resize(new_size, Image.BICUBIC)
    return image  # Return original if exactly target_size


def generate_sr(input_dir, output_dir, model, sample=None):
    print(f"Generating SR for {input_dir} to {output_dir}")
    img_paths = list(paths.list_images(input_dir))
    if sample:
        img_paths = img_paths[0:sample]
    for img_path in img_paths:
        print(img_path)

        image = Image.open(img_path)
        lr_tensor = to_tensor(image).unsqueeze(0)  # Add batch dimension
        with torch.no_grad():  # Disable gradient calculation for inference
            sr_tensor = model(lr_tensor)  # Pass the LR tensor through the model
        sr_image = to_pil_image(sr_tensor.squeeze(0))  # Remove the batch dimension
        sr_image = resize_to_target(sr_image)
        save_path = os.path.join(output_dir, os.path.basename(img_path))
        sr_image.save(save_path)
    return None


def generate_enhanced_sr_wild(
    input_dir, output_dir, model, sample=None, show_lanczos=False
):
    """
    Generates SR-enhanced images with optional Lanczos resizing visualization.

    :param input_dir: Directory with input images.
    :param output_dir: Directory to save enhanced images.
    :param model: Super-resolution model for processing.
    :param sample: Number of images to process (default: all).
    :param show_lanczos: Whether to include a Lanczos-resized version in the output.
    """
    print(f"Generating SR enhanced for in the wild {input_dir} to {output_dir}")
    img_paths = list(paths.list_images(input_dir))
    if sample:
        img_paths = img_paths[:sample]

    for img_path in img_paths:
        print(f"Processing {img_path}")

        # Open the original image
        image = Image.open(img_path)

        # Prepare input tensor
        lr_tensor = to_tensor(image).unsqueeze(0)  # Add batch dimension

        # Perform super-resolution using the model
        with torch.no_grad():
            sr_tensor = model(lr_tensor)  # Pass the LR tensor through the model

        # Convert to PIL image
        sr_image = to_pil_image(sr_tensor.squeeze(0))  # Remove the batch dimension
        sr_image = resize_to_target(sr_image)  # Resize to target resolution

        # Apply enhancement
        en_sr_image = enhance(sr_image)

        # Optionally resize the original image with Lanczos
        if show_lanczos:
            lanczos_resized = image.resize(
                (en_sr_image.width, en_sr_image.height), resample=Image.LANCZOS
            )

        # Resize the original image for side-by-side display
        wild_image_resized = image.resize(
            (
                int(en_sr_image.width * en_sr_image.height / en_sr_image.height),
                en_sr_image.height,
            ),
            resample=Image.NEAREST,
        )

        # Create the combined image
        num_images = 3 if show_lanczos else 2
        combined_width = (
            wild_image_resized.width
            + (lanczos_resized.width if show_lanczos else 0)
            + en_sr_image.width
        )
        combined_image = Image.new("RGB", (combined_width, en_sr_image.height))

        combined_image.paste(wild_image_resized, (0, 0))
        if show_lanczos:
            combined_image.paste(lanczos_resized, (wild_image_resized.width, 0))
        combined_image.paste(
            en_sr_image,
            (
                wild_image_resized.width
                + (lanczos_resized.width if show_lanczos else 0),
                0,
            ),
        )

        # Save the combined image
        save_path = os.path.join(output_dir, os.path.basename(img_path))
        combined_image.save(save_path)
        print(f"Saved combined image to {save_path}")

    print("Wild vs. SR enhancement completed.")


if __name__ == "__main__":
    model_arch = "ninasr_b2"
    dataset_type = "bwpa"
    scale = 4
    model_name = f"{model_arch}_x{scale}_{dataset_type}"
    output_dir = f"/home/oliver/ADRA/experiments-superres/lib/torchSR/user_datasets/3_model_usage/{dataset_type}/UserSupplied/sr_output/{model_name}"
    input_dir = f"/home/oliver/ADRA/experiments-superres/lib/torchSR/user_datasets/3_model_usage/{dataset_type}/UserSupplied/UserSupplied_test_LR_bicubic/X{scale}"
    wild_in_dir = f"/home/oliver/ADRA/experiments-superres/lib/torchSR/user_datasets/1_source/{dataset_type}/profiled/low_res"
    wild_out_dir = f"/home/oliver/ADRA/experiments-superres/lib/torchSR/user_datasets/3_model_usage/{dataset_type}/UserSupplied/sr_output/{model_name}/wild/lo_res"
    hr_dir = f"/home/oliver/ADRA/experiments-superres/lib/torchSR/user_datasets/3_model_usage/{dataset_type}/UserSupplied/UserSupplied_test_HR"
    visualization_dir = os.path.join(output_dir, "visual")

    print(f"Evaluating testset for {model_name} on {input_dir}")

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(visualization_dir, exist_ok=True)
    os.makedirs(wild_out_dir, exist_ok=True)

    model = load_model(model_name, scale)

    generate_sr(input_dir, output_dir, model)

    create_visualization(input_dir, output_dir, hr_dir, visualization_dir)

    metrics = {
        "0_nn": calculate_psnr_ssim(input_dir, hr_dir, upscale_type="lr_nn"),
        "1_lanczos": calculate_psnr_ssim(input_dir, hr_dir, upscale_type="lr_lanczos"),
        "2_sr": calculate_psnr_ssim(output_dir, hr_dir, upscale_type="sr"),
        "3_sr_enhanced": calculate_psnr_ssim(
            output_dir, hr_dir, upscale_type="sr_enhanced"
        ),
    }

    pprint(metrics)

    generate_enhanced_sr_wild(
        wild_in_dir, wild_out_dir, model, sample=50, show_lanczos=True
    )
