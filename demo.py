import os

import numpy as np
import torch
from imutils import paths
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from skimage.transform import resize
from torchsr.models import ninasr_b0  # or import rcan if needed
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


def calculate_psnr_ssim(folder_sr, folder_hr, scale):
    psnr_list = []
    ssim_list = []

    # Get list of image files in both folders
    files_sr = sorted(os.listdir(folder_sr))  # Assuming identical filenames
    files_hr = sorted(os.listdir(folder_hr))

    for file_sr, file_hr in zip(files_sr, files_hr):
        # Check if both files are images and have the same filename
        if file_sr.endswith(("jpg", "jpeg", "png")) and file_hr.endswith(
            ("jpg", "jpeg", "png")
        ):
            # Open SR image
            sr_img = Image.open(os.path.join(folder_sr, file_sr)).convert("RGB")
            sr_img_np = np.array(sr_img)

            # Open LR image and convert to numpy array
            hr_img = Image.open(os.path.join(folder_hr, file_hr)).convert("RGB")

            # Resize LR image to match SR image resolution (upscale by a factor of `scale`)

            hr_img_resized = np.array(resize_image_to_match(hr_img, sr_img))

            # Compute PSNR and SSIM for upscaled LR and SR images
            psnr_value = psnr(
                hr_img_resized, sr_img_np, data_range=255
            )  # PSNR between the resized LR and SR images
            ssim_value = ssim(
                hr_img_resized, sr_img_np, multichannel=True, win_size=3, data_range=255
            )  # SSIM comparison

            psnr_list.append(psnr_value)
            ssim_list.append(ssim_value)

            # print(f"PSNR for {file_sr}: {psnr_value:.4f}")
            # print(f"SSIM for {file_sr}: {ssim_value:.4f}")

    return {"psnr": psnr_list, "ssim": ssim_list}


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
                hr_image,
                (input_bi_image.width + input_nn_image.width + output_image.width, 0),
            )

            # Save the combined image in the visualization folder
            combined_image.save(os.path.join(visualization_dir, filename))

            # print(f"Saved visualization for {filename}")

        except Exception as e:
            print(f"Error processing {filename}: {e}")


generate_sr = False
scale = "8"
model = f"ninasr_b0_x{scale}"
output_dir = f"/home/oliver/ADRA/experiments-superres/dataset/hi_res_focus_zip/bw/UserSupplied/sr_output/{model}"
input_dir = f"/home/oliver/ADRA/experiments-superres/dataset/hi_res_focus_zip/bw/UserSupplied/UserSupplied_test_LR_bicubic/X{scale}"
hr_dir = f"/home/oliver/ADRA/experiments-superres/dataset/hi_res_focus_zip/bw/UserSupplied/UserSupplied_test_HR"
visualization_dir = os.path.join(output_dir, "visual")

print(f"Evaluating testset for {model} on {input_dir}")

os.makedirs(output_dir, exist_ok=True)
os.makedirs(visualization_dir, exist_ok=True)

if generate_sr:
    # Load your trained model
    model_path = f"./{model}_model.pt"
    model = ninasr_b0(
        scale=int(scale)
    )  # Use the correct scale for your model (8 in this case)
    model.load_state_dict(torch.load(model_path))
    model.eval()  # Set the model to evaluation mode

    img_paths = list(paths.list_images(input_dir))
    for img_path in img_paths:
        print(img_path)

        # img_path = "/home/oliver/ADRA/experiments-superres/dataset/hi_res_focus_zip/bw/UserSupplied/UserSupplied_test_LR_bicubic/X8/266222200-003_2715_988.jpg"
        # img_path = "/home/oliver/ADRA/experiments-superres/dataset/pa_clean/lo_res/292608105-002_1139_275.jpg"
        # Open the image file
        image = Image.open(img_path)

        # Convert the PIL image to a tensor
        lr_tensor = to_tensor(image).unsqueeze(0)  # Add batch dimension

        # Run the super-resolution model (forward pass)
        with torch.no_grad():  # Disable gradient calculation for inference
            sr_tensor = model(lr_tensor)  # Pass the LR tensor through the model

        # Convert the super-resolved tensor back to a PIL image
        sr_image = to_pil_image(sr_tensor.squeeze(0))  # Remove the batch dimension

        # # Show the super-resolved image
        # sr_image.show()

        # Optionally, save the output image
        save_path = os.path.join(output_dir, os.path.basename(img_path))
        sr_image.save(save_path)

metrics = calculate_psnr_ssim(output_dir, hr_dir, scale=int(scale))
psnr = sum(metrics["psnr"]) / len(metrics["psnr"])
ssim = sum(metrics["ssim"]) / len(metrics["ssim"])

print(f"HR original vs. SR output: PSNR - {psnr} , SSIM - {ssim}")

# metrics = calculate_psnr_ssim(output_dir, input_dir, scale=8)
# psnr = sum(metrics["psnr"]) / len(metrics["psnr"])
# ssim = sum(metrics["ssim"]) / len(metrics["ssim"])

# print(f"LR downscaled vs. SR output: PSNR - {psnr} , SSIM - {ssim}")

create_visualization(input_dir, output_dir, hr_dir, visualization_dir)
