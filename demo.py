import torch
from PIL import Image
from torchsr.models import ninasr_b0  # or import rcan if needed
from torchvision.transforms.functional import to_pil_image, to_tensor

# Load your trained model
model_path = "./ninasr_b0_x8_model.pt"
model = ninasr_b0(scale=8)  # Use the correct scale for your model (8 in this case)
model.load_state_dict(torch.load(model_path))
model.eval()  # Set the model to evaluation mode

# Provide the file path to the low-resolution image
image_path = "/home/oliver/ADRA/experiments-superres/dataset/hi_res_focus_zip/bw/UserSupplied/UserSupplied_test_LR_bicubic/X8/266222200-003_2715_988.jpg"
image_path = "/home/oliver/ADRA/experiments-superres/dataset/pa_clean/lo_res/292608105-002_1139_275.jpg"
# Open the image file
image = Image.open(image_path)

# Convert the PIL image to a tensor
lr_tensor = to_tensor(image).unsqueeze(0)  # Add batch dimension

# Run the super-resolution model (forward pass)
with torch.no_grad():  # Disable gradient calculation for inference
    sr_tensor = model(lr_tensor)  # Pass the LR tensor through the model

# Convert the super-resolved tensor back to a PIL image
sr_image = to_pil_image(sr_tensor.squeeze(0))  # Remove the batch dimension

# Show the super-resolved image
sr_image.show()

# Optionally, save the output image
sr_image.save("super_resolved_image.jpg")
