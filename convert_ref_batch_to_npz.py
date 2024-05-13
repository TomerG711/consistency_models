import numpy as np
from PIL import Image

# Define the image size for resizing
image_size = 64

# Load the original NPZ file
original_npz = np.load('/opt/consistency_models/samples/lsun_bedroom_256_ref_set/VIRTUAL_lsun_bedroom256.npz')

# Extract the images from the NPZ file
images = original_npz['arr_0']

# Initialize an empty array to store resized images
resized_images = []

print("Starting NPZ ref conversion")
i=0
# Resize each image and append to resized_images
for img in images:
    img = Image.fromarray(img)
    width, height = img.size
    scale = image_size / min(width, height)
    img = img.resize(
        (int(round(scale * width)), int(round(scale * height))),
        resample=Image.BOX,
    )
    arr = np.array(img)
    h, w, _ = arr.shape
    h_off = (h - image_size) // 2
    w_off = (w - image_size) // 2
    arr = arr[h_off: h_off + image_size, w_off: w_off + image_size]
    print(f"Converted image {i}")
    resized_images.append(arr)
    i+=1

# Convert the resized images list to a NumPy array
resized_images = np.array(resized_images)

# Save the resized images to a new NPZ file
np.savez(
    '/opt/consistency_models/samples/lsun_bedroom_256_ref_set/VIRTUAL_lsun_bedroom256_resized_to_64_images_only.npz',
    arr_0=resized_images,
)
