import numpy as np
from PIL import Image
import os

npz_path = '/opt/consistency_models/samples/clean_64/samples_100x64x64x3.npz'
data = np.load(npz_path)

images = data['arr_0']

output_dir = '/opt/consistency_models/samples/clean_64/images_converted'

print(f"Creating output dir: {output_dir}")
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
print(f"Output dir {output_dir} created")
print(f"Starting to convert {len(images)} images")
for i in range(len(images)):
    print(f"Converting image {i}")
    img = Image.fromarray(images[i])
    img_path = f'{output_dir}/sample_{i + 1}.png'
    img.save(img_path)

print("Images conversion completed")
