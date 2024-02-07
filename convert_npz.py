import numpy as np
from PIL import Image
import os

npz_path = '/opt/consistency_models/samples/samples_100x256x256x3.npz'
data = np.load(npz_path)

images = data['arr_0']

output_dir = '/opt/consistency_models/samples/images_converted'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)
for i in range(len(images)):
    img = Image.fromarray(images[i])
    img_path = f'{output_dir}/sample_{i + 1}.png'
    img.save(img_path)
