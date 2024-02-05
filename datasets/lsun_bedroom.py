"""
Convert an LSUN lmdb database into a directory of images.
"""

import argparse
import io
import os

from PIL import Image
# import lmdb
import numpy as np


def read_images(lmdb_path, image_size):
    env = lmdb.open(lmdb_path, map_size=1099511627776, max_readers=100, readonly=True)
    with env.begin(write=False) as transaction:
        cursor = transaction.cursor()
        for _, webp_data in cursor:
            img = Image.open(io.BytesIO(webp_data))
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
            yield arr


def read_images_tomer(images_dir, image_size):
    i = 0
    print("Starting to read images")
    for root, dirs, files in os.walk(images_dir):
        for file in files:
            img_path = os.path.join(root, file)
            print(f"Image {i}: {img_path}")
            img = Image.open(img_path)
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
            yield arr
            i = i + 1


def dump_images(out_dir, images, prefix):
    print("Starting to dump images")
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    for i, img in enumerate(images):
        img_name = os.path.join(out_dir, f"{prefix}_{i:07d}.png")
        print(f"Dumping image: {img_name}")
        Image.fromarray(img).save(img_name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-size", help="new image size", type=int, default=256)
    parser.add_argument("--prefix", help="class name", type=str, default="bedroom")
    parser.add_argument("--tomer_reader", help="Wether to use tomer reader or regular lmdb reader ", type=bool,
                        default=True)
    parser.add_argument("--tomer_reader_images_path", help="Path to images dir when using tomer reader", type=str,
                        default='./lsun_bedroom/data0')
    # parser.add_argument("lmdb_path", help="path to an LSUN lmdb database")
    parser.add_argument("out_dir", help="path to output directory")
    args = parser.parse_args()
    if args.tomer_reader:
        images = read_images_tomer(args.tomer_reader_images_path, args.image_size)
    else:
        images = read_images(args.lmdb_path, args.image_size)
    dump_images(args.out_dir, images, args.prefix)


if __name__ == "__main__":
    main()
