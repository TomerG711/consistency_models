import torch as th

from cm import dist_util, logger
from cm.image_datasets import load_data
from cm.karras_diffusion import iterative_superres
from cm.random_util import get_generator
from cm.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
)

import torch
from PIL import Image
import numpy as np

from image_sample import create_argparser


def save_image(tensor, path):
    """
    Saves a given torch tensor as a PNG image to the specified path.

    Parameters:
    tensor (torch.Tensor): The image tensor to save, with values in the range [-1, 1].
    path (str): The path where the image will be saved.
    """

    # Scale the tensor to the range [0, 1]
    tensor = (tensor + 1) / 2

    # Convert the tensor to a numpy array
    numpy_array = tensor.cpu().numpy()

    # Scale to [0, 255] and convert to uint8
    numpy_array = (numpy_array * 255).astype(np.uint8)

    # Check the shape and convert to appropriate format
    if numpy_array.ndim == 3 and numpy_array.shape[0] == 3:
        # If the tensor is in the format [C, H, W], transpose to [H, W, C]
        numpy_array = numpy_array.transpose(1, 2, 0)

    # Create a PIL image
    image = Image.fromarray(numpy_array)

    # Save the image
    image.save(path)


def superres_sample_image(
        diffusion,
        model,
        steps,
        image,
        clip_denoised=True,
        model_kwargs=None,
        generator=None,
        ts=None,
):
    if generator is None:
        generator = get_generator("dummy")
    else:
        generator = get_generator(generator)

    def denoiser(x_t, sigma):
        _, denoised = diffusion.denoise(model, x_t, sigma, **model_kwargs)
        if clip_denoised:
            denoised = denoised.clamp(-1, 1)
        return denoised

    x_out, images = iterative_superres(
        distiller=denoiser,
        images=image,
        x=generator.randn(*image.shape, device=dist_util.dev()),
        ts=ts,
        t_min=0.002,
        t_max=80.0,
        rho=diffusion.rho,
        steps=steps,
        generator=generator
    )

    x_out = ((x_out + 1) * 127.5).clamp(0, 255).to(th.uint8)
    x_out = x_out.permute(0, 2, 3, 1)
    x_out = x_out.contiguous()

    images = ((images + 1) * 127.5).clamp(0, 255).to(th.uint8)
    images = images.permute(0, 2, 3, 1)
    images = images.contiguous()

    return x_out, images


def main():
    args = create_argparser().parse_args()
    dist_util.setup_dist()
    logger.configure()

    if "consistency" in args.training_mode:
        distillation = True
    else:
        distillation = False

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys()),
        distillation=distillation,
    )
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(dist_util.dev())
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()

    # all_files = _list_image_files_recursively(args.data_dir)

    print(args)
    ts = tuple(int(x) for x in args.ts.split(","))
    data = load_data(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        class_cond=args.class_cond,
        deterministic=True,
        random_crop=False,
        random_flip=False,
    )
    model_kwargs = {}

    logger.log("sampling...")
    i = 0
    for batch, _ in data:
        logger.log(f"Sampling image {i}")
        logger.log(f"{batch.shape}")
        x_out, image = superres_sample_image(
            diffusion,
            model,
            args.steps,
            batch,
            generator=args.generator,
            ts=ts,
            model_kwargs=model_kwargs,
        )
        save_image(x_out, args.outdir + f"out_{i}.png")
        save_image(image, args.outdir + f"in_{i}.png")
        i += 1


if __name__ == "__main__":
    main()
