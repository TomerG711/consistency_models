import torch as th
import cv2
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
# from piq import LPIPS
import lpips

from image_sample import create_argparser
from inverse.measurements import get_operator
import matplotlib.pyplot as plt  # TOM:
import os

def save_image(tensor, path):
    """
    Saves a given torch tensor as a PNG image to the specified path.

    Parameters:
    tensor (torch.Tensor): The image tensor to save, with values in the range [-1, 1].
    path (str): The path where the image will be saved.
    """

    # Scale the tensor to the range [0, 1]
    # tensor = (tensor + 1) / 2

    # Convert the tensor to a numpy array
    numpy_array = tensor.cpu().numpy()

    # Scale to [0, 255] and convert to uint8
    numpy_array = (numpy_array * 255).astype(np.uint8)

    # Check the shape and convert to appropriate format
    if numpy_array.ndim == 3 and numpy_array.shape[0] == 3:
        # If the tensor is in the format [C, H, W], transpose to [H, W, C]
        numpy_array = numpy_array.transpose(1, 2, 0)

    # Create a PIL image
    # img_ = Image.fromarray(cv2.cvtColor(numpy_array, cv2.COLOR_BGR2RGB))
    image = Image.fromarray(numpy_array)

    # Save the image
    image.save(path)
    # img_.save(path)


def superres_sample_image(
        diffusion,
        model,
        steps,
        image,
        clip_denoised=True,
        model_kwargs=None,
        generator=None,
        ts=None,
        orig_for_psnr=None
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

    x_out, images, psnr_per_iter, lpips_per_iter = iterative_superres(
        distiller=denoiser,
        images=image,
        x=generator.randn(image.shape, device=dist_util.dev()),
        ts=ts,
        t_min=0.002,
        t_max=80.0,
        rho=diffusion.rho,
        steps=steps,
        generator=generator,
        orig_for_psnr=orig_for_psnr
    )
    x_out_original = x_out.detach().clone()
    # x_out_original = x_out_original.contiguous()
    # x_out_original = x_out_original.permute(0, 2, 3, 1)
    x_out = ((x_out + 1) * 127.5).clamp(0, 255).to(th.uint8)
    x_out = x_out.permute(0, 2, 3, 1)
    x_out = x_out.contiguous()

    images = ((images + 1) * 127.5).clamp(0, 255).to(th.uint8)
    # images = images.permute(0, 2, 3, 1)
    images = images.contiguous()

    return x_out, images, x_out_original, psnr_per_iter, lpips_per_iter


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

    # print(args)
    ts = tuple(int(x) for x in args.ts.split(","))
    data = load_data(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        class_cond=args.class_cond,
        deterministic=True,
        random_crop=False,
        random_flip=False,
        only_once=True,
    )
    model_kwargs = {}

    logger.log("sampling...")
    i = 0
    sr_operator = get_operator(
        name="super_resolution",
        device=dist_util.dev(),
        in_shape=(1, 3, 256, 256),
        scale_factor=4,
    )
    # lpips_fn = LPIPS(replace_pooling=True, reduction="none")
    lpips_fn = lpips.LPIPS(net='vgg').to(dist_util.dev())
    psnrs = []
    lpipses = []
    psnr_per_img_per_iter = np.zeros((300, 38))  # TOMER
    # lpips_per_img_per_iter = np.zeros((val_loader.sampler.num_samples,config.time_travel.T_sampling))
    lpips_per_img_per_iter = np.zeros((300, 38))  # TOMER
    img_ind = -1
    for gt, _ in data:
        img_ind += 1
        # logger.log(f"Sampling image {i}")
        # logger.log(f"{batch.shape}")
        gt = gt.to(dist_util.dev())
        y = sr_operator.forward(gt)
        y += th.torch.randn_like(y, device=y.device) * 0.05
        # logger.log(f"{y.shape}")
        # print(f"{((y+1)/2).min()}, {((y+1)/2).max()}")
        save_image(((y + 1) / 2)[0], args.out_dir + f"/{i}_y.png")  # Save y before upsampling
        y = sr_operator.transpose(y)
        # logger.log(f"{y.shape}")
        x_out, image, x_out_original, psnr_per_iter, lpips_per_iter = superres_sample_image(
            diffusion=diffusion,
            model=model,
            steps=args.steps,
            image=y,
            generator=args.generator,
            ts=ts,
            model_kwargs=model_kwargs,
            orig_for_psnr=((gt + 1) / 2).clamp(0, 1)
        )
        psnr_per_img_per_iter[img_ind, :] = psnr_per_iter
        lpips_per_img_per_iter[img_ind, :] = lpips_per_iter
        # batch = batch.permute(0, 2, 3, 1)
        # batch = batch.reshape(1,256,256,3)#.cpu().numpy().astype(np.float32)
        # print(x_out_original)
        # print(batch)
        # print(x_out_original.shape)
        # print(batch.shape)
        x_out_original = (x_out_original + 1) / 2
        x_out_original = x_out_original.clamp(0, 1)
        gt = (gt + 1) / 2
        # x_out_original = x_out_original.permute(0, 3, 1, 2)
        # batch = batch.permute(0, 3, 2, 1)
        # x_out_original = x_out_original.clamp(0, 1)
        # image = image.permute(0, 3, 1, 2)
        # image = image[:, [2, 1, 0], :, :]

        # print(x_out_original.shape)  # BGR
        # print(image.shape)  # BGR
        # print(batch.shape)  # RGB
        # mse = torch.mean(((x_out_original + 1) / 2 - (batch + 1) / 2) ** 2)
        # print(f"X_OUT - MIN: {x_out_original.min()}, MAX: {x_out_original.max()}")
        # print(f"GT - MIN: {gt.min()}, MAX: {gt.max()}")
        mse = torch.mean((x_out_original - gt) ** 2)
        psnr = 10 * torch.log10(1 / mse)
        # print(psnr)
        # print(x_out_original.shape)
        # print(batch.shape)
        curr_lpips = lpips_fn(x_out_original, gt)
        psnrs.append(psnr.cpu().numpy())
        lpipses.append(curr_lpips.item())
        logger.log(f"Image {i} - PSNR: {psnr:.2f}, AVG PSNR: {np.mean(psnrs):.2f}, "
                   f"LPIPS: {curr_lpips.item():.4f}, AVG LPIPS: {np.mean(lpipses):.4f}")
        # print(f"{x_out_original.min()}, {x_out_original.max()}")
        # print(f"{gt.min()}, {gt.max()}")
        save_image(x_out_original[0], args.out_dir + f"/{i}_out.png")  # BAD
        save_image(gt[0], args.out_dir + f"/{i}_gt.png")
        i += 1

    plt.figure(1)
    psnr_per_iter_mean = np.mean(psnr_per_img_per_iter, axis=0)
    psnr_per_iter_std = np.std(psnr_per_img_per_iter, axis=0)
    plt.plot(range(1, 1 + psnr_per_iter_mean.shape[0]), psnr_per_iter_mean, '-', color='black')
    plt.fill_between(range(1, 1 + psnr_per_iter_mean.shape[0]), psnr_per_iter_mean - psnr_per_iter_std,
                     psnr_per_iter_mean + psnr_per_iter_std, color='gray', alpha=0.2)
    plt.grid()
    plt.xlabel('Iteration', fontsize=13)
    plt.ylabel('PSNR [dB]', fontsize=13)
    plt.ylim((0, 40))
    plt.savefig(os.path.join(args.out_dir, f"PSNR_vs_Iter.png"))

    plt.figure(2)
    lpips_per_iter_mean = np.mean(lpips_per_img_per_iter, axis=0)
    lpips_per_iter_std = np.std(lpips_per_img_per_iter, axis=0)
    plt.plot(range(1, 1 + lpips_per_iter_mean.shape[0]), lpips_per_iter_mean, '-', color='black')
    plt.fill_between(range(1, 1 + lpips_per_iter_mean.shape[0]), lpips_per_iter_mean - lpips_per_iter_std,
                     lpips_per_iter_mean + lpips_per_iter_std, color='gray', alpha=0.2)
    plt.grid()
    plt.xlabel('Iteration', fontsize=13)
    plt.ylabel('LPIPS', fontsize=13)
    plt.ylim((0, 2))
    plt.savefig(os.path.join(args.out_dir, f"LPIPS_vs_Iter.png"))

    logger.log(f"Final results - PSNR: {np.mean(psnrs)}, LPIPS: {np.mean(lpipses)}")


if __name__ == "__main__":
    main()
