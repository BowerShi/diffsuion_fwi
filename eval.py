import torch
from denoising_diffusion_pytorch.denoising_diffusion_pytorch import num_to_groups, Trainer, Unet, GaussianDiffusion
from torchvision import transforms as T, utils
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import io
import hydra
from omegaconf import DictConfig
import os

def reverse_linear_transform(normalized_data):
    data = (normalized_data + 1.0) / 2.0 * (4500.0 - 1500.0093994140625) + 1500.0093994140625
    return data

def plot_gif(all_images, index, path_folder):
    frames = []
    for arr in all_images:
        # Create a Matplotlib figure
        fig, ax = plt.subplots()
        cax = ax.imshow(arr)  # Adjust colormap and scaling as needed
        ax.axis('off')  # Hide axes
        # plt.colorbar(cax)  # Optional: add colorbar for visualization
        
        # Save the figure to a buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
        buf.seek(0)
        plt.close(fig)
        
        # Convert buffer to PIL Image and append to frames
        frame = Image.open(buf)
        frames.append(frame)
    name = os.path.join(path_folder, f"results/animated_{index}.gif")
    frames[0].save(
        name, 
        save_all=True, 
        append_images=frames[1:], 
        duration=0.001,  # Duration per frame in milliseconds
        loop=0  # 0 = infinite loop
    )
'''
 First generate dataset normalized_fwi.pt using geofwi_stats.py
 train.py is to train the diffusion model
 eval.py is to evaluate the diffusion model
'''
def plot_png(all_images, num_of_pics, path_folder, label = 'last'):
    plt.figure(figsize=(20, 20))
    plt.title(f"{label} {num_of_pics} pictures of predicted x_start")
    if label == 'last':
        for ii in range(num_of_pics):
            plt.subplot(int(np.sqrt(num_of_pics)), int(np.sqrt(num_of_pics)), ii+1)
            plt.imshow(all_images[all_images.shape[0] - num_of_pics + ii, :, :])
    elif label == 'first':
        for ii in range(num_of_pics):
            plt.subplot(int(np.sqrt(num_of_pics)), int(np.sqrt(num_of_pics)), ii+1)
            plt.imshow(all_images[ii, :, :])
    plt.savefig(f'{path_folder}/results/{label}_{num_of_pics}.png',dpi=100)


@hydra.main(config_path="config", config_name="config")
def main(cfg: DictConfig):
    model = Unet(
        dim = 64,
        channels=1,
        dim_mults = (1, 2, 8),
        flash_attn = False
    )

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable_params}")

    diffusion = GaussianDiffusion(
        model,
        image_size = 100,
        timesteps = 1000,           # number of steps
        sampling_timesteps = 100,   # number of sampling timesteps (using ddim for faster inference [see citation for ddim paper])
        auto_normalize = False,
        is_working_with_fwi = 5,
        **cfg
        # True for conditional generation
    )
    # result_folder should be the address where you store model-300.pt
    # folder should be the address where you store normalized_fwi.pt (normalized to [-1, 1]) via linear transform
    trainer = Trainer(
        diffusion_model=diffusion, 
        results_folder = cfg.result_folder_path, 
        folder = cfg.training_data_path, 
        train_batch_size = 16
    )

    trainer.load(300) # load model-300.pt
    trainer.batch_size 
    # fidscore = trainer.fid_scorer.fid_score()
    # print(f"FID score is {fidscore}")
    # with torch.inference_mode():

    # default setting, because we cannot handle batches
    batches = num_to_groups(1, 1)
    all_images_list = list(map(lambda n: trainer.ema.ema_model.sample(batch_size=n), batches))

    all_images = torch.cat(all_images_list, dim = 0)
    print(all_images.shape)
    all_images = all_images.squeeze(0).squeeze(1).detach().cpu().numpy()
    all_images = reverse_linear_transform(all_images)

    # plot the images
    num_of_pics = 9
    path_folder = '/work/10225/bowenshi0610/vista/denoising-diffusion-pytorch'
    plot_png(all_images, num_of_pics, path_folder, label = 'last')
    plot_png(all_images, num_of_pics, path_folder, label = 'first')
    plot_gif(all_images, cfg.index, path_folder)        
    
if __name__ == "__main__":
    main()