import torch
from denoising_diffusion_pytorch.denoising_diffusion_pytorch import num_to_groups, Trainer, Unet, GaussianDiffusion
from torchvision import transforms as T, utils
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import io
def reverse_linear_transform(normalized_data):
    data = (normalized_data + 1.0) / 2.0 * (4500.0 - 1500.0093994140625) + 1500.0093994140625
    return data
'''
 First generate dataset normalized_fwi.pt using geofwi_stats.py
 train.py is to train the diffusion model
 eval.py is to evaluate the diffusion model
'''


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
    sampling_timesteps = 250,   # number of sampling timesteps (using ddim for faster inference [see citation for ddim paper])
    auto_normalize = False,
    is_working_with_fwi = 4
    # True for conditional generation
)
# result_folder should be the address where you store model-300.pt
# folder should be the address where you store normalized_fwi.pt (normalized to [-1, 1]) via linear transform
trainer = Trainer(diffusion_model=diffusion, results_folder = '/work/10225/bowenshi0610/vista/results2', folder='/work/10225/bowenshi0610/vista/fwi_dataset/GeoFWI/normalized_fwi.pt', train_batch_size=16)

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

num_of_pics = 9
plt.figure(figsize=(20, 20))
plt.title(f"Last {num_of_pics} pictures of predicted x_start")
for ii in range(num_of_pics):
	plt.subplot(int(np.sqrt(num_of_pics)), int(np.sqrt(num_of_pics)), ii+1)
	plt.imshow(all_images[all_images.shape[0]-1-ii, :, :])

plt.savefig('/work/10225/bowenshi0610/vista/denoising-diffusion-pytorch/preview3.png',dpi=100)

# 	plt.colorbar(orientation='horizontal',shrink=0.6,label='Vel');
    
all_images = reverse_linear_transform(all_images)


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
    
frames[0].save(
    'animated.gif', 
    save_all=True, 
    append_images=frames[1:], 
    duration=0.001,  # Duration per frame in milliseconds
    loop=0  # 0 = infinite loop
)
