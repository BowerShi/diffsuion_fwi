import torch
from denoising_diffusion_pytorch.denoising_diffusion_pytorch import num_to_groups, Trainer, Unet, GaussianDiffusion
from torchvision import transforms as T, utils
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
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
    is_working_with_fwi =True # True for conditional generation
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
all_images = all_images.squeeze(1).detach().cpu().numpy()

plt.figure(figsize=(5, 5))
for ii in range(1):
	plt.subplot(1,1,ii+1)
	plt.imshow(all_images[ii,:,:])
plt.savefig('/work/10225/bowenshi0610/vista/denoising-diffusion-pytorch/preview3.png',dpi=100)

# 	plt.colorbar(orientation='horizontal',shrink=0.6,label='Vel');
'''


for ii in range(1):
    print(all_images[ii,0,...])
all_images = reverse_linear_transform(all_images.detach().cpu().numpy())
all_images = all_images.squeeze(2)
all_images = all_images.squeeze(0)
min_value = all_images.min().min().min()
max_value = all_images.max().max().max()
pillow_images = [Image.fromarray(((image - min_value) / (max_value - min_value) * 100 + 150).astype(np.uint8)) for image in all_images]
pillow_images[0].save(
    "output.gif",  # Output file name
    save_all=True,  # Save all frames
    append_images=pillow_images[1:],  # Append the rest of the images
    duration=0.01,  # Duration between frames in milliseconds
    loop=0  # Infinite loop
)

'''