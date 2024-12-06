import torch
from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer

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
    sampling_timesteps = 250    # number of sampling timesteps (using ddim for faster inference [see citation for ddim paper])
)

trainer = Trainer(
    diffusion,
    '/work/10225/bowenshi0610/vista/fwi_dataset/GeoFWI/normalized_fwi.pt',
    train_batch_size = 32,
    train_lr = 8e-5,
    train_num_steps = 300000,         # total training steps
    gradient_accumulate_every = 2,    # gradient accumulation steps
    ema_decay = 0.995,                # exponential moving average decay
    amp = True,                       # turn on mixed precision
    num_fid_samples= 40000,
    calculate_fid = False              # whether to calculate fid during training
)

trainer.train()
