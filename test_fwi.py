import torch
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
import deepwave
from deepwave import scalar
import numpy as np
device = torch.device('cuda' if torch.cuda.is_available()
                      else 'cpu')
from torchaudio.functional import biquad
from scipy.ndimage import gaussian_filter
from scipy.signal import butter
import matplotlib.pyplot as plt
import deepwave
from deepwave import scalar
ny = 2301
nx = 751
dx = 4
v_true = torch.from_file('/work/10225/bowenshi0610/vista/fwi_dataset/GeoFWI/vp.bin',
                         size=ny*nx).reshape(ny, nx)

# Select portion of model for inversion
ny = 100
nx = 100
v_true = v_true[:ny, :nx]

np.random.seed(20232425)
inds=np.arange(49500)
np.random.shuffle(inds)

data=np.load('/work/10225/bowenshi0610/vista/fwi_dataset/GeoFWI/geofwi.npy')

v_true = torch.tensor(data[inds[19],:,:].T)

v_init = (torch.tensor(1/gaussian_filter(1/v_true.numpy(), 20))
          .to(device))
v = v_init.clone()
v.requires_grad_()

n_shots = 10

n_sources_per_shot = 1
d_source = 10  # 20 * 4m = 80m
first_source = 0.  # 10 * 4m = 40m
source_depth = 0.5  # 2 * 4m = 8m

n_receivers_per_shot = 100
d_receiver = 1  # 6 * 4m = 24m
first_receiver = 0  # 0 * 4m = 0m
receiver_depth = 0  # 2 * 4m = 8m

freq = 25
nt = 200
dt = 0.004
peak_time = 1.5 / freq

observed_data = (
    torch.from_file('/work/10225/bowenshi0610/vista/fwi_dataset/GeoFWI/marmousi_data.bin',
                    size=n_shots*n_receivers_per_shot*nt)
    .reshape(n_shots, n_receivers_per_shot, nt)
)

observed_data = (
    observed_data[:n_shots, :n_receivers_per_shot, :nt].to(device)
)

# source_locations
source_locations = torch.zeros(n_shots, n_sources_per_shot, 2,
                               dtype=torch.long, device=device)
source_locations[..., 1] = source_depth
source_locations[:, 0, 0] = (torch.arange(n_shots) * d_source +
                             first_source)

# receiver_locations
receiver_locations = torch.zeros(n_shots, n_receivers_per_shot, 2,
                                 dtype=torch.long, device=device)
receiver_locations[..., 1] = receiver_depth
receiver_locations[:, :, 0] = (
    (torch.arange(n_receivers_per_shot) * d_receiver +
     first_receiver)
    .repeat(n_shots, 1)
)

# source_amplitudes
source_amplitudes = (
    (deepwave.wavelets.ricker(freq, nt, dt, peak_time))
    .repeat(n_shots, n_sources_per_shot, 1).to(device)
)
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
    is_working_with_fwi = 3 # True for conditional generation
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

# Setup optimiser to perform inversion
optimiser = torch.optim.SGD([v], lr=1e9, momentum=0.9)
loss_fn = torch.nn.MSELoss()

# Run optimisation/inversion
n_epochs = 300
v_true = v_true

all_images = []
for cut_freq in [10, 15, 20, 25, 30]:
    for epoch in range(n_epochs):
        optimiser.zero_grad()
        if epoch !=0 and epoch % 10 == 0:
            torch.save(v, '/work/10225/bowenshi0610/vista/fwi_dataset/GeoFWI/inverted.pt')
            all_images_list = list(map(lambda n: trainer.ema.ema_model.sample(batch_size=n), batches))
            v_temp = 0.1 * v + 0.9 * reverse_linear_transform(torch.cat(all_images_list, dim = 0).squeeze(0).squeeze(0).T)
            print(reverse_linear_transform(torch.cat(all_images_list, dim = 0).squeeze(0).squeeze(0).T))
            
        out = scalar(
            v_temp if epoch !=0 and epoch % 50 == 0 else v, dx, dt,
            source_amplitudes=source_amplitudes,
            source_locations=source_locations,
            receiver_locations=receiver_locations,
            pml_freq=freq,
        )
        loss = loss_fn(out[-1], observed_data)
        
        loss.backward()
        
        torch.nn.utils.clip_grad_value_(
            v,
            torch.quantile(v.grad.detach().abs(), 0.98)
        )
        if epoch !=0 and epoch % 50 == 0:
            all_images.append(v_temp.detach().cpu().numpy())
        print(f'epoch: {epoch}, loss: ',loss.detach().cpu().numpy())
        optimiser.step()
'''
'''    

vmin = v_true.min()
vmax = v_true.max()
_, ax = plt.subplots(3, figsize=(30, 10), sharex=True,
                     sharey=True)
ax[0].imshow(v_init.cpu().T, 
             vmin=vmin, vmax=vmax)
ax[0].set_title("Initial")
ax[1].imshow(v.detach().cpu().T, 
             vmin=vmin, vmax=vmax)
ax[1].set_title("Out")
ax[2].imshow(v_true.cpu().T, 
             vmin=vmin, vmax=vmax)
ax[2].set_title("True")
plt.tight_layout()
plt.savefig('example_simple_fwi.jpg')
torch.save(v, '/work/10225/bowenshi0610/vista/fwi_dataset/GeoFWI/inverted.pt')
'''
'''
frames = []
for arr in all_images:
    # Create a Matplotlib figure
    fig, ax = plt.subplots()
    cax = ax.imshow(arr.T)  # Adjust colormap and scaling as needed
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
    duration=0.1,  # Duration per frame in milliseconds
    loop=0  # 0 = infinite loop
)
## Second attempt: constrained velocity and frequency filtering

def taper(x):
    return deepwave.common.cosine_taper_end(x, 100)


# Generate a velocity model constrained to be within a desired range
class Model(torch.nn.Module):
    def __init__(self, initial, min_vel, max_vel):
        super().__init__()
        self.min_vel = min_vel
        self.max_vel = max_vel
        self.model = torch.nn.Parameter(
            torch.logit((initial - min_vel) /
                        (max_vel - min_vel))
        )

    def forward(self):
        return (torch.sigmoid(self.model) *
                (self.max_vel - self.min_vel) +
                self.min_vel)


observed_data = taper(observed_data)
model = Model(v_init, 1000, 4500).to(device)


# Define a function to taper the ends of traces


# Run optimisation/inversion
n_epochs = 200
all_images = []
for cutoff_freq in [10, 15, 20, 25, 30]:
    sos = butter(6, cutoff_freq, fs=1/dt, output='sos')
    sos = [torch.tensor(sosi).to(observed_data.dtype).to(device)
           for sosi in sos]

    def filt(x):
        return biquad(biquad(biquad(x, *sos[0]), *sos[1]), *sos[2])
    observed_data_filt = filt(observed_data)
    optimiser = torch.optim.LBFGS(model.parameters(),
                                  line_search_fn='strong_wolfe')
    for epoch in range(n_epochs):
        def closure():
            optimiser.zero_grad()
            v = model()
            out = scalar(
                v, dx, dt,
                source_amplitudes=source_amplitudes,
                source_locations=source_locations,
                receiver_locations=receiver_locations,
                max_vel=2500,
                pml_freq=freq,
                time_pad_frac=0.2,
            )
            out_filt = filt(taper(out[-1]))
            loss = 1e6*loss_fn(out_filt, observed_data_filt)
            loss.backward()
            return loss

        optimiser.step(closure)
        
        
v = model()
vmin = v_true.min()
vmax = v_true.max()
_, ax = plt.subplots(3, figsize=(30, 10), sharex=True,
                     sharey=True)
ax[0].imshow(v_init.cpu().T, 
             vmin=vmin, vmax=vmax)
ax[0].set_title("Initial")
ax[1].imshow(v.detach().cpu().T, 
             vmin=vmin, vmax=vmax)
ax[1].set_title("Out")
ax[2].imshow(v_true.cpu().T, 
             vmin=vmin, vmax=vmax)
ax[2].set_title("True")
plt.tight_layout()
plt.savefig('example_increasing_freq_fwi.jpg')
torch.save(v, '/work/10225/bowenshi0610/vista/fwi_dataset/GeoFWI/inverted_multi.pt')


frames = []
for arr in all_images:
    # Create a Matplotlib figure
    fig, ax = plt.subplots()
    cax = ax.imshow(arr.T)  # Adjust colormap and scaling as needed
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
    duration=0.1,  # Duration per frame in milliseconds
    loop=0  # 0 = infinite loop
)
## Second attempt: constrained velocity and frequency filtering
