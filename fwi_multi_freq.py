import torch
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
import numpy as np
from torchaudio.functional import biquad
from scipy.ndimage import gaussian_filter
from scipy.signal import butter
import deepwave
from deepwave import scalar

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

ny = 100
nx = 100
dx = 4
data=np.load('/work/10225/bowenshi0610/vista/fwi_dataset/GeoFWI/geofwi.npy')

v_true = torch.tensor(data[39363,:,:].T)

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
from PIL import Image
import io
def linear_transform(data):
    normalized_data = 2 * (data - 1500.0093994140625) / (4500.0 - 1500.0093994140625) - 1
    return normalized_data

def reverse_linear_transform(normalized_data):
    data = (normalized_data + 1.0) / 2.0 * (4500.0 - 1500.0093994140625) + 1500.0093994140625
    return data

def taper(x):
    return deepwave.common.cosine_taper_end(x, 100)


# Generate a velocity model constrained to be within a desired range


observed_data = taper(observed_data)
# model = Model(v_init, 1000, 2500).to(device)
v = v_init.clone().to(device).requires_grad_(True)
loss_fn = torch.nn.MSELoss()
# Run optimisation/inversion
n_epochs = 30
all_images = []
plt.figure(figsize=(35,5))
index = 0
for cutoff_freq in [10, 15, 20, 25, 30]:
    index = index + 1
    sos = butter(6, cutoff_freq, fs=1/dt, output='sos')
    sos = [torch.tensor(sosi).to(observed_data.dtype).to(device)
           for sosi in sos]

    def filt(x):
        return biquad(biquad(biquad(x, *sos[0]), *sos[1]), *sos[2])
    observed_data_filt = filt(observed_data)
    
    optimiser = torch.optim.LBFGS([v],
                                  line_search_fn='strong_wolfe')
    print(v.T)
    for epoch in range(n_epochs):
        print("epoch: ", epoch)
        def closure():
            optimiser.zero_grad()
            out = scalar(
                v, dx, dt,
                source_amplitudes=source_amplitudes,
                source_locations=source_locations,
                receiver_locations=receiver_locations,
                max_vel=4600,
                pml_freq=freq,
                time_pad_frac=0.2,
            )
            out_filt = filt(taper(out[-1]))
            loss = 1e5*loss_fn(out_filt, observed_data_filt)
            loss.backward()
            print(loss)
            return loss

        optimiser.step(closure)
    plt.subplot(1, 7, index)
    plt.imshow(v.clone().detach().cpu().T)
    
plt.subplot(1, 7, 6)
plt.imshow(v_init.cpu().T)
plt.subplot(1, 7, 7)
plt.imshow(v_true.cpu().T)
plt.savefig('example_increasing_freq_fwi.jpg')
torch.save(v, '/work/10225/bowenshi0610/vista/fwi_dataset/GeoFWI/inverted_multi.pt')

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
'''