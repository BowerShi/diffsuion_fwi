import numpy as np
import torch
import matplotlib.pyplot as plt
from torchvision.transforms import Resize, InterpolationMode
from torchvision.transforms.functional import to_tensor, to_pil_image
import math
from torchvision import transforms as T, utils
def calculate_max_and_min(data): 
    flat_data = data.flatten()
    data_min = flat_data.min()
    data_max = flat_data.max()
    print(f"Minimum Value: {data_min}, Maximum Value: {data_max}")
    return data_min, data_max

def plot_velocity_distribution(data):
    flat_data = data.flatten()
    data_min, data_max = 1500.0093994140625, 4500.0
    plt.hist(flat_data, bins=range(int(data_min - 100.), int(data_max + 100.), 100), edgecolor='black')
    plt.title("Pixel Intensity Distribution")
    plt.xlabel("Intensity")
    plt.ylabel("Frequency")
    plt.savefig("stats.png")

def linear_transform(data):
    normalized_data = 2 * (data - 1500.0093994140625) / (4500.0 - 1500.0093994140625) - 1
    return normalized_data
    
def reverse_linear_transform(normalized_data):
    data = (normalized_data + 1.0) / 2.0 * (4500.0 - 1500.0093994140625) + 1500.0093994140625
    return data


# 转换为 PyTorch Tensor
np.random.seed(20232425)
data=np.load('/work/10225/bowenshi0610/vista/fwi_dataset/GeoFWI/geofwi.npy')
print(data.shape[0])
data = np.random.permutation(data)
normalized_data = linear_transform(data[:45000,...])
tempdata = normalized_data
np.random.seed(20232425)
inds=np.arange(45000)
np.random.shuffle(inds)
for ii in range(25): 
    print(tempdata[ii,99,99])
tempdata = torch.tensor(tempdata).unsqueeze(1)
utils.save_image(tempdata[0:25,...], '/work/10225/bowenshi0610/vista/fwi_dataset/GeoFWI/samples-random2.png', nrow = 5)

'''
plt.figure(figsize=(20, 20))
for ii in range(25):
	plt.subplot(5,5,ii+1)
	plt.imshow(tempdata[inds[ii],:,:])

# plt.colorbar(orientation='horizontal',shrink=0.6,label='Vel')
plt.savefig('/work/10225/bowenshi0610/vista/fwi_dataset/GeoFWI/samples-random.png',dpi=300)
'''
tensor_data = torch.tensor(normalized_data).unsqueeze(1)  # shape: batch * 1 * 100 * 100
test_data = linear_transform(data[45000:,...])
test_data = torch.tensor(test_data).unsqueeze(1)
'''
resizer = Resize((128, 128), interpolation = InterpolationMode.NEAREST_EXACT)
resized_tensor = resizer(tensor_data)  # shape: batch * 1 * 128 * 128
torch_tensor = torch.tensor(resized_tensor, dtype=torch.float32)
'''
torch.save(tensor_data, "normalized_fwi.pt")
torch.save(test_data, "normalized_fwi_test.pt")

loaded_tensor = torch.load("normalized_fwi.pt")


plt.figure(figsize=(20, 20))
for ii in range(25):
	plt.subplot(5,5,ii+1)
	plt.imshow(reverse_linear_transform(loaded_tensor[ii,...].squeeze(0)))
    
for ii in range(25):
    print(loaded_tensor[ii,...].min(), loaded_tensor[ii,...].max())

# plt.colorbar(orientation='horizontal',shrink=0.6,label='Vel')
plt.savefig('transformed-samples-random.png',dpi=300)
# 输出结果

