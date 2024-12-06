import numpy as np
import matplotlib.pyplot as plt

sizes=np.load('/work/10225/bowenshi0610/vista/fwi_dataset/GeoFWI/geofwi-size-layer-fault-salt-1-10.npy')
print(sizes)
geofwi=np.load('/work/10225/bowenshi0610/vista/fwi_dataset/GeoFWI/geofwi.npy')
nsample=geofwi.shape[0]

np.random.seed(20232425)
inds=np.arange(49500)
np.random.shuffle(inds)

plt.figure(figsize=(20, 20))
for ii in range(25):
	plt.subplot(5,5,ii+1)
	plt.imshow(geofwi[inds[ii],:,:])

print(geofwi[inds[ii],:,:])
# plt.colorbar(orientation='horizontal',shrink=0.6,label='Vel')
plt.savefig('samples-random.png',dpi=300)


plt.figure(figsize=(20, 20))
for ii in range(25):
	plt.subplot(5,5,ii+1)
	plt.imshow(geofwi[ii,:,:])
# 	plt.colorbar(orientation='horizontal',shrink=0.6,label='Vel');
plt.savefig('samples-increasing.png',dpi=300)




plt.figure(figsize=(20, 20))
for ii in range(25):
	plt.subplot(5,5,ii+1)
	plt.imshow(geofwi[nsample-ii-1,:,:])
# 	plt.colorbar(orientation='horizontal',shrink=0.6,label='Vel');
plt.savefig('samples-decreasing.png',dpi=300)



plt.figure(figsize=(20, 20))
for ii in range(25):
	plt.subplot(5,5,ii+1)
	plt.imshow(geofwi[nsample-sizes[-1]-1-ii,:,:])
# 	plt.colorbar(orientation='horizontal',shrink=0.6,label='Vel');
plt.savefig('samples-faults.png',dpi=300)
print("I am here")
plt.figure(figsize = (40, 40))

for ii in range(30):
	plt.subplot(8, 8, 2*ii+1)
	plt.imshow(geofwi[sizes[:ii+1].sum()-1,:,:])
	print(geofwi[sizes[:ii+1].sum()-1,:,:].min(), geofwi[sizes[:ii+1].sum()-1,:,:].max())
	plt.subplot(8, 8, 2*ii+2)
	plt.imshow(geofwi[sizes[:ii+1].sum()-2,:,:])
	print(geofwi[sizes[:ii+1].sum()-2,:,:].min(), geofwi[sizes[:ii+1].sum()-2,:,:].max())
  
plt.savefig('/work/10225/bowenshi0610/vista/fwi_dataset/GeoFWI/samples-different.png',dpi=300)