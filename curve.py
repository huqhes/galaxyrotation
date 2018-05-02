import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from astropy.io import fits
from skimage.measure import block_reduce

# IMPORT DATA
data = fits.open('DDO154_RO_CUBE_THINGS.FITS')
header = data[0].header
cube = (data[0].data)[0]

# SET AXES' REFERENCE VALUES AND DELTAS
ra0 = header['CRVAL1'] # arcseconds
dec0 = header['CRVAL2'] # arcseconds
vel0 = header['CRVAL3'] # m/s
dRA = header['CDELT1'] # arcseconds
dDec = header['CDELT2'] # arcseconds
dV = header['CDELT3'] # m/s

# ESTIMATE DISTANCE TO GALAXY (vel0/h0) AND ANGULAR DISTANCE
h0 = 71000 # HUBBLE CONSTANT (m/s/Mpc)
distance = vel0/h0 # Mpc
dPos = dDec * 2*np.pi * distance / 1.296 # pc
print('Distance = ', distance, ' Mpc')
print('Marginal angular distance = ' , dPos, ' pc')

# PLOT CUBE SUM WITHOUT NOISE REDUCTION
plt.figure(1)
plt.rcParams['image.cmap'] = 'cubehelix'
plt.subplot(121)
plt.imshow(np.sum(cube,axis=0))
plt.axis('off')
plt.title('No Noise Reduction')

# NOISE REDUCTION
cube[cube < 0.001] = 0

# PLOT CUBE WITH NOISE REDUCTION
plt.subplot(122)
plt.imshow(np.sum(cube,axis=0))
plt.axis('off')
plt.title('Noise Reduction')

# PLOT ANIMATED CUBE WITH CHANGING WAVELENGTH
fig = plt.figure(2)
ims = []
for i in range(cube.shape[0]):
    im = plt.imshow(cube[i], animated=True)
    ims.append([im])

ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True, repeat_delay=100)
plt.title('Spectral Imaging of Galaxy')
plt.axis('off')

# COMPRESS CUBE DATA
block = 5
cube_ = block_reduce(cube, block_size=(1,block,block), func=np.mean)
cube_[cube_ < 0.001] = 0

plt.figure(3)
plt.subplot(121)
plt.axis('off')
plt.imshow(np.sum(cube_, axis=0))
if (block == 1):
    plt.title('Uncompressed Image')
else:
    plt.title('Compressed Image')

# PLOT ROTATION CURVE
mid = [np.floor(cube_.shape[1]/2), np.floor(cube_.shape[2]/2)]
velocities = {}
for i in range(cube_.shape[0]):
    v = np.abs(np.floor(cube_.shape[0]/2 - i) * dV)
    for x in range(cube_.shape[1]):
        for y in range(cube_.shape[2]):
            if (cube_[i,x,y] > 0):
                r = np.floor(np.sqrt((mid[0]-x)**2 + (mid[1]-y)**2)) * dPos * block
                if (velocities.get(r,None) is None):
                    velocities[r] = [v]
                else:
                    velocities[r].append(v)

plt.subplot(122)
r = list(velocities.keys())
v = list(velocities.values())

for i in range(np.shape(v)[0]):
    v[i] = np.mean(v[i])

plt.scatter(r,v)
plt.xlabel('Radius (parsec)')
plt.ylabel('Velocity (m/s)')
plt.title('Rotation Curve')
plt.show()