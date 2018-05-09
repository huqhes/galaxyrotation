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
dPos = dDec * 2*np.pi * distance / 1.296 # Mpc/10^6 arcseconds = pc
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
block = 1
cube_ = cube
#cube_ = block_reduce(cube, block_size=(1,block,block), func=np.mean)
cube_[cube_ < 0.001] = 0

plt.figure(3)
plt.subplot(121)
plt.axis('off')
plt.imshow(np.sum(cube_, axis=0))
if (block == 1):
    plt.title('Uncompressed Image')
else:
    plt.title('Compressed Image (Block size %i)' %block)

# IMAGE PROCESSING
mid = [np.floor(cube_.shape[1]/2), np.floor(cube_.shape[2]/2)]
velocities = {}
mass = {}
distanceM = distance*3.086*10**22 #distance in meters
beam = np.abs((1024/block)*dRA*(1024/block)*dDec) / (4.25*10**10)
for i in range(cube_.shape[0]):
    v = np.abs(np.floor(cube_.shape[0]/2 - i) * dV)
    for x in range(cube_.shape[1]):
        for y in range(cube_.shape[2]):
            # Rotation curve data
            if (cube_[i,x,y] > 0):
                r = np.floor(np.sqrt((mid[0]-x)**2 + (mid[1]-y)**2)) * dPos * block
                if (velocities.get(r,None) is None):
                    velocities[r] = [v]
                else:
                    velocities[r].append(v)

                # m ~= (4*pi*R^2 * flux)^(1/n)
                # R - distance from observer
                # flux - W/m^2, not Jy/beam
                m = (4*np.pi* (distanceM)**2 * (cube_[i,x,y]* 10**26 * beam) )**(1/3.5)    
                if (mass.get(r,None) is None):
                    mass[r] = m
                else:
                    mass[r] = mass[r] + m

# ROTATION CURVE - AVERAGE VELOCITY
r = list(velocities.keys())
v = list(velocities.values())
m = list(mass.values())

for i in range(np.shape(v)[0]):
    v[i] = np.mean(v[i])
    if (i>0):
        m[i] = m[i] + m[i-1]

# Ignore division by 0 for r=0
np.seterr(divide='ignore')

# Radius in meters
rM = np.asarray(r) * 3.086*10**16

G = 6.67408 * 10**-11
# v = sqrt(GM/r)
v_stellar = np.sqrt(G*np.divide(np.asarray(m),rM))

ax = plt.subplot(122)
# ax = plt.subplot(111)
ax.scatter(r,v)
ax.scatter(r,v_stellar)
plt.xlabel('Radius (parsec)')
plt.ylabel('Velocity (m/s)')
plt.legend()
plt.title('Rotation Curve')
plt.show()