import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from astropy.io import fits
from skimage.measure import block_reduce

# IMPORT DATA

# Change for each galaxy:
data = fits.open('DDO154_RO_CUBE_THINGS.FITS')
bmaj = 7.94
bmin = 6.27

header = data[0].header
cube = (data[0].data)[0]

# SET AXES' REFERENCE VALUES AND DELTAS
name = header['OBJECT'] # Galaxy name
ra0 = header['CRVAL1'] # arcminutes
dec0 = header['CRVAL2'] # arcminutes
vel0 = header['CRVAL3'] # m/s
dRA = header['CDELT1'] # arcminutes
dDec = header['CDELT2'] # arcminutes
dV = header['CDELT3'] # m/s

# ESTIMATE DISTANCE TO GALAXY (vel0/h0) AND ANGULAR DISTANCE
h0 = 71000 # HUBBLE CONSTANT (m/s/Mpc)
distance = vel0/h0 # Mpc
dPos = dDec * 1000 * distance / 206265 * 10**6 # Mpc = 10^6 pc
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
# block = 1
cube_ = cube
# cube_ = block_reduce(cube, block_size=(1,block,block), func=np.mean)
cube_[cube_ < 0.001] = 0

plt.figure(3)
plt.subplot(121)
plt.axis('off')
plt.imshow(np.sum(cube_, axis=0))
# if (block == 1):
#     plt.title('Uncompressed Image')
# else:
#     plt.title('Compressed Image (Block size %i)' %block)
plt.title(name)

# Beam area from THINGS paper, depends on galaxy
beamA = np.pi*bmaj*bmin

# Solid angle
sa = np.abs(1024*1024*dDec*dRA)

# Beam/pixel
beam = sa / beamA

# IMAGE PROCESSING
mid = [np.floor(cube_.shape[1]/2), np.floor(cube_.shape[2]/2)]
velocities = {}
mass = {}

for i in range(cube_.shape[0]):
    v = np.abs(np.floor(cube_.shape[0]/2 - i) * dV)
    for x in range(cube_.shape[1]):
        for y in range(cube_.shape[2]):
            # Rotation curve data
            if (cube_[i,x,y] > 0):
                # r = pixel distance * position change per pixel
                r = np.floor(np.sqrt((dPos*(mid[0]-x))**2 + (dPos*(mid[1]-y))**2))
                if (velocities.get(r,None) is None):
                    velocities[r] = [v]
                else:
                    velocities[r].append(v)

                # Sum all S(v)
                s = np.abs(cube_[i,x,y] * beam * dV * 10**-3)
                if (mass.get(r,None) is None):
                    mass[r] = s
                else:
                    mass[r] = mass[r] + s

# ROTATION CURVE - AVERAGE VELOCITY
r = list(velocities.keys())
v = list(velocities.values())
m = list(mass.values())

# Take mean of velocity for each radius
for i in range(np.shape(v)[0]):
    v[i] = np.mean(v[i])

r,v,m = zip(*sorted(zip(r,v,m)))

# m is currently sum(S*v) (v in km/s)
# M_HI = M_sun * 2.36*10^5 * distance(Mpc)^2 * m
m = np.multiply(1.989*10**30 * 2.36*10**5 * distance**2,m)

for i in range(np.shape(r)[0]):
    # Cumulative sum mass
    if (i>0):
        m[i] = m[i] + m[i-1]

# Ignore division by 0 for r=0
np.seterr(divide='ignore')

# Radius in meters (3.086*10^16 m in a parsec)
rM = np.multiply(np.asarray(r),3.086*10**16)

G = 6.67408 * 10**-11 # m^3/(kg*s^2)
# v = sqrt(GM/r)
v_stellar = np.sqrt(np.multiply(G,np.divide(np.asarray(m),rM)))

print('Total stellar mass: ', m[724], ' kg')

ax = plt.subplot(122)
# ax = plt.subplot(111)
s1 = ax.scatter(r,v, label='Measured velocity')
s2 = ax.scatter(r,v_stellar, label='Expected velocity')
plt.xlabel('Radius (parsec)')
plt.ylabel('Velocity (m/s)')
plt.legend(handles=[s1, s2])
plt.title('Rotation Curve')
plt.show()