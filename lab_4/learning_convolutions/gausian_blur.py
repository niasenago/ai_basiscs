import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import fftconvolve

def gaussian_kernel(kernel_size, sigma):
    """
    Create a Gaussian kernel.
    :param kernel_size: Size of the kernel (must be odd).
    :param sigma: Standard deviation of the Gaussian distribution.
    :return: Gaussian kernel as a 2D numpy array.
    """
    ax = np.arange(-(kernel_size // 2), kernel_size // 2 + 1)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    return kernel / np.sum(kernel) 

img = cv2.imread('carrot.jpg')
# Convert BGR image to RGB
image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


## blur and all naughty stuff
height, width, channels = image_rgb.shape

kernel_size = 63  # Kernel size (must be odd)
sigma = kernel_size / 6   # Standard deviation for Gaussian distribution
kernel_matrix = gaussian_kernel(kernel_size, sigma)

half_kernel = kernel_size // 2


blured_image = np.zeros_like(image_rgb, dtype=np.float32)
# for i in range(half_kernel, height - half_kernel):
#     for j in range(half_kernel, width - half_kernel):
#         for c in range(channels):  # Loop over color channels
#             # Extract the window 
#             window = image_rgb[i - half_kernel:i + half_kernel + 1, j - half_kernel:j + half_kernel + 1, c]            
#             # Apply the kernel to the window
#             blured_pixel = np.sum(window * kernel_matrix)
            
#             blured_image[i, j, c] = blured_pixel

for c in range(channels):  # Apply per channel
    blured_image[:, :, c] = fftconvolve(image_rgb[:, :, c], kernel_matrix, mode='same')            

# Convert back to uint8 (if needed)
blured_image = np.clip(blured_image, 0, 255).astype(np.uint8)


# Create subplots
fig, axs = plt.subplots(1, 2, figsize=(10, 5))


axs[0].imshow(image_rgb)
axs[0].set_title('Original Image')


axs[1].imshow(blured_image)
axs[1].set_title('Blured image')

# Remove ticks from the subplots
for ax in axs:
    ax.set_xticks([])
    ax.set_yticks([])

# Display the subplots
plt.tight_layout()
plt.show()