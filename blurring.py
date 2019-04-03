import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import copy
import blurring_savgol as blur_savgol


window_length, polyorder = 141, 1
##########################################################################
# L1
# 各向同性
image = np.load("D:\\Physic_Model_Data\\line1_Vp_exactly.npy")
image_new = copy.copy(image)
image_new[985:1901, :] = blur_savgol.initmodel_savgol(image_new[985:1901, :], window_length, polyorder)
np.save("D:\\Physic_Model_Data\\line1_Vp.npy", image_new)

image = np.load("D:\\Physic_Model_Data\\line1_Vs_exactly.npy")
image_new = copy.copy(image)
image_new[985:1901, :] = blur_savgol.initmodel_savgol(image_new[985:1901, :], window_length, polyorder)
np.save("D:\\Physic_Model_Data\\line1_Vs.npy", image_new)

image = np.load("D:\\Physic_Model_Data\\line1_Rho_exactly.npy")
image_new = copy.copy(image)
image_new[985:1901, :] = blur_savgol.initmodel_savgol(image_new[985:1901, :], window_length, polyorder)
np.save("D:\\Physic_Model_Data\\line1_Rho.npy", image_new)

# 各向异性
image = np.load("D:\\Physic_Model_Data\\L1_coe0.npy")
image_new = copy.copy(image)
image_new[985:1901, :] = blur_savgol.initmodel_savgol(image_new[985:1901, :], window_length, polyorder)
np.save("D:\\Physic_Model_Data\\coe0_blur.npy", image_new)

image = np.load("D:\\Physic_Model_Data\\L1_coe1.npy")
image_new = copy.copy(image)
image_new[985:1901, :] = blur_savgol.initmodel_savgol(image_new[985:1901, :], window_length, polyorder)
np.save("D:\\Physic_Model_Data\\coe1_blur.npy", image_new)

image = np.load("D:\\Physic_Model_Data\\L1_coe1_new.npy")
image_new = copy.copy(image)
image_new[985:1901, :] = blur_savgol.initmodel_savgol(image_new[985:1901, :], window_length, polyorder)
np.save("D:\\Physic_Model_Data\\coe1_blur_new.npy", image_new)

image = np.load("D:\\Physic_Model_Data\\L1_coe2.npy")
image_new = copy.copy(image)
image_new[985:1901, :] = blur_savgol.initmodel_savgol(image_new[985:1901, :], window_length, polyorder)
np.save("D:\\Physic_Model_Data\\coe2_blur.npy", image_new)

##########################################################################
# L2
# 各向同性
image = np.load("D:\\Physic_Model_Data\\line2_Vp_exactly.npy")
image_new = copy.copy(image)
image_new[985:1901, :] = blur_savgol.initmodel_savgol(image_new[985:1901, :], window_length, polyorder)
np.save("D:\\Physic_Model_Data\\line2_Vp.npy", image_new)

image = np.load("D:\\Physic_Model_Data\\line2_Vs_exactly.npy")
image_new = copy.copy(image)
image_new[985:1901, :] = blur_savgol.initmodel_savgol(image_new[985:1901, :], window_length, polyorder)
np.save("D:\\Physic_Model_Data\\line2_Vs.npy", image_new)

image = np.load("D:\\Physic_Model_Data\\line2_Rho_exactly.npy")
image_new = copy.copy(image)
image_new[985:1901, :] = blur_savgol.initmodel_savgol(image_new[985:1901, :], window_length, polyorder)
np.save("D:\\Physic_Model_Data\\line2_Rho.npy", image_new)

# 各向异性
image = np.load("D:\\Physic_Model_Data\\L2_coe0.npy")
image_new = copy.copy(image)
image_new[985:1901, :] = blur_savgol.initmodel_savgol(image_new[985:1901, :], window_length, polyorder)
np.save("D:\\Physic_Model_Data\\L2_coe0_blur.npy", image_new)

image = np.load("D:\\Physic_Model_Data\\L2_coe1.npy")
image_new = copy.copy(image)
image_new[985:1901, :] = blur_savgol.initmodel_savgol(image_new[985:1901, :], window_length, polyorder)
np.save("D:\\Physic_Model_Data\\L2_coe1_blur.npy", image_new)

image = np.load("D:\\Physic_Model_Data\\L2_coe1_new.npy")
image_new = copy.copy(image)
image_new[985:1901, :] = blur_savgol.initmodel_savgol(image_new[985:1901, :], window_length, polyorder)
np.save("D:\\Physic_Model_Data\\L2_coe1_blur_new.npy", image_new)

image = np.load("D:\\Physic_Model_Data\\L2_coe2.npy")
image_new = copy.copy(image)
image_new[985:1901, :] = blur_savgol.initmodel_savgol(image_new[985:1901, :], window_length, polyorder)
np.save("D:\\Physic_Model_Data\\L2_coe2_blur.npy", image_new)

##########################################################################

# para1, para2 = 100, 3
# w = signal.gaussian(para1, para2)
# w = w/w.sum()

# ##########################################################################
# # L1
# # 各向同性
# image = np.load("D:\\Physic_Model_Data\\line1_Vp_exactly.npy")
# image_new = copy.copy(image)
# image_new[985:1901, :] = signal.sepfir2d(image_new[985:1901, :], w, w)
# np.save("D:\\Physic_Model_Data\\line1_Vp.npy", image_new)

# image = np.load("D:\\Physic_Model_Data\\line1_Vs_exactly.npy")
# image_new = copy.copy(image)
# image_new[985:1901, :] = signal.sepfir2d(image_new[985:1901, :], w, w)
# np.save("D:\\Physic_Model_Data\\line1_Vs.npy", image_new)

# image = np.load("D:\\Physic_Model_Data\\line1_Rho_exactly.npy")
# image_new = copy.copy(image)
# image_new[985:1901, :] = signal.sepfir2d(image_new[985:1901, :], w, w)
# np.save("D:\\Physic_Model_Data\\line1_Rho.npy", image_new)

# # image = np.load("D:\\Physic_Model_Data\\line1_Epsilon_exactly.npy")
# # np.save("D:\\Physic_Model_Data\\line1_Epsilon.npy", image)

# # image = np.load("D:\\Physic_Model_Data\\line1_Delta_exactly.npy")
# # np.save("D:\\Physic_Model_Data\\line1_Delta.npy", image)

# # image = np.load("D:\\Physic_Model_Data\\line1_Gamma_exactly.npy")
# # np.save("D:\\Physic_Model_Data\\line1_Gamma.npy", image)

# # para1, para2 = 200, 50
# # image = np.load("D:\\Physic_Model_Data\\line1_Vp_exactly.npy")
# # w = signal.convolve2d(image, laplacian) # robinson kirsch sobel 
# # image_new = signal.medfilt2d(image, 211)
# # np.save("D:\\Physic_Model_Data\\line1_Vp.npy", image_new)

# # image = np.load("D:\\Physic_Model_Data\\line1_Vs_exactly.npy")
# # # w = signal.convolve2d(image, laplacian) # robinson kirsch sobel 
# # image_new = signal.medfilt2d(image, 211)
# # np.save("D:\\Physic_Model_Data\\line1_Vs.npy", image_new)

# # image = np.load("D:\\Physic_Model_Data\\line1_Rho_exactly.npy")
# # # w = signal.convolve2d(image, laplacian) # robinson kirsch sobel 
# # image_new = signal.medfilt2d(image, 211)
# # np.save("D:\\Physic_Model_Data\\line1_Rho.npy", image_new)

# # image = np.load("D:\\Physic_Model_Data\\line1_Epsilon_exactly.npy")
# # # w = signal.convolve2d(image, laplacian) # robinson kirsch sobel 
# # image_new = signal.medfilt2d(image, 211)
# # np.save("D:\\Physic_Model_Data\\line1_Epsilon.npy", image_new)

# # image = np.load("D:\\Physic_Model_Data\\line1_Delta_exactly.npy")
# # # w = signal.convolve2d(image, laplacian) # robinson kirsch sobel 
# # image_new = signal.medfilt2d(image, 211)
# # np.save("D:\\Physic_Model_Data\\line1_Delta.npy", image_new)

# # image = np.load("D:\\Physic_Model_Data\\line1_Gamma_exactly.npy")
# # # w = signal.convolve2d(image, laplacian) # robinson kirsch sobel 
# # image_new = signal.medfilt2d(image, 211)
# # np.save("D:\\Physic_Model_Data\\line1_Gamma.npy", image_new)

# # 各向异性
# image = np.load("D:\\Physic_Model_Data\\L1_coe0.npy")
# image_new = copy.copy(image)
# image_new[985:1901, :] = signal.sepfir2d(image_new[985:1901, :], w, w)
# np.save("D:\\Physic_Model_Data\\coe0_blur.npy", image_new)

# image = np.load("D:\\Physic_Model_Data\\L1_coe1.npy")
# image_new = copy.copy(image)
# image_new[985:1901, :] = signal.sepfir2d(image_new[985:1901, :], w, w)
# np.save("D:\\Physic_Model_Data\\coe1_blur.npy", image_new)

# image = np.load("D:\\Physic_Model_Data\\L1_coe2.npy")
# image_new = copy.copy(image)
# image_new[985:1901, :] = signal.sepfir2d(image_new[985:1901, :], w, w)
# np.save("D:\\Physic_Model_Data\\coe2_blur.npy", image_new)

# ##########################################################################
# # L2
# # 各向同性
# image = np.load("D:\\Physic_Model_Data\\line2_Vp_exactly.npy")
# image_new = copy.copy(image)
# image_new[985:1901, :] = signal.sepfir2d(image_new[985:1901, :], w, w)
# np.save("D:\\Physic_Model_Data\\line2_Vp.npy", image_new)

# image = np.load("D:\\Physic_Model_Data\\line2_Vs_exactly.npy")
# image_new = copy.copy(image)
# image_new[985:1901, :] = signal.sepfir2d(image_new[985:1901, :], w, w)
# np.save("D:\\Physic_Model_Data\\line2_Vs.npy", image_new)

# image = np.load("D:\\Physic_Model_Data\\line2_Rho_exactly.npy")
# image_new = copy.copy(image)
# image_new[985:1901, :] = signal.sepfir2d(image_new[985:1901, :], w, w)
# np.save("D:\\Physic_Model_Data\\line2_Rho.npy", image_new)

# # image = np.load("D:\\Physic_Model_Data\\line2_Epsilon_exactly.npy")
# # np.save("D:\\Physic_Model_Data\\line2_Epsilon.npy", image)

# # image = np.load("D:\\Physic_Model_Data\\line2_Delta_exactly.npy")
# # np.save("D:\\Physic_Model_Data\\line2_Delta.npy", image)

# # image = np.load("D:\\Physic_Model_Data\\line2_Gamma_exactly.npy")
# # np.save("D:\\Physic_Model_Data\\line2_Gamma.npy", image)

# # para1, para2 = 200, 50
# # image = np.load("D:\\Physic_Model_Data\\line2_Vp_exactly.npy")
# # w = signal.convolve2d(image, laplacian) # robinson kirsch sobel 
# # image_new = signal.medfilt2d(image, 211)
# # np.save("D:\\Physic_Model_Data\\line2_Vp.npy", image_new)

# # image = np.load("D:\\Physic_Model_Data\\line2_Vs_exactly.npy")
# # # w = signal.convolve2d(image, laplacian) # robinson kirsch sobel 
# # image_new = signal.medfilt2d(image, 211)
# # np.save("D:\\Physic_Model_Data\\line2_Vs.npy", image_new)

# # image = np.load("D:\\Physic_Model_Data\\line2_Rho_exactly.npy")
# # # w = signal.convolve2d(image, laplacian) # robinson kirsch sobel 
# # image_new = signal.medfilt2d(image, 211)
# # np.save("D:\\Physic_Model_Data\\line2_Rho.npy", image_new)

# # image = np.load("D:\\Physic_Model_Data\\line2_Epsilon_exactly.npy")
# # # w = signal.convolve2d(image, laplacian) # robinson kirsch sobel 
# # image_new = signal.medfilt2d(image, 211)
# # np.save("D:\\Physic_Model_Data\\line2_Epsilon.npy", image_new)

# # image = np.load("D:\\Physic_Model_Data\\line2_Delta_exactly.npy")
# # # w = signal.convolve2d(image, laplacian) # robinson kirsch sobel 
# # image_new = signal.medfilt2d(image, 211)
# # np.save("D:\\Physic_Model_Data\\line2_Delta.npy", image_new)

# # image = np.load("D:\\Physic_Model_Data\\line2_Gamma_exactly.npy")
# # # w = signal.convolve2d(image, laplacian) # robinson kirsch sobel 
# # image_new = signal.medfilt2d(image, 211)
# # np.save("D:\\Physic_Model_Data\\line2_Gamma.npy", image_new)

# # 各向异性
# image = np.load("D:\\Physic_Model_Data\\L2_coe0.npy")
# image_new = copy.copy(image)
# image_new[985:1901, :] = signal.sepfir2d(image_new[985:1901, :], w, w)
# np.save("D:\\Physic_Model_Data\\L2_coe0_blur.npy", image_new)

# image = np.load("D:\\Physic_Model_Data\\L2_coe1.npy")
# image_new = copy.copy(image)
# image_new[985:1901, :] = signal.sepfir2d(image_new[985:1901, :], w, w)
# np.save("D:\\Physic_Model_Data\\L2_coe1_blur.npy", image_new)

# image = np.load("D:\\Physic_Model_Data\\L2_coe2.npy")
# image_new = copy.copy(image)
# image_new[985:1901, :] = signal.sepfir2d(image_new[985:1901, :], w, w)
# np.save("D:\\Physic_Model_Data\\L2_coe2_blur.npy", image_new)

# ##########################################################################

a=image_new[:, 1696]
b=image[:, 1696]

c=image_new[:, 425]
d=image[:, 425]

# a=image_new[1200:1900, 1696]
# b=image[1200:1900, 1696]

# c=image_new[1200:1900, 425]
# d=image[1200:1900, 425]

plt.figure()
plt.plot(a)
plt.plot(b)
plt.plot(c)
plt.plot(d)
plt.show()

# plt.figure() 
# plt.imshow(image) 
# plt.gray() 
# plt.title('Original image') 
# plt.show()
# plt.figure() 
# plt.imshow(image_new) 
# plt.gray() 
# plt.title('Filtered image') 
# plt.show()

# plt.figure()
# plt.plot(a)
# plt.plot(b, "red")
# plt.show()

# plt.figure()
# plt.set_cmap("cubehelix_r")
# image_new = image_new[985:1901, :]
# image = image[985:1901, :]
# vmin = np.percentile(image_new[image_new>1500], 1)
# vmax = np.percentile(image_new[image_new>0], 99)
# print("min={}, max={}".format(vmin, vmax))
# plt.imshow(image_new, aspect=0.4, vmin=vmin, vmax=vmax)
# plt.colorbar(shrink=0.75)
# plt.show()
