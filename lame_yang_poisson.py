import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib
from matplotlib import cm

##################################################################
# 载入反演数据 0
# background = np.load("D:\\Physic_Model_Data\\inversion_back.npy")
# nmat = int(background.shape[0]/3) # 矩阵拆分为三部分
# inv1 = np.exp(background[0:nmat, :])
# inv2 = np.exp(background[nmat:2*nmat, :])
# inv3 = np.exp(background[2*nmat:3*nmat, :])
# np.save("D:\\Physic_Model_Data\\iso_vp.npy", inv1)
# np.save("D:\\Physic_Model_Data\\iso_vs.npy", inv2)
# np.save("D:\\Physic_Model_Data\\iso_rho.npy", inv3)

# # poisson
# sigma = ((inv1/inv2)**2-2)/((inv1/inv2)**2-1)/2
# np.save("D:\\Physic_Model_Data\\iso_sigma.npy", sigma)

# # yang
# yang = 2*inv3*(1+sigma)*inv2**2
# np.save("D:\\Physic_Model_Data\\iso_yang.npy", yang)

# # Zp
# zp = inv1*inv3
# np.save("D:\\Physic_Model_Data\\iso_zp.npy", zp)

# # Zs
# zs = inv2*inv3
# np.save("D:\\Physic_Model_Data\\iso_zs.npy", zs)

##################################################################
# # 载入反演数据 1
# inv1 = np.load("D:\\Physic_Model_Data\\iso_vp.npy") # Vp
# inv2 = np.load("D:\\Physic_Model_Data\\iso_vs.npy") # Vs
# inv3 = np.load("D:\\Physic_Model_Data\\iso_rho.npy") # Rho
# sigma = np.load("D:\\Physic_Model_Data\\iso_sigma.npy")
# yang = np.load("D:\\Physic_Model_Data\\iso_yang.npy")
# zp = np.load("D:\\Physic_Model_Data\\iso_zp.npy")
# zs = np.load("D:\\Physic_Model_Data\\iso_zs.npy")

# 载入反演数据 0
inv1 = np.load("D:\\Physic_Model_Data\\iso_inv1.npy") # Vp
inv2 = np.load("D:\\Physic_Model_Data\\iso_inv2.npy") # Vs
inv3 = np.load("D:\\Physic_Model_Data\\iso_inv3.npy") # Rho
sigma = np.load("D:\\Physic_Model_Data\\iso_sigma.npy")
yang = np.load("D:\\Physic_Model_Data\\iso_yang.npy")
zp = np.load("D:\\Physic_Model_Data\\iso_zp.npy")
zs = np.load("D:\\Physic_Model_Data\\iso_zs.npy")
brittleness = yang/sigma

##################################################################
# font = {"family":"normal",
# 		"weight":"bold",
# 		"size":22}
# matplotlib.rc("font", **font)

# alternatively
matplotlib.rcParams.update({"font.size":22})
##################################################################
# # 绘图
# ax = plt.subplot(111)
# im = ax.imshow(inv1, aspect=1, interpolation="nearest", cmap=cm.jet)
# divider = make_axes_locatable(ax)
# cax = divider.append_axes("right", size="1%", pad=0.05)
# plt.colorbar(im, cax=cax)

plt.figure(figsize=(21, 8),)
plt.subplot(111)
inv1 = inv1[:, 350:1850]
plt.xticks(np.arange(0, 2000, 500), np.arange(350, 2350, 500))
plt.yticks(np.arange(0, 400, 100), np.arange(3000, 3800, 200))
plt.xlabel("CDP")
plt.ylabel("Time(ms)")
plt.imshow(inv1, aspect=2, interpolation="nearest", cmap=cm.jet)
# plt.colorbar(orientation="horizontal", label="Km/s")
plt.colorbar(orientation="vertical", label="Km/s", pad=0.01)
plt.title("$Vp$")
plt.savefig("D:\\Physic_Model_Data\\pictures\\Vp.png", dpi=300)

plt.figure(figsize=(21, 8))
plt.subplot(111)
inv2 = inv2[:, 350:1850]
plt.xticks(np.arange(0, 2000, 500), np.arange(350, 2350, 500))
plt.yticks(np.arange(0, 400, 100), np.arange(3000, 3800, 200))
plt.xlabel("CDP")
plt.ylabel("Time(ms)")
plt.imshow(inv2, aspect=2, cmap=cm.jet)
# plt.colorbar(orientation="horizontal", label="Km/s")
plt.colorbar(orientation="vertical", label="Km/s", pad=0.01)
plt.title("$Vs$")
plt.savefig("D:\\Physic_Model_Data\\pictures\\Vs.png", dpi=300)

plt.figure(figsize=(21, 8))
plt.subplot(111)
inv3 = inv3[:, 350:1850]
plt.xticks(np.arange(0, 2000, 500), np.arange(350, 2350, 500))
plt.yticks(np.arange(0, 400, 100), np.arange(3000, 3800, 200))
plt.xlabel("CDP")
plt.ylabel("Time(ms)")
plt.imshow(inv3, aspect=2, cmap=cm.jet)
# plt.colorbar(orientation="horizontal", label="g/cc")
plt.colorbar(orientation="vertical", label="g/cc", pad=0.01)
plt.title("$Rho$")
plt.savefig("D:\\Physic_Model_Data\\pictures\\Rho.png", dpi=300)

plt.figure(figsize=(21, 8))
plt.subplot(111)
sigma = sigma[:, 350:1850]
plt.xticks(np.arange(0, 2000, 500), np.arange(350, 2350, 500))
plt.yticks(np.arange(0, 400, 100), np.arange(3000, 3800, 200))
plt.xlabel("CDP")
plt.ylabel("Time(ms)")
plt.imshow(sigma, aspect=2, cmap=cm.jet)
# plt.colorbar(orientation="horizontal")
plt.colorbar(orientation="vertical", pad=0.01)
plt.title("$Sigma$")
plt.savefig("D:\\Physic_Model_Data\\pictures\\Sigma.png", dpi=300)

plt.figure(figsize=(21, 8))
plt.subplot(111)
yang = yang[:, 350:1850]
plt.xticks(np.arange(0, 2000, 500), np.arange(350, 2350, 500))
plt.yticks(np.arange(0, 400, 100), np.arange(3000, 3800, 200))
plt.xlabel("CDP")
plt.ylabel("Time(ms)")
plt.imshow(yang, aspect=2, cmap=cm.jet)
# plt.colorbar(orientation="horizontal")
plt.colorbar(orientation="vertical", pad=0.01, label="GPa")
plt.title("$Young$")
plt.savefig("D:\\Physic_Model_Data\\pictures\\Yang.png", dpi=300)

plt.figure(figsize=(21, 8))
plt.subplot(111)
zp = zp[:, 350:1850]
plt.xticks(np.arange(0, 2000, 500), np.arange(350, 2350, 500))
plt.yticks(np.arange(0, 400, 100), np.arange(3000, 3800, 200))
plt.xlabel("CDP")
plt.ylabel("Time(ms)")
plt.imshow(zp, aspect=2, cmap=cm.jet)
# plt.colorbar(orientation="horizontal")
plt.colorbar(orientation="vertical", pad=0.01, label="Km/s*g/cc")
plt.title("$Zp$")
plt.savefig("D:\\Physic_Model_Data\\pictures\\Zp.png", dpi=300)

plt.figure(figsize=(21, 8))
plt.subplot(111)
zs = zs[:, 350:1850]
plt.xticks(np.arange(0, 2000, 500), np.arange(350, 2350, 500))
plt.yticks(np.arange(0, 400, 100), np.arange(3000, 3800, 200))
plt.xlabel("CDP")
plt.ylabel("Time(ms)")
plt.imshow(zs, aspect=2, cmap=cm.jet)
# plt.colorbar(orientation="horizontal")
plt.colorbar(orientation="vertical", pad=0.01, label="Km/s*g/cc")
plt.title("$Zs$")
plt.savefig("D:\\Physic_Model_Data\\pictures\\Zs.png", dpi=300)
# plt.show()

plt.figure(figsize=(21, 8))
plt.subplot(111)
brittleness = brittleness[:, 350:1850]
plt.xticks(np.arange(0, 2000, 500), np.arange(350, 2350, 500))
plt.yticks(np.arange(0, 400, 100), np.arange(3000, 3800, 200))
plt.xlabel("CDP")
plt.ylabel("Time(ms)")
plt.imshow(brittleness, aspect=2, cmap=cm.jet)
# plt.colorbar(orientation="horizontal")
plt.colorbar(orientation="vertical", pad=0.01, label="Brittleness")
plt.title("$Brittleness$")
plt.savefig("D:\\Physic_Model_Data\\pictures\\brittleness.png", dpi=300)
plt.show()
