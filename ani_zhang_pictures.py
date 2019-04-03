import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib
from matplotlib import cm


##################################################################
# 载入反演数据 0
inv1 = np.load("D:\\Physic_Model_Data\\ani_inv1.npy") # AVO1
inv2 = np.load("D:\\Physic_Model_Data\\ani_inv2.npy") # AVO2
inv3 = np.load("D:\\Physic_Model_Data\\ani_inv3.npy") # AVO3
vp = np.load("D:\\Physic_Model_Data\\iso_inv1.npy") # Vp
eps = np.log(inv3/vp)

##################################################################
# font = {"family":"normal",
# 		"weight":"bold",
# 		"size":22}
# matplotlib.rc("font", **font)

# alternatively
matplotlib.rcParams.update({"font.size":22})

##################################################################
# 绘图
plt.figure(figsize=(21, 8),)
plt.subplot(111)
inv1 = inv1[:, 350:1850]
plt.xticks(np.arange(0, 2000, 500), np.arange(350, 2350, 500))
plt.yticks(np.arange(0, 400, 100), np.arange(3000, 3800, 200))
plt.xlabel("CDP")
plt.ylabel("Time(ms)")
plt.imshow(inv1, aspect=2, cmap=cm.jet)
plt.colorbar(orientation="vertical", label="AVO1", pad=0.01)
plt.title("$AVO1$")
plt.savefig("D:\\Physic_Model_Data\\pictures\\AVO1.png", dpi=300)
# plt.show()

plt.figure(figsize=(21, 8),)
plt.subplot(111)
inv2 = inv2[:, 350:1850]
plt.xticks(np.arange(0, 2000, 500), np.arange(350, 2350, 500))
plt.yticks(np.arange(0, 400, 100), np.arange(3000, 3800, 200))
plt.xlabel("CDP")
plt.ylabel("Time(ms)")
plt.imshow(inv2, aspect=2, cmap=cm.jet)
plt.colorbar(orientation="vertical", label="AVO2", pad=0.01)
plt.title("$AVO2$")
plt.savefig("D:\\Physic_Model_Data\\pictures\\AVO2.png", dpi=300)
# plt.show()

plt.figure(figsize=(21, 8),)
plt.subplot(111)
inv3 = inv3[:, 350:1850]
plt.xticks(np.arange(0, 2000, 500), np.arange(350, 2350, 500))
plt.yticks(np.arange(0, 400, 100), np.arange(3000, 3800, 200))
plt.xlabel("CDP")
plt.ylabel("Time(ms)")
plt.imshow(inv3, aspect=2, cmap=cm.jet)
plt.colorbar(orientation="vertical", label="AVO3", pad=0.01)
plt.title("$AVO3$")
plt.savefig("D:\\Physic_Model_Data\\pictures\\AVO3.png", dpi=300)
# plt.show()

plt.figure(figsize=(21, 8),)
plt.subplot(111)
eps = eps[:, 350:1850]
plt.xticks(np.arange(0, 2000, 500), np.arange(350, 2350, 500))
plt.yticks(np.arange(0, 400, 100), np.arange(3000, 3800, 200))
plt.xlabel("CDP")
plt.ylabel("Time(ms)")
plt.imshow(eps, aspect=2, cmap=cm.jet)
plt.colorbar(orientation="vertical", label="AVO3", pad=0.01)
plt.title("$Epsilon$")
plt.savefig("D:\\Physic_Model_Data\\pictures\\Epsilon.png", dpi=300)
plt.show()
