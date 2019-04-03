import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib


##################################################################
# font = {"family":"normal",
# 		"weight":"bold",
# 		"size":22}
# matplotlib.rc("font", **font)

# alternatively
# matplotlib.rcParams.update({"font.size":20})

##################################################################

##############################################################
# # 样品参数（泥质含量）(%)
# clay_volume = np.array([0.25, 0.3, 0.35, 0.4, 0.45, 0.5])*100

# vpx = np.array([3122.22,3109.58,3090.56,3086.241,3068.395,3055.92])
# vpz = np.array([2776,2732,2690,2650,2585,2568])

# vsx = np.array([2142.3,2133.31,2105.236,2099.691,2072.82,2059.156])
# vsz = np.array([1950,1917,1878,1837,1790,1766])

# zp = np.array([7.30088,7.13052,6.8864,6.731,6.5142,6.36864])
# zs = np.array([5.1285,5.00337,4.80768,4.66598,4.5108,4.37968])

# pani = np.array([0.12471902,0.138206442,0.148907063,0.164619245,0.187,0.19])
# sani = np.array([0.098615385,0.112837767,0.120998935,0.143,0.158,0.166])

# rho = np.array([2.63,2.61,2.56,2.54,2.52,2.48])
# epsilon = np.array([0.12471902,0.138206442,0.148907063,0.164619245,0.187,0.19])
# gamma = np.array([0.098615385,0.112837767,0.120998935,0.143,0.158,0.166])

# young = np.array([20.26034625,19.47162115,18.50147123,17.78511885,16.78480239,16.26367676])
# poisson = np.array([0.012959067,0.015049875,0.02457824,0.037468088,0.039392633,0.05137019])
# brittleness = np.array([1563.410837,1293.806129,752.7581897,474.6737762,426.0898882,316.5975583])

# c11 = np.array([25.63791783,25.2373631,24.45199645,24.19320412,23.72592065,23.15984468])
# c33 = np.array([20.26724288,19.48058064,18.524416,17.83715,16.839207,16.35466752])
# c44 = np.array([10.000575,9.59146029,9.02882304,8.57140526,8.074332,7.73451488])
# c66 = np.array([12.07025163,11.87814016,11.34596766,11.19810383,10.82738854,10.51550611])

# c13bottom = np.array([1.197733861,1.158061973,1.355807289,1.349837224,1.505849397,1.54236334])
# c13top = np.array([5.508944054,5.371438579,5.710000413,5.661562974,5.905625741,5.900537851])
# deltabottom = np.array([0.048053478,0.046088539,0.050239425,0.038048908,0.050669024,0.041684851])
# deltatop = np.array([0.324737185,0.327268067,0.361189324,0.353119732,0.401833919,0.395828645])

##############################################################
# 孔隙度(%)
porosity = np.array([2.1,3.8,4.7,6.5,11.5,16.4])

vpx = np.array([3748.976,3512.03,3267.856,2998.594,2742.45,2505.8])
vpz = np.array([2992,2830,2672,2503,2350,2200])

vsx = np.array([2496.382,2311.764,2119.98,1942.97,1735.776,1564.014])
vsz = np.array([2023,1898,1780,1655,1512,1389])

zp = np.array([7.65952,7.1882,6.65328,6.13235,5.6635,5.236])
zs = np.array([5.17888,4.82092,4.4322,4.05475,3.64392,3.30582])

pani = np.array([0.253,0.241,0.223,0.198,0.167,0.139])
sani = np.array([0.234,0.218,0.191,0.174,0.148,0.126])

rho = np.array([2.56,2.54,2.49,2.45,2.41,2.38])
epsilon = np.array([0.253,0.241,0.223,0.198,0.167,0.139])
gamma = np.array([0.234,0.218,0.191,0.174,0.148,0.126])

young = np.array([22.60736871,19.9699125,17.37347542,14.91895361,12.63686514,10.73172354])
poisson = np.array([0.078917633,0.091239388,0.101076153,0.111594239,0.146802762,0.168578878])
brittleness = np.array([286.4679012,218.8738093,171.8850092,133.6892817,86.08056797,63.65995355])

c11 = np.array([35.98034188,31.32926099,26.59041826,22.02933664,18.12568713,14.94410006])
c33 = np.array([22.91728384,20.342606,17.77756416,15.34927205,13.309225,11.5192])
c44 = np.array([10.47687424,9.15010616,7.889316,6.71061125,5.50960704,4.59178398])
c66 = np.array([15.95372311,13.57440209,11.19084485,9.249074431,7.261133156,5.821812705])

c13bottom = np.array([2.697308686,2.836136532,2.953268787,2.572369632,2.77294842,2.658301138])
c13top = np.array([9.661247641,9.221788643,8.649909948,7.362143839,6.925224789,6.165940935])
deltabottom = np.array([0.032962607,0.040402293,0.056272568,0.04354269,0.0374093,0.028663781])
deltatop = np.array([0.439810263,0.466115083,0.499941911,0.465379425,0.451757201,0.424442379])

##############################################################
# # 绘图
# x = clay_volume

# fig, ax = plt.subplots()
# ax.plot(x, vpx, marker="o", color="red", label="Vpx")
# ax.plot(x, vpz, marker="o", color="red", label="Vpz", linestyle="--")
# ax.plot(x, vsx, marker="o", color="blue", label="Vsx")
# ax.plot(x, vsz, marker="o", color="blue", label="Vsz", linestyle="--")
# ax.set_xlabel("Clay Volume(%)")
# ax.set_ylabel("Velocity(m/s)")
# ax.legend()
# plt.savefig("D:\\Physic_Model_Data\\stone\\Velocity", dpi=300)
# # plt.show()

# fig, ax = plt.subplots()
# ax.plot(x, zp, marker="o", color="red", label="Zp")
# ax.plot(x, zs, marker="o", color="blue", label="Zs")
# ax.set_xlabel("Clay Volume(%)")
# ax.set_ylabel("Impedance(m/s*g/cc)")
# ax.legend()
# plt.savefig("D:\\Physic_Model_Data\\stone\\Impedance", dpi=300)
# # plt.show()

# fig, ax = plt.subplots()
# ax.plot(x, pani, marker="o", color="red", label="P_ani")
# ax.plot(x, sani, marker="o", color="blue", label="S_ani")
# ax.set_xlabel("Clay Volume(%)")
# ax.set_ylabel("ani")
# ax.legend()
# plt.savefig("D:\\Physic_Model_Data\\stone\\ani", dpi=300)
# # plt.show()

# fig, ax = plt.subplots()
# ax.plot(x, rho, marker="o", color="black", label=r"$\rho$")
# ax.set_xlabel("Clay Volume(%)")
# ax.set_ylabel("Density(g/cc)")
# ax.legend()
# plt.savefig("D:\\Physic_Model_Data\\stone\\Density", dpi=300)
# # plt.show()

# fig, ax = plt.subplots()
# ax.plot(x, epsilon, marker="o", color="red", label=r"$\epsilon$")
# ax.plot(x, gamma, marker="o", color="blue", label=r"$\gamma$")
# ax.set_xlabel("Clay Volume(%)")
# ax.set_ylabel("ani params")
# ax.legend()
# plt.savefig("D:\\Physic_Model_Data\\stone\\ani params", dpi=300)
# # plt.show()

# fig, ax = plt.subplots()
# ax.plot(x, young, marker="o", color="black", label="E")
# ax.set_xlabel("Clay Volume(%)")
# ax.set_ylabel("Young's modulus(GPa)")
# ax.legend()
# plt.savefig("D:\\Physic_Model_Data\\stone\\Young", dpi=300)
# # plt.show()

# fig, ax = plt.subplots()
# ax.plot(x, poisson, marker="o", color="black", label=r"$\nu$")
# ax.set_xlabel("Clay Volume(%)")
# ax.set_ylabel("Poisson ratio")
# ax.legend()
# plt.savefig("D:\\Physic_Model_Data\\stone\\Poisson", dpi=300)
# # plt.show()

# fig, ax = plt.subplots()
# ax.plot(x, brittleness, marker="o", color="black", label="brittleness")
# ax.set_xlabel("Clay Volume(%)")
# ax.set_ylabel("Brittleness")
# ax.legend()
# plt.savefig("D:\\Physic_Model_Data\\stone\\Brittleness", dpi=300)
# # plt.show()

# fig, ax = plt.subplots()
# ax.plot(x, c11, marker="o", label="C11")
# ax.plot(x, c33, marker="o", label="C33")
# ax.plot(x, c44, marker="o", label="C44")
# ax.plot(x, c66, marker="o", label="C66")
# ax.set_xlabel("Clay Volume(%)")
# ax.set_ylabel("Params")
# ax.legend()
# plt.savefig("D:\\Physic_Model_Data\\stone\\Params1", dpi=300)
# # plt.show()

# fig, ax = plt.subplots()
# ax.plot(x, c13top, marker="o", color="black", label="C13+")
# ax.plot(x, c13bottom, marker="o", color="black", label="C13-", linestyle="--")
# ax.set_xlabel("Clay Volume(%)")
# ax.set_ylabel("Params")
# ax.legend()
# plt.savefig("D:\\Physic_Model_Data\\stone\\Params2", dpi=300)
# # plt.show()

# fig, ax = plt.subplots()
# ax.plot(x, deltatop, marker="o", color="black", label=r"$\delta+$")
# ax.plot(x, deltabottom, marker="o", color="black", label=r"$\delta-$", linestyle="--")
# ax.set_xlabel("Clay Volume(%)")
# ax.set_ylabel("Params")
# ax.legend()
# plt.savefig("D:\\Physic_Model_Data\\stone\\Params3", dpi=300)
# plt.show()

##############################################################
# 绘图
x = porosity

fig, ax = plt.subplots()
ax.plot(x, vpx, marker="o", color="red", label="Vpx")
ax.plot(x, vpz, marker="o", color="red", label="Vpz", linestyle="--")
ax.plot(x, vsx, marker="o", color="blue", label="Vsx")
ax.plot(x, vsz, marker="o", color="blue", label="Vsz", linestyle="--")
ax.set_xlabel("Porosity(%)")
ax.set_ylabel("Velocity(m/s)")
ax.legend()
plt.savefig("D:\\Physic_Model_Data\\stone\\Velocity", dpi=300)
# plt.show()

fig, ax = plt.subplots()
ax.plot(x, zp, marker="o", color="red", label="Zp")
ax.plot(x, zs, marker="o", color="blue", label="Zs")
ax.set_xlabel("Porosity(%)")
ax.set_ylabel("Impedance(m/s*g/cc)")
ax.legend()
plt.savefig("D:\\Physic_Model_Data\\stone\\Impedance", dpi=300)
# plt.show()

fig, ax = plt.subplots()
ax.plot(x, pani, marker="o", color="red", label="P_ani")
ax.plot(x, sani, marker="o", color="blue", label="S_ani")
ax.set_xlabel("Porosity(%)")
ax.set_ylabel("ani")
ax.legend()
plt.savefig("D:\\Physic_Model_Data\\stone\\ani", dpi=300)
# plt.show()

fig, ax = plt.subplots()
ax.plot(x, rho, marker="o", color="black", label=r"$\rho$")
ax.set_xlabel("Porosity(%)")
ax.set_ylabel("Density(g/cc)")
ax.legend()
plt.savefig("D:\\Physic_Model_Data\\stone\\Density", dpi=300)
# plt.show()

fig, ax = plt.subplots()
ax.plot(x, epsilon, marker="o", color="red", label=r"$\epsilon$")
ax.plot(x, gamma, marker="o", color="blue", label=r"$\gamma$")
ax.set_xlabel("Porosity(%)")
ax.set_ylabel("ani params")
ax.legend()
plt.savefig("D:\\Physic_Model_Data\\stone\\ani params", dpi=300)
# plt.show()

fig, ax = plt.subplots()
ax.plot(x, young, marker="o", color="black", label="E")
ax.set_xlabel("Porosity(%)")
ax.set_ylabel("Young's modulus(GPa)")
ax.legend()
plt.savefig("D:\\Physic_Model_Data\\stone\\Young", dpi=300)
# plt.show()

fig, ax = plt.subplots()
ax.plot(x, poisson, marker="o", color="black", label=r"$\nu$")
ax.set_xlabel("Porosity(%)")
ax.set_ylabel("Poisson ratio")
ax.legend()
plt.savefig("D:\\Physic_Model_Data\\stone\\Poisson", dpi=300)
# plt.show()

fig, ax = plt.subplots()
ax.plot(x, brittleness, marker="o", color="black", label="brittleness")
ax.set_xlabel("Porosity(%)")
ax.set_ylabel("Brittleness")
ax.legend()
plt.savefig("D:\\Physic_Model_Data\\stone\\Brittleness", dpi=300)
# plt.show()

fig, ax = plt.subplots()
ax.plot(x, c11, marker="o", label="C11")
ax.plot(x, c33, marker="o", label="C33")
ax.plot(x, c44, marker="o", label="C44")
ax.plot(x, c66, marker="o", label="C66")
ax.set_xlabel("Porosity(%)")
ax.set_ylabel("Params")
ax.legend()
plt.savefig("D:\\Physic_Model_Data\\stone\\Params1", dpi=300)
# plt.show()

fig, ax = plt.subplots()
ax.plot(x, c13top, marker="o", color="black", label="C13+")
ax.plot(x, c13bottom, marker="o", color="black", label="C13-", linestyle="--")
ax.set_xlabel("Porosity(%)")
ax.set_ylabel("Params")
ax.legend()
plt.savefig("D:\\Physic_Model_Data\\stone\\Params2", dpi=300)
# plt.show()

fig, ax = plt.subplots()
ax.plot(x, deltatop, marker="o", color="black", label=r"$\delta+$")
ax.plot(x, deltabottom, marker="o", color="black", label=r"$\delta-$", linestyle="--")
ax.set_xlabel("Porosity(%)")
ax.set_ylabel("Params")
ax.legend()
plt.savefig("D:\\Physic_Model_Data\\stone\\Params3", dpi=300)
plt.show()
