import numpy as np 
import matplotlib.pyplot as plt 


# 矿物含量（粘土和石英相对变化）
temp = np.arange(100)/100
# kerogen = 0
# calcite = 0
kerogen = 0.07
calcite = 0.18
clay = (1 - kerogen - calcite) * temp
quartz = 1 - kerogen - calcite - clay

# 体积模量(kerogen, calcite, clay, quartz)
k1, k2, k3, k4 = 2.9, 77, 25, 37

# 剪切模量
g1, g2, g3, g4 = 2.7, 32, 9, 44

def voigt(arg1, arg2, arg3, arg4):
	# 所占体积单位为1
	v1, v2, v3, v4 = arg1, arg2, arg3, arg4
	# 总体积模量
	k = arg1*k1 + arg2*k2 + arg3*k3 + arg4*k4
	# 总剪切模量
	g = arg1*g1 + arg2*g2 + arg3*g3 + arg4*g4
	return [k, g]

def reuss(arg1, arg2, arg3, arg4):
	# 所占体积单位为1
	v1, v2, v3, v4 = arg1, arg2, arg3, arg4
	# 总体积模量
	k = 1/(arg1/k1 + arg2/k2 + arg3/k3 + arg4/k4)
	# 总剪切模量
	g = 1/(arg1/g1 + arg2/g2 + arg3/g3 + arg4/g4)
	return [k, g]

def hashin_shtrikman():
	pass

def KT(k_stroma, u_stroma, aerfa):
    global u_equ, k_equ
    beta = u_stroma*(3*k_stroma+u_stroma)/(3*k_stroma+4*u_stroma)
    kesi = u_stroma*(9*k_stroma+8*u_stroma)/(6*(k_stroma+2*u_stroma))
    p_mi = k_stroma/(k_water+pi*aerfa*beta)
    q_mi = (1+8*u_stroma/(pi*aerfa*(u_stroma+2*beta))+2*(k_water+2*u_stroma/3)/(k_water+pi*aerfa*beta))/5
    kequ = (k_stroma*(k_stroma+4*u_stroma/3)+ porosity *p_mi*4*u_stroma*(k_water-k_stroma)/3)/((k_stroma+4*u_stroma/3)- porosity *p_mi*(k_water-k_stroma))
    uequ = u_stroma*(u_stroma+kesi- porosity *kesi*q_mi)/(u_stroma+kesi+ porosity *u_stroma*q_mi)
    return kequ, uequ

# 体积模量与剪切模量
v_k = []
v_g = []
r_k = []
r_g = []
for i,j in zip(clay, quartz):
	v_k_temp, v_g_temp = voigt(kerogen, calcite, i, j)
	r_k_temp, r_g_temp = reuss(kerogen, calcite, i, j)
	v_k.append(v_k_temp)
	v_g.append(v_g_temp)
	r_k.append(r_k_temp)
	r_g.append(r_g_temp)

# 体积模量	
v_k = np.array(v_k)
r_k = np.array(r_k)

# 剪切模量
v_g = np.array(v_g)
r_g = np.array(r_g)

# 泊松比
v_mu = (3*v_k-2*v_g)/(6*v_k+2*v_g)
r_mu = (3*r_k-2*r_g)/(6*r_k+2*v_g)

# 杨氏模量
v_e = 9*v_k*v_g/(3*v_k+v_g)
r_e = 9*r_k*r_g/(3*r_k+r_g)

# 平面波模量
v_h = v_k+4/3*v_g
r_h = r_k+4/3*r_g

##############################################################
# 样品参数（泥质含量）(%)
clay_volume = np.array([0.25, 0.3, 0.35, 0.4, 0.45, 0.5])*100

# 杨氏模量
young = np.array([20.26034625,
				19.47162115,
				18.50147123,
				17.78511885,
				16.78480239,
				16.26367676])

# 泊松比
poisson = np.array([0.012959067,
				0.015049875,
				0.02457824,
				0.037468088,
				0.039392633,
				0.05137019])

# 剪切模量
shear = np.array([10.000575,
				9.59146029,
				9.02882304,
				8.57140526,
				8.074332,
				7.73451488,])

# 平面波模量
plane = np.array([20.26724288,
				19.48058064,
				18.524416,
				17.83715,
				16.839207,
				16.35466752,])
##############################################################

# 绘图
x = (clay)*100

plt.figure()
plt.plot(x, v_k, color="black")
plt.plot(x, r_k, color="black")
plt.plot(x, (v_k+r_k)/2, color="black", linestyle="--")
plt.xlim(x.min(), x.max())
plt.annotate('V', xy=(x[70]+1, v_k[70]+1), xycoords='data', size=15)
plt.annotate('R', xy=(x[60]+1, r_k[60]+1), xycoords='data', size=15)
plt.annotate('(V+R)/2', xy=(x[65]+1, ((v_k+r_k)/2)[65]+1), xycoords='data', size=15)
plt.xlabel("Clay volume(%)", size=15)
plt.ylabel("Bulk modulus(GPa)", size=15)
plt.savefig("D:\\Physic_Model_Data\\Bulk", dpi=300)
# plt.show()

plt.figure()
plt.plot(x, v_g, color="black")
plt.plot(x, r_g, color="black")
plt.plot(x, (v_g+r_g)/2, color="black", linestyle="--")
plt.xlim(x.min(), x.max())
plt.annotate('V', xy=(x[70]+1, v_g[70]+1), xycoords='data', size=15)
plt.annotate('R', xy=(x[60]+1, r_g[60]+1), xycoords='data', size=15)
plt.annotate('(V+R)/2', xy=(x[65]+1, ((v_g+r_g)/2)[65]+1), xycoords='data', size=15)
plt.scatter(clay_volume, shear, c=clay_volume)
plt.xlabel("Clay volume(%)", size=15)
plt.ylabel("Shear modulus(C44)(GPa)", size=15)
plt.savefig("D:\\Physic_Model_Data\\Shear", dpi=300)
# plt.show()

plt.figure()
plt.plot(x, v_h, color="black")
plt.plot(x, r_h, color="black")
plt.plot(x, (v_h+r_h)/2, color="black", linestyle="--")
plt.xlim(x.min(), x.max())
plt.annotate('V', xy=(x[70]+1, v_h[70]+1), xycoords='data', size=15)
plt.annotate('R', xy=(x[60]+1, r_h[60]+1), xycoords='data', size=15)
plt.annotate('(V+R)/2', xy=(x[65]+1, ((v_h+r_h)/2)[65]+1), xycoords='data', size=15)
plt.scatter(clay_volume, plane, c=clay_volume)
plt.xlabel("Clay volume(%)", size=15)
plt.ylabel("Plane wave modulus(C33)(GPa)", size=15)
plt.savefig("D:\\Physic_Model_Data\\Planewave", dpi=300)
# plt.show()

plt.figure()
plt.plot(x, v_e, color="black")
plt.plot(x, r_e, color="black")
plt.plot(x, (v_e+r_e)/2, color="black", linestyle="--")
plt.xlim(x.min(), x.max())
plt.annotate('V', xy=(x[70]+1, v_e[70]+1), xycoords='data', size=15)
plt.annotate('R', xy=(x[60]+1, r_e[60]+1), xycoords='data', size=15)
plt.annotate('(V+R)/2', xy=(x[65]+1, ((v_e+r_e)/2)[65]+1), xycoords='data', size=15)
plt.scatter(clay_volume, young, c=clay_volume)
plt.xlabel("Clay volume(%)", size=15)
plt.ylabel("Young's modulus(GPa)", size=15)
plt.savefig("D:\\Physic_Model_Data\\Young", dpi=300)
# plt.show()

plt.figure()
plt.plot(x, v_mu, color="black")
plt.plot(x, r_mu, color="black")
plt.plot(x, (v_mu+r_mu)/2, color="black", linestyle="--")
plt.xlim(x.min(), x.max())
plt.annotate('V', xy=(x[90]-1, v_mu[90]), xycoords='data', size=15)
plt.annotate('R', xy=(x[90]-1, r_mu[90]), xycoords='data', size=15)
plt.annotate('(V+R)/2', xy=(x[55], ((v_mu+r_mu)/2)[55]), xycoords='data', size=15)
plt.scatter(clay_volume, poisson, c=clay_volume)
plt.xlabel("Clay volume(%)", size=15)
plt.ylabel("Poisson's ratio(GPa)", size=15)
plt.savefig("D:\\Physic_Model_Data\\Poisson", dpi=300)
plt.show()
