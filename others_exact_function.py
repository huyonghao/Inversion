import numpy as np
import scipy.linalg as linalg
import pandas as pd
from scipy.optimize import fsolve

'''
WRONG EQUATION !!!
'''
###################################
# rho1=2.68
# a01=4.345
# b01=2.584
# C33_1=a01**2*rho1
# C44_1=b01**2*rho1
# E1=0.26
# G1=0.18
# D1=0.12
# C11_1=2.0*C33_1*E1+C33_1
# C66_1=2.0*C44_1*G1+C44_1
# C13_1=(D1*2*C33_1*(C33_1-C44_1)+(C33_1-C44_1)**2)**0.5-C44_1

# rho2=2.538
# a02=3.670
# b02=2.149
# C33_2=a02**2*rho2
# C44_2=b02**2*rho2
# E2=0.1
# G2=0.1
# D2=0.12
# C11_2=2.0*C33_2*E2+C33_2
# C66_2=2.0*C44_2*G2+C44_2
# C13_2=(D2*2*C33_2*(C33_2-C44_2)+(C33_2-C44_2)**2)**0.5-C44_2

# C55_1=C44_1
# C55_2=C44_2
# C55_0=(C55_1+C55_2)/2
# d_C55=C55_2-C55_1


###################################
# # #读取测井数据
def others_function(data, theta_max, theta_nums):

	a, b, rho = data["Vp"], data["Vs"], data["Rho"]
	E, D, G = data["Epsilon"], data["Delta"], data["Gamma"]

	a, b, rho = np.array(a).reshape(-1,1), np.array(b).reshape(-1,1), np.array(rho).reshape(-1,1)
	E, D, G = np.array(E).reshape(-1,1), np.array(D).reshape(-1,1), np.array(G).reshape(-1,1)

	angle_range = np.linspace(0.1, theta_max, theta_nums)
	vertial_samples_num = a.shape[0]
	i = angle_range*np.pi/180
	R1 = np.zeros((2, 2, vertial_samples_num-1, len(angle_range)))


	for vertial_samples in range(vertial_samples_num-1):
		rho1=rho[vertial_samples]
		a01=a[vertial_samples]
		b01=b[vertial_samples]
		C33_1=a01**2*rho1
		C44_1=b01**2*rho1
		E1=E[vertial_samples]
		G1=G[vertial_samples]
		D1=D[vertial_samples]
		C11_1=2.0*C33_1*E1+C33_1
		C66_1=2.0*C44_1*G1+C44_1
		C13_1=(D1*2*C33_1*(C33_1-C44_1)+(C33_1-C44_1)**2)**0.5-C44_1

		rho2=rho[vertial_samples+1]
		a02=a[vertial_samples+1]
		b02=b[vertial_samples+1]
		C33_2=a02**2*rho2
		C44_2=b02**2*rho2
		E2=E[vertial_samples+1]
		G2=G[vertial_samples+1]
		D2=D[vertial_samples+1]
		C11_2=2.0*C33_2*E2+C33_2
		C66_2=2.0*C44_2*G2+C44_2
		C13_2=(D2*2*C33_2*(C33_2-C44_2)+(C33_2-C44_2)**2)**0.5-C44_2

		C55_1=C44_1
		C55_2=C44_2
		C55_0=(C55_1+C55_2)/2
		d_C55=C55_2-C55_1

		###################################
		DA1 = (2*C55_1 + C13_1 - C11_1)/C11_1
		DA2 = (2*C55_2 + C13_2 - C11_2)/C11_2


		SINI = np.sin(i)
		COSI = np.cos(i)
		TANI = SINI/COSI


		DthetaI1 = ((C33_1 - C44_1)**2 + 2*(2*(C13_1 + C44_1)**2 - 
				(C33_1 - C44_1)*(C11_1 + C33_1 - 2*C44_1))*SINI**2 + 
				((C11_1 + C33_1 - 2*C44_1)**2 - 4*(C13_1 + C44_1)**2)*SINI**4)**0.5

		a1 = (0.5/rho1*(C33_1 + C44_1 + (C11_1 - C33_1)*SINI**2 + DthetaI1))**0.5
		ep1 = rho1*a1**2/C11_1 - 1

		I_Np1 = 1/(((C55_1 + C13_1)*SINI*COSI/a1**2)**2 + 
				(rho1 - (C11_1*SINI**2/a1**2 + C55_1*COSI**2/a1**2))**2)**0.5
		SINIA = I_Np1*(C55_1 + C13_1)*SINI*COSI/a1**2
		Sp1 = SINIA/SINI - 1

		ia = np.arcsin(SINIA)
		COSIA = np.cos(ia)
		TANIA = SINIA/COSIA
		rp = SINI/a1

		def function_0(x):
			return x/(0.5/rho1*(C33_1 + C44_1 + (C11_1 - C33_1)*x**2 - ((C33_1 - C44_1)**2 + 
				2*(2*(C13_1 + C44_1)**2 - (C33_1 - C44_1)*(C11_1 + C33_1 - 2*C44_1))*x**2 + 
				((C11_1 + C33_1 - 2*C44_1)**2 - 4*(C13_1 + C44_1)**2)*x**4)**0.5))**0.5 - rp

		SINJ = fsolve(function_0, rp*b01)
		DthetaJ1 = ((C33_1 - C44_1)**2 + 2*(2*(C13_1 + C44_1)**2 - 
					(C33_1 - C44_1)*(C11_1 + C33_1 - 2*C44_1))*SINJ**2 + 
					((C11_1 + C33_1 - 2*C44_1)**2 - 4*(C13_1 + C44_1)**2)*SINJ**4)**0.5

		j = np.arcsin(SINJ)
		COSJ = np.cos(j)
		TANJ = SINJ/COSJ

		b1 = (0.5/rho1*(C33_1 + C44_1 + (C11_1 - C33_1)*SINJ**2 - DthetaJ1))**0.5
		es1 = rho1*b1**2/C55_1 - 1

		I_Ns1 = 1/(((C55_1 + C13_1)*SINJ*COSJ/b1**2)**2 + 
				(rho1 - (C11_1*SINJ**2/b1**2 + C55_1*COSJ**2/b1**2))**2)**0.5
		SINJA = -I_Ns1*(rho1 - (C11_1*SINJ**2/b1**2 + C55_1*COSJ**2/b1**2))
		Ss1 = SINJA/SINJ - 1

		ja = np.arcsin(SINJA)
		COSJA = np.cos(ja)
		TANJA = SINJA/COSJA

		def function_1(x):
			return x/(0.5/rho2*(C33_2 + C44_2 + (C11_2 - C33_2)*x**2 + ((C33_2 - C44_2)**2 + 
				2*(2*(C13_2 + C44_2)**2 - (C33_2 - C44_2)*(C11_2 + C33_2 - 2*C44_2))*x**2 + 
				((C11_2 + C33_2 - 2*C44_2)**2 - 4*(C13_2 + C44_2)**2)*x**4)**0.5))**0.5 - rp

		SINI2 = fsolve(function_1, rp*a02)
		DthetaI2 = ((C33_2 - C44_2)**2 + 2*(2*(C13_2 + C44_2)**2 - 
					(C33_2 - C44_2)*(C11_2 + C33_2 - 2*C44_2))*SINI2**2 + 
					((C11_2 + C33_2 - 2*C44_2)**2 - 4*(C13_2 + C44_2)**2)*SINI2**4)**0.5

		i2 = np.arcsin(SINI2)
		COSI2 = np.cos(i2)
		TANI2 = SINI2/COSI2

		a2 = (0.5/rho2*(C33_2 + C44_2 + (C11_2 - C33_2)*SINI2**2 + DthetaI2))**0.5
		ep2 = rho2*a2**2/C11_2 - 1

		I_Np2 = 1/(((C55_2 + C13_2)*SINI2*COSI2/a2**2)**2 + 
				(rho2 - (C11_2*SINI2**2/a2**2 + C55_2*COSI2**2/a2**2))**2)**0.5
		SINIA2 = I_Np2*(C55_2 + C13_2)*SINI2*COSI2/a2**2
		Sp2 = SINIA2/SINI2 - 1

		ia2 = np.arcsin(SINIA2)
		COSIA2 = np.cos(ia2)
		TANIA2 = SINIA2/COSIA2

		def function_2(x):
			return x/(0.5/rho2*(C33_2 + C44_2 + (C11_2 - C33_2)*x**2 - ((C33_2 - C44_2)**2 + 
				2*(2*(C13_2 + C44_2)**2 - (C33_2 - C44_2)*(C11_2 + C33_2 - 2*C44_2))*x**2 + 
				((C11_2 + C33_2 - 2*C44_2)**2 - 4*(C13_2 + C44_2)**2)*x**4)**0.5))**0.5 - rp

		SINJ2 = fsolve(function_2, rp*b02)
		DthetaJ2 = ((C33_2 - C44_2)**2 + 2*(2*(C13_2 + C44_2)**2 - 
					(C33_2 - C44_2)*(C11_2 + C33_2 - 2*C44_2))*SINJ2**2 + 
					((C11_2 + C33_2 - 2*C44_2)**2 - 4*(C13_2 + C44_2)**2)*SINJ2**4)**0.5

		j2 = np.arcsin(SINJ2)
		COSJ2 = np.cos(j2)
		TANJ2 = SINJ2/COSJ2

		b2 = (0.5/rho2*(C33_2 + C44_2 + (C11_2 - C33_2)*SINJ2**2 - DthetaJ2))**0.5
		es2 = rho2*b2**2/C55_2 - 1

		I_Ns2 = 1/(((C55_2 + C13_2)*SINJ2*COSJ2/b2**2)**2 + 
				(rho2 - (C11_2*SINJ2**2/b2**2 + C55_2*COSJ2**2/b2**2))**2)**0.5
		SINJA2 = -I_Ns2*(rho2 - (C11_2*SINJ2**2/b2**2 + C55_2*COSJ2**2/b2**2))
		Ss2 = SINJA2/SINJ2 - 1

		ja2 = np.arcsin(SINJA2)
		COSJA2 = np.cos(ja2)
		TANJA2 = SINJA2/COSJA2

		d_Sp = Sp2 - Sp1
		d_Ss = Ss2 - Ss1
		Sp0 = (Sp2 + Sp1)/2
		Ss0 = (Ss2 + Ss1)/2

		a0 = (a1 + a2)/2
		b0 = (b1 + b2)/2
		d_a = a2 - a1
		d_b = b2 - b1

		SINI0 = a0*rp
		SINIA0 = (1 + Sp0)*SINI0
		i0 = np.arcsin(SINI0)
		ia0 = np.arcsin(SINIA0)
		COSI0 = np.cos(i0)
		COSIA0 = np.cos(ia0)
		TANI0 = SINI0/COSI0
		TANIA0 = SINIA0/COSIA0

		SINJ0 = b0*rp
		SINJA0 = (1 + Ss0)*SINJ0
		j0 = np.arcsin(SINJ0)
		ja0 = np.arcsin(SINJA0)
		COSJ0 = np.cos(j0)
		COSJA0 = np.cos(ja0)
		TANJ0 = SINJ0/COSJ0
		TANJA0 = SINJA0/COSJA0

		rhoA1 = 1/(2*b01**2)*(C33_1 - C13_1)
		rhoA2 = 1/(2*b02**2)*(C33_2 - C13_2)
		d_rhoA = rhoA2 - rhoA1
		rhoA0 = 0.5*(rhoA1 + rhoA2)

		d_a = a2 - a1
		d_b = b2 - b1
		a0 = (a1 + a2)/2
		b0 = (b1 + b2)/2

		ep1_1 = SINIA
		ep3_1 = COSIA
		es1_1 = COSJA
		es3_1 = -SINJA

		ep1_2 = SINIA2
		ep3_2 = COSIA2
		es1_2 = COSJA2
		es3_2 = -SINJA2

		X1 = np.array([[ep1_1, es1_1],
			[(C13_1*SINI/a1*ep1_1 + C33_1*COSI/a1*ep3_1), (C13_1*SINI/a1*es1_1 + C33_1*COSJ/b1*es3_1)]])

		X2 = np.array([[ep1_2, es1_2],
			[(C13_2*SINI2/a2*ep1_2 + C33_2*COSI2/a2*ep3_2), (C13_2*SINI2/a2*es1_2 + C33_2*COSJ2/b2*es3_2)]])

		Y1 = np.array([[C55_1*(SINI/a1*ep3_1 + COSI/a1*ep1_1), C55_1*(SINI/a1*es3_1 + COSJ/b1*es1_1)],
			[ep3_1, es3_1]])

		Y2 = np.array([[C55_2*(SINI2/a2*ep3_2 + COSI2/a2*ep1_2), C55_2*(SINI2/a2*es3_2 + COSJ2/b2*es1_2)],
			[ep3_2, es3_2]])

		d_X = X2 - X1
		d_Y = Y2 - Y1

		for theta in range(len(angle_range)):
			X1_t = X1[:, :, theta]
			X2_t = X2[:, :, theta]
			Y1_t = Y1[:, :, theta]
			Y2_t = Y2[:, :, theta]
			R1[:, :, vertial_samples, theta] = (linalg.inv(X1_t)*X2_t - 
							linalg.inv(Y1_t)*Y2_t)**1*linalg.inv(linalg.inv(X1_t)*X2_t + 
							linalg.inv(Y1_t)*Y2_t)
		print("vertial samples number: %d" % vertial_samples)
	return R1[0,0]