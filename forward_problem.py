# -*- coding: utf-8 -*-
"""
Created on Wed Sep 19 14:12:18 2018

ISO: please enter "https://wiki.seg.org/wiki/" to get more informations. 
ANI: Plane-wave reflection and transmission coefficients for a transversely isotropic solid Mark Graebner
ANI: REFLECTION COEFFICIENTS AND AZIMUTHAL AVO ANALYSIS IN ANISOTROPIC MEDIA By Andreas Ru¨ger

@author: 泳浩
"""
import numpy as np


def main():
    pass


if __name__=="__main__":
    main()
    

class Normal(object):
    '''
    Normal expression
    '''
    def __init__(self, V, rho):
    # def __init__(self, Vp, Vs, rho):
        # Zp = data["Vp"]*data["Rho"]
        # Zs = data["Vs"]*data["Rho"]
        # self.Zp_1, self.Zp_2 = np.array(Zp_1[:-1]), np.array(Zp_1[1:])
        # self.Zs_1, self.Zs_2 = np.array(Zs_1[:-1]), np.array(Zs_1[1:])
        Z = V*(np.array(rho).reshape(-1, 1))
        self.Z_1, self.Z_2 = np.array(Z[:-1]), np.array(Z[1:])
    # def reflection_p(self):
    #     reflect = (self.Zp_2-self.Zp_1)/(self.Zp_2+self.Zp_1)
    # def reflection_p(self):
    #     reflect = (self.Zs_2-self.Zs_1)/(self.Zs_2+self.Zs_1)
    
    def reflection_p(self):
        return (self.Z_2-self.Z_1)/(self.Z_2+self.Z_1)


class Zoeppritz(object):
    '''
    Zoeppritz exact
    '''
    def __init__(self, Vp, Vs, rho, theta):
        Vp1, Vp2 = np.array(Vp[:-1]), np.array(Vp[1:])
        Vs1, Vs2 = np.array(Vs[:-1]), np.array(Vs[1:])
        rho1, rho2 = np.array(rho[:-1]), np.array(rho[1:])
        theta1 = np.array(theta[:-1])
        Vp1, Vp2, Vs1, Vs2, rho1, rho2 = Vp1.reshape(-1, 1), Vp2.reshape(-1, 1), Vs1.reshape(-1, 1), Vs2.reshape(-1, 1), rho1.reshape(-1, 1), rho2.reshape(-1, 1)
        snell = np.sin(theta1)/Vp1
        theta2 = np.arcsin(Vp2*snell)
        phi1 = np.arcsin(Vs1*snell)
        phi2 = np.arcsin(Vs2*snell)
        M = np.zeros((4, 4, theta1.shape[0], theta1.shape[1]))
        # vector = np.zeros((4, 1))
        vector = np.zeros((4, theta1.shape[0], theta1.shape[1]))
        # Matrix Vector Zoeppritz
        Z_1, Z_2 = rho1*Vp1, rho2*Vp2
        W_1, W_2 = rho1*Vs1, rho2*Vs2
        M[0, 0], M[0, 1], M[0, 2], M[0, 3] = np.cos(theta1), -np.sin(phi1), np.cos(theta2), np.sin(phi2)
        M[1, 0], M[1, 1], M[1, 2], M[1, 3] = np.sin(theta1), np.cos(phi1), -np.sin(theta2), np.cos(phi2)
        M[2, 0], M[2, 1] = Z_1*np.cos(2*phi1), -W_1*np.sin(2*phi1)
        M[2, 2], M[2, 3] = -Z_2*np.cos(2*phi2), -W_2*np.sin(2*phi2)
        M[3, 0], M[3, 1] = (Vs1/Vp1)*W_1*np.sin(2*theta1), W_1*np.cos(2*phi1) 
        M[3, 2], M[3, 3] = (Vs2/Vp2)*W_2*np.sin(2*theta2), -W_2*np.cos(2*phi2)
        vector[0], vector[1], vector[2], vector[3] = np.cos(theta1), -np.sin(theta1), -Z_1*np.cos(2*phi1), (Vs1/Vp1)*W_1*np.sin(2*theta1)
        self.M = M
        self.V = vector
        self.theta1 = theta1

    def zoeppritz_exact(self, time_windows):
        t = time_windows
        V = self.V[:, t, :]
        M = self.M[:, :, t, :]
        R = np.zeros((4, self.theta1.shape[1]))
        for i in range(self.theta1.shape[1]):
            m = M[:, :, i]
            v = V[:, i].reshape(-1, 1)
            # R[:, i] = np.dot(np.linalg.inv(m), v).flatten()
            R[:, i] = np.linalg.solve(m, v).flatten()
        return R

    def zoeppritz_exact_all(self):
        V = self.V[:, :, :]
        M = self.M[:, :, :, :]
        R = np.zeros((4, self.theta1.shape[0], self.theta1.shape[1]))
        for j in range(self.theta1.shape[0]):
            for i in range(self.theta1.shape[1]):
                m = M[:, :, j, i]
                v = V[:, j, i].reshape(-1, 1)
                # R[:, j, i] = np.dot(np.linalg.inv(m), v).flatten()
                R[:, j, i] = np.linalg.solve(m, v).flatten()
        return R


class Aki_Richard(object):
    '''
    Aki-richard terms 3 expression
    '''
    def __init__(self, Vp, Vs, rho, theta):
        Vp1, Vp2 = np.array(Vp[:-1]), np.array(Vp[1:])
        Vs1, Vs2 = np.array(Vs[:-1]), np.array(Vs[1:])
        rho1, rho2 = np.array(rho[:-1]), np.array(rho[1:])
        theta1, theta2 = np.array(theta[:-1]), np.array(theta[1:])
        self.Vp = (Vp1+Vp2)/2
        self.Vs = (Vs1+Vs2)/2
        self.dVp = Vp2-Vp1
        self.dVs = Vs2-Vs1
        self.rho = (rho1+rho2)/2
        self.drho = rho2-rho1
        self.theta = (theta1+theta2)/2
        
    # def wiggens(self):
    #     Rp = (self.dVp/self.Vp + self.drho/self.rho)/2
    #     G = self.dVp/self.Vp/2 - 4*(self.Vs/self.Vp)**2*self.dVs/self.Vs - 2*(self.Vs/self.Vp)**2*self.drho/self.rho
    #     C = self.dVp/self.Vp/2
    #     Rp, G, C = Rp.reshape(-1, 1), G.reshape(-1, 1), C.reshape(-1, 1)
    #     R = Rp + G*np.sin(self.theta) + C*(np.tan(self.theta)*np.sin(self.theta))**2
    #     return R 

    def aki_richards(self):
        A = (1/2*(1-4*((self.Vs/self.Vp).reshape(-1, 1)*np.sin(self.theta))**2)*(self.drho/self.rho).reshape(-1, 1))
        B = 1/(2*np.cos(self.theta)**2)*(self.dVp/self.Vp).reshape(-1, 1)
        C = (4*((self.Vs/self.Vp).reshape(-1, 1)*np.sin(self.theta))**2*(self.dVs/self.Vs).reshape(-1, 1))
        R = A+B-C
        return R
    

class Thomsen(object):
    '''
    VTI
    Thomsen & Ruger approximate expresion
    '''
    def __init__(self, Vp, Vs, rho, theta, epsilon, delta, gamma):
        Vp1, Vp2 = np.array(Vp[:-1]), np.array(Vp[1:])
        Vs1, Vs2 = np.array(Vs[:-1]), np.array(Vs[1:])
        rho1, rho2 = np.array(rho[:-1]), np.array(rho[1:])
        theta1, theta2 = np.array(theta[:-1]), np.array(theta[1:])
        delta1, delta2 = np.array(delta[:-1]), np.array(delta[1:])
        epsilon1, epsilon2 = np.array(epsilon[:-1]), np.array(epsilon[1:])
        gamma1, gamma2 = np.array(gamma[:-1]), np.array(gamma[1:])
        self.Vp0 = Vp
        self.Vs0 = Vs
        self.rho0 = rho
        self.epsilon0 = epsilon
        self.eta0 = epsilon-delta
        self.Vp1 = Vp[:-1]
        self.Vs1 = Vs[:-1]
        self.rho1 = rho[:-1]
        self.epsilon1 = epsilon[:-1]
        self.eta1 = (epsilon-delta)[:-1]
        self.Vp = (Vp1+Vp2)/2
        self.Vs = (Vs1+Vs2)/2
        self.dVp = Vp2-Vp1
        self.dVs = Vs2-Vs1
        self.rho = (rho1+rho2)/2
        self.drho = rho2-rho1
        self.theta = (theta1+theta2)/2
        self.ddelta = delta2 - delta1
        self.depsilon = epsilon2 - epsilon1
        self.dgamma = gamma2 - gamma1
        self.dZ = rho2*Vp2-rho1*Vp1
        self.Z = (rho2*Vp2+rho1*Vp1)/2
        self.dG = rho2*Vs2**2-rho1*Vs1**2
        self.G = (rho2*Vs2**2+rho1*Vs1**2)/2

    def ruger_approximate(self): #Ruger
        K = (2*self.Vs/self.Vp)**2
        A = (self.dZ/self.Z)
        B = (self.dVp/self.Vp-K*(self.dG/self.G-2*self.dgamma)+self.ddelta)
        C = (self.dVp/self.Vp+self.depsilon)
        A, B, C = A.reshape(-1, 1), B.reshape(-1, 1), C.reshape(-1, 1)
        D, E, F = 1/2, np.sin(self.theta)**2/2, (np.tan(self.theta)*np.sin(self.theta))**2/2
        Rp_vti = A*D + B*E + C*F
        return Rp_vti

    def iso_approximate(self): #
        k = (self.Vs/self.Vp).reshape(-1, 1)
        A, B, C = self.dVp/self.Vp/2, self.dVs/self.Vs/2, self.drho/self.rho/2 # 注意这个地方除了２
        D, E, F = self.ddelta/2, self.depsilon, self.dgamma/2
        A, B, C = A.reshape(-1, 1), B.reshape(-1, 1), C.reshape(-1, 1)
        D, E, F = D.reshape(-1, 1), E.reshape(-1, 1), F.reshape(-1, 1)
        # 表达方式 1
        # Rpp_iso = (1+np.tan(self.theta)**2)*A-8*k**2*np.sin(self.theta)**2*B+(1-4*k**2*np.sin(self.theta)**2)*C #gidlow
        # Rpp_ani = np.sin(self.theta)**2*D+(np.sin(self.theta)*np.tan(self.theta))**2*E+8*(k*np.sin(self.theta))**2*F
        # 表达方式 2 （都一样）
        # G, H, I = (1+np.tan(self.theta)**2), -8*k**2*np.sin(self.theta)**2, (4*k**2*np.sin(self.theta)**2-np.tan(self.theta)**2)
        G, H, I = (1+np.tan(self.theta)**2), -8*k**2*np.sin(self.theta)**2, (1-4*k**2*np.sin(self.theta)**2)
        J, K, L = np.sin(self.theta)**2, (np.sin(self.theta)*np.tan(self.theta))**2, 8*(k*np.sin(self.theta))**2
        # 最终式子结合
        Rpp_iso = G*A+H*B+I*C
        Rpp_ani = J*D+K*E+L*F
        Rpp_vti = Rpp_iso+Rpp_ani
        return [Rpp_iso, Rpp_vti]

    def iso_ani_paras(self, k): #
        """
        k = Vs/Vp
        """
        A, B, C = self.dVp/self.Vp, self.dVs/self.Vs, self.drho/self.rho # 注意这个地方没有除了２
        D, E, F = self.ddelta/2, self.depsilon, self.dgamma/2
        A, B, C = A.reshape(-1, 1), B.reshape(-1, 1), C.reshape(-1, 1)
        D, E, F = D.reshape(-1, 1), E.reshape(-1, 1), F.reshape(-1, 1)
        # G, H, I = (1+np.tan(self.theta)**2), -8*k**2*np.sin(self.theta)**2, (4*k**2*np.sin(self.theta)**2-np.tan(self.theta)**2)
        G, H, I = (1+np.tan(self.theta)**2)/2, -8*k**2*np.sin(self.theta)**2/2, (1-4*k**2*np.sin(self.theta)**2)/2
        J, K, L = np.sin(self.theta)**2, (np.sin(self.theta)*np.tan(self.theta))**2, 8*(k*np.sin(self.theta))**2
        Rpp_iso = G*A+H*B+I*C
        Rpp_ani = J*D+K*E+L*F
        Rpp_vti = Rpp_iso+Rpp_ani
        return [A, B, C, G, H, I, Rpp_vti]

    # def ani_zhang(self, k):
    #     """
    #     k = (2Vs/Vp)**2
    #     """
    #     # k = (2*self.Vs.mean()/self.Vp.mean())**2
    #     A, B, C = np.array(self.rho0*self.Vp0), np.array(self.rho0*self.Vs0**2*np.exp(self.eta0/k)), np.array(self.Vp0*np.exp(self.epsilon0))
    #     # A, B, C = A[top:bottom], B[top:bottom], C[top:bottom]
    #     A, B, C = np.log(A), np.log(B), np.log(C)
    #     dlnA = A[1:]-A[:-1]
    #     dlnB = B[1:]-B[:-1]
    #     dlnC = C[1:]-C[:-1]
    #     D, E, F = 1/2*np.ones((self.theta.shape)), -k/2*np.sin(self.theta)**2, np.tan(self.theta)**2/2
    #     dlnA, dlnB, dlnC = dlnA.reshape(-1, 1), dlnB.reshape(-1, 1), dlnC.reshape(-1, 1) 
    #     Rpp_vti = dlnA*D+dlnB*E+dlnC*F
    #     return [dlnA, dlnB, dlnC, D, E, F, Rpp_vti]

    def ani_zhang(self):
        """
        k = (2Vs/Vp)**2
        """
        # k = (2*self.Vs.mean()/self.Vp.mean())**2
        # k = np.array((2*self.Vs0/self.Vp0)**2)
        k = np.array((2*self.Vs/self.Vp)**2)
        # k = ((2*self.Vs/self.Vp)**2)
        k = np.insert(k, k.shape[0], 0)
        A, B, C = np.array(self.rho0*self.Vp0), np.array(self.rho0*self.Vs0**2*np.exp(self.eta0/k)), np.array(self.Vp0*np.exp(self.epsilon0))
        # A, B, C = A[top:bottom], B[top:bottom], C[top:bottom]
        A, B, C = np.log(A), np.log(B), np.log(C)
        dlnA = A[1:]-A[:-1]
        dlnB = B[1:]-B[:-1]
        dlnC = C[1:]-C[:-1]
        k = k[:-1]
        # k = k[1:]
        k = k.reshape(-1, 1)
        D, E, F = 1/2*np.ones((self.theta.shape)), -k/2*np.sin(self.theta)**2, np.tan(self.theta)**2/2
        dlnA, dlnB, dlnC = dlnA.reshape(-1, 1), dlnB.reshape(-1, 1), dlnC.reshape(-1, 1) 
        Rpp_vti = dlnA*D+dlnB*E+dlnC*F
        return [dlnA, dlnB, dlnC, D, E, F, Rpp_vti]

    def ani_zhang_5(self):
        """
        k = (2Vs/Vp)**2
        """
        A = (1/2*(1-4*((self.Vs/self.Vp).reshape(-1, 1)*np.sin(self.theta))**2)*(self.drho/self.rho).reshape(-1, 1))
        B = 1/(2*np.cos(self.theta)**2)*(self.dVp/self.Vp).reshape(-1, 1)
        C = -(4*((self.Vs/self.Vp).reshape(-1, 1)*np.sin(self.theta))**2*(self.dVs/self.Vs).reshape(-1, 1))
        D = (self.ddelta).reshape(-1, 1)*np.sin(self.theta)**4
        E = (self.depsilon).reshape(-1, 1)*np.sin(self.theta)**4*np.tan(self.theta)**2/2
        R = A+B+C+D+E
        return R                

    def expectation(self): # not include theta
        Rp, Rs, Rr = self.dZ/self.Z, self.dG/self.G, self.drho/self.rho
        return Rp, Rs, Rr


class Weak_Anisotropy(Thomsen):

    def __init__(self, Vp, Vs, rho, theta, epsilon, delta, gamma):
        super(Weak_Anisotropy, self).__init__(Vp, Vs, rho, theta, epsilon, delta, gamma) 
        self.Vp = np.array(Vp)
        self.Vs = np.array(Vs)
        self.rho = np.array(rho)
        self.theta = np.array(theta)
        self.epsilon = np.array(epsilon)
        self.delta = np.array(delta)
        self.gamma = np.array(gamma)

    #Thomsen approximate phase velocity
    def weakVp_phase(self):
        return self.Vp.reshape(-1, 1)*(1+self.delta.reshape(-1, 1)*(np.cos(self.theta)*np.sin(self.theta))**2+self.epsilon.reshape(-1, 1)*np.sin(self.theta)**4)
   
    def weakVsv_phase(self):
        return self.Vs.reshape(-1, 1)*(1+(self.epsilon.reshape(-1, 1)-self.delta.reshape(-1, 1))*(self.Vp.reshape(-1, 1)/self.Vs.reshape(-1, 1)*np.sin(self.theta)*np.cos(self.theta))**2)
    
    def weakVsh_phase(self):
        return self.Vs.reshape(-1, 1)*(1+self.gamma.reshape(-1, 1)*np.sin(self.theta)**2)


class Darey_Hron_init(object):
    '''
    VTI
    Darey_Hron accuratly phase velocity Vp Vsv Vsh
    '''
    def __init__(self, theta, rho, C_11, C_13, C_33, C_44, C_66):
        D = np.sqrt((C_33-C_44)**2+2*np.sin(theta)**2*(2*(C_13+C_44)**2-(C_33-C_44)*(C_11+C_33-2*C_44))
            +((C_11+C_33-2*C_44)**2-4*(C_13+C_44)**2)*np.sin(theta)**4)
        self.Vp = np.sqrt((C_33+C_44+(C_11-C_33)*np.sin(theta)**2+D)/2/rho)
        self.Vsv = np.sqrt((C_33+C_44+(C_11-C_33)*np.sin(theta)**2-D)/2/rho)
        self.Vsh = np.sqrt((C_66*np.sin(theta)**2+C_44*np.cos(theta)**2)/rho)

    def DH_Vp_phase(self):
        return self.Vp

    def DH_Vsv_phase(self):
        return self.Vsv

    def DH_Vsh_phase(self):
        return self.Vsh


class Darey_Hron_trans(Thomsen):
    '''
    VTI
    Darey_Hron accuratly phase velocity Vp Vsv Vsh
    '''
    def __init__(self, Vp, Vs, rho, theta, epsilon, delta, gamma):
        super(Darey_Hron_trans, self).__init__(Vp, Vs, rho, theta, epsilon, delta, gamma)
        f = 1-(Vs/Vp)**2
        C_11 = np.array(rho*(1+2*epsilon)*Vp**2).reshape(-1, 1)
        C_22 = np.array(rho*(1+2*epsilon)*Vp**2).reshape(-1, 1)
        C_33 = np.array(rho*Vp**2).reshape(-1, 1)
        C_44 = np.array(rho*Vs**2).reshape(-1, 1)
        C_55 = np.array(C_44).reshape(-1, 1)
        C_66 = np.array(rho*(1+2*gamma)*Vs**2).reshape(-1, 1)
        C_12 = np.array(rho*Vp**2*(1+2*epsilon-2*(1-f)*(1+2*gamma))).reshape(-1, 1)
        C_13 = np.array(rho*Vp**2*np.sqrt(f*(f+2*delta))-rho*Vs**2).reshape(-1, 1)
        theta = np.array(theta).reshape(1, -1)
        rho = np.array(rho).reshape(-1, 1)
        D = np.sqrt((C_33-C_44)**2+2*np.sin(theta)**2*(2*(C_13+C_44)**2-(C_33-C_44)*(C_11+C_33-2*C_44))\
            +((C_11+C_33-2*C_44)**2-4*(C_13+C_44)**2)*np.sin(theta)**4)
        self.Vp = np.sqrt((C_33+C_44+(C_11-C_33)*np.sin(theta)**2+D)/2/rho)
        self.Vsv = np.sqrt((C_33+C_44+(C_11-C_33)*np.sin(theta)**2-D)/2/rho)
        self.Vsh = np.sqrt((C_66*np.sin(theta)**2+C_44*np.cos(theta)**2)/rho)

    def DH_Vp_phase(self):
        return self.Vp

    def DH_Vsv_phase(self):
        return self.Vsv

    def DH_Vsh_phase(self):
        return self.Vsh


# class Graebner_Ruger(object):
#     """docstring for Graebner_Ruger wrong"""
#     def __init__(self, Vp, Vs, rho, theta, epsilon, delta, gamma):
#         Vp = np.array(Vp).reshape(-1, 1)
#         Vs = np.array(Vs).reshape(-1, 1)
#         rho = np.array(rho).reshape(-1, 1)
#         epsilon = np.array(epsilon).reshape(-1, 1)
#         delta = np.array(delta).reshape(-1, 1)
#         gamma = np.array(gamma).reshape(-1, 1)
#         C_33 = np.array(rho*Vp**2).reshape(-1, 1)
#         C_55 = np.array(rho*Vs**2).reshape(-1, 1)
#         C_44 = C_55
#         C_11 = np.array((1 + 2*epsilon)*C_33).reshape(-1, 1)
#         C_66 = np.array((1 + 2*gamma)*C_33).reshape(-1, 1)
#         C_13 = np.sqrt(2*C_33*(C_33 - C_55)*delta + (C_33 - C_55)**2) - C_55
#         p, q = np.sin(theta)/Vp, np.cos(theta)/Vp
#         theta1 = np.array(theta[:-1])
#         K_1 = rho/C_33 + rho/C_55 - (C_11/C_55 + C_55/C_33 - (C_13 + C_55)**2/C_33/C_55)*p**2
#         K_2 = C_11/C_33*p**2 - rho/C_33
#         K_3 = p**2 - rho/C_55
#         q_alpha = np.sqrt(K_1 - np.sqrt(K_1**2 - 4*K_2*K_3))/np.sqrt(2)
#         q_beta = np.sqrt(K_1 + np.sqrt(K_1**2 - 4*K_2*K_3))/np.sqrt(2)
#         a_11, a_13, a_33, a_55 = C_11/rho, C_13/rho, C_33/rho, C_55/rho
#         l_alpha = np.sqrt((a_33*q_alpha**2 + a_55*p**2 - 1)/
#             (a_11*p**2 + a_55*q_alpha**2 - 1 + a_33*q_alpha**2 + a_55*p**2 - 1))
#         m_alpha = np.sqrt((a_11*p**2 + a_55*q_alpha**2 - 1)/
#             (a_11*p**2 + a_55*q_alpha**2 - 1 + a_33*q_alpha**2 + a_55*p**2 - 1))
#         l_beta = np.sqrt((a_11*p**2 + a_55*q_beta**2 - 1)/
#             (a_11*p**2 + a_55*q_beta**2 - 1 + a_33*q_beta**2 + a_55*p**2 - 1))
#         m_beta = np.sqrt((a_33*q_beta**2 + a_55*p**2 - 1)/
#             (a_11*p**2 + a_55*q_beta**2 - 1 + a_33*q_beta**2 + a_55*p**2 - 1))
#         M = np.zeros((4, 4, theta1.shape[0], theta1.shape[1]))
#         vector = np.zeros((4, theta1.shape[0], theta1.shape[1]))
#         M[0, 0], M[0, 1], M[0, 2], M[0, 3] = l_alpha[:-1], m_beta[:-1], -l_alpha[1:], -m_beta[1:]
#         M[2, 0], M[2, 1], M[2, 2], M[2, 3] = m_alpha[:-1], -l_beta[:-1], m_alpha[1:], -l_beta[1:]
#         M[1, 0] = p[:-1]*l_alpha[:-1]*a_13[:-1] + q_alpha[:-1]*m_alpha[:-1]*a_33[:-1]
#         M[1, 1] = p[:-1]*m_beta[:-1]*a_13[:-1] - q_beta[:-1]*l_beta[:-1]*a_33[:-1]
#         M[1, 2] = -(p[1:]*l_alpha[1:]*a_13[1:] + q_alpha[1:]*m_alpha[1:]*a_33[1:])
#         M[1, 3] = -(p[1:]*m_beta[1:]*a_13[1:] - q_beta[1:]*l_beta[1:]*a_33[1:])
#         M[3, 0] = a_55[:-1]*(q_alpha[:-1]*l_alpha[:-1] + p[:-1]*m_alpha[:-1])
#         M[3, 1] = a_55[:-1]*(q_beta[:-1]*m_beta[:-1] - p[:-1]*l_beta[:-1])
#         M[3, 2] = a_55[1:]*(q_alpha[1:]*l_alpha[1:] + p[1:]*m_alpha[1:])
#         M[3, 3] = a_55[1:]*(q_beta[1:]*m_beta[1:] - p[1:]*l_beta[1:])
#         vector[0], vector[1], vector[2], vector[3] = -M[0, 0], -M[1, 0], M[2, 0], M[3, 0]
#         self.M = M
#         self.V = vector
#         self.theta1 = theta1

#     def vti_exact(self, time_windows):
#         t = time_windows
#         V = self.V[:, t, :]
#         M = self.M[:, :, t, :]
#         R = np.zeros((4, self.theta1.shape[1]))
#         for i in range(self.theta1.shape[1]):
#             m = M[:, :, i]
#             v = V[:, i].reshape(-1, 1)
#             R[:, i] = np.dot(np.linalg.inv(m), v).flatten()
#         return R

#     def vti_exact_all(self):
#         V = self.V[:, :, :]
#         M = self.M[:, :, :, :]
#         R = np.zeros((4, self.theta1.shape[0], self.theta1.shape[1]))
#         for j in range(self.theta1.shape[0]):
#             for i in range(self.theta1.shape[1]):
#                 m = M[:, :, j, i]
#                 det_m = np.linalg.det(m)
#                 v = V[:, j, i].reshape(-1, 1)
#                 R[:, j, i] = (np.dot(m.T, v)/det_m).flatten()
#         return R


class Graebner(object):
    """docstring for Graebner"""
    def __init__(self, Vp, Vs, rho, theta, epsilon, delta, gamma):
        Vp = np.array(Vp).reshape(-1, 1)
        Vs = np.array(Vs).reshape(-1, 1)
        rho = np.array(rho).reshape(-1, 1)
        epsilon = np.array(epsilon).reshape(-1, 1)
        delta = np.array(delta).reshape(-1, 1)
        gamma = np.array(gamma).reshape(-1, 1)
        theta1 = np.array(theta[:-1])

        A = rho*Vp**2*(1 + 2*epsilon)
        C = rho*Vp**2
        L = rho*Vs**2
        F = np.sqrt(2*C*(C - L)*delta + (C - L)**2) - L
        Z = (1 + 2*gamma)*C

        p, q = np.sin(theta)/Vp, np.cos(theta)/Vp

        K_1 = rho/C + rho/L - (A/L + L/C - (F + L)**2/C/L)*p**2
        K_2 = A/C*p**2 - rho/C
        K_3 = p**2 - rho/L

        q_alpha = np.sqrt(K_1 - np.sqrt(K_1**2 - 4*K_2*K_3))/np.sqrt(2)
        q_beta = np.sqrt(K_1 + np.sqrt(K_1**2 - 4*K_2*K_3))/np.sqrt(2)

        A_, F_, C_, L_ = A/rho, F/rho, C/rho, L/rho

        l_alpha = np.sqrt((C_*q_alpha**2 + L_*p**2 - 1)/
            (A_*p**2 + L_*q_alpha**2 - 1 + C_*q_alpha**2 + L_*p**2 - 1))

        m_alpha = np.sqrt((A_*p**2 + L_*q_alpha**2 - 1)/
            (A_*p**2 + L_*q_alpha**2 - 1 + C_*q_alpha**2 + L_*p**2 - 1))

        l_beta = np.sqrt((A_*p**2 + L_*q_beta**2 - 1)/
            (A_*p**2 + L_*q_beta**2 - 1 + C_*q_beta**2 + L_*p**2 - 1))

        m_beta = np.sqrt((C_*q_beta**2 + L_*p**2 - 1)/
            (A_*p**2 + L_*q_beta**2 - 1 + C_*q_beta**2 + L_*p**2 - 1))

        a = L*(q_alpha*l_alpha + p*m_alpha)
        b = L*(q_beta*m_beta - p*l_beta)
        c = p*l_alpha*F + q_alpha*m_alpha*C
        d = p*m_beta*F - q_beta*l_beta*C

        M = np.zeros((4, 4, theta1.shape[0], theta1.shape[1]))
        vector = np.zeros((4, theta1.shape[0], theta1.shape[1]))

        M[0, 0], M[0, 1], M[0, 2], M[0, 3] = l_alpha[:-1], m_beta[:-1], -l_alpha[1:], -m_beta[1:]
        M[1, 0], M[1, 1], M[1, 2], M[1, 3] = m_alpha[:-1], -l_beta[:-1], m_alpha[1:], -l_beta[1:]
        M[2, 0], M[2, 1], M[2, 2], M[2, 3] = a[:-1], b[:-1], a[1:], b[1:]
        M[3, 0], M[3, 1], M[3, 2], M[3, 3] = c[:-1], d[:-1], -c[1:], -d[1:]
        vector[0], vector[1] = -l_alpha[:-1], m_alpha[:-1]
        vector[2] = L[:-1]*(q_alpha[:-1]*l_alpha[:-1] + p[:-1]*m_alpha[:-1])
        vector[3] = -(p[:-1]*l_alpha[:-1]*F[:-1] + q_alpha[:-1]*m_alpha[:-1]*C[:-1])
        self.M = M
        self.V = vector
        self.theta1 = theta1

    def vti_exact(self, time_windows):
        t = time_windows
        V = self.V[:, t, :]
        M = self.M[:, :, t, :]
        R = np.zeros((4, self.theta1.shape[1]))
        for i in range(self.theta1.shape[1]):
            m = M[:, :, i]
            v = V[:, i].reshape(-1, 1)
            # R[:, i] = np.dot(np.linalg.inv(m), v).flatten()
            R[:, i] = np.linalg.solve(m, v).flatten()
        return R

    def vti_exact_all(self):
        V = self.V[:, :, :]
        M = self.M[:, :, :, :]
        R = np.zeros((4, self.theta1.shape[0], self.theta1.shape[1]))
        for j in range(self.theta1.shape[0]):
            for i in range(self.theta1.shape[1]):
                m = M[:, :, j, i]
                v = V[:, j, i].reshape(-1, 1)
                # R[:, j, i] = np.dot(np.linalg.inv(m), v).flatten()
                R[:, j, i] = np.linalg.solve(m, v).flatten()
        return R
