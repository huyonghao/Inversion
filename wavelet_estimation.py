import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
# from bruges.filters import ricker, rotate_phase
from scipy import linalg as la
from numpy.linalg import lstsq
import forward_problem as forward
from sklearn import linear_model

# # # #读取测井数据
# data = pd.DataFrame(pd.read_csv("D:\\Physic_Model_Data\\line1_05.csv"))
# del data["Unnamed: 0"]

# #地震子波 （不同角度）
# theta_max, theta_space, Fmax, t_samples, dt_space = 45, 1, 30, np.arange(-50, 50, 1), 0.002 #采样间隔2ms,采样点数100
# ricker = (1-2*(np.pi*Fmax*t_samples*dt_space)**2)*np.exp(-(np.pi*Fmax*t_samples*dt_space)**2)
# # theta = np.arange(0, theta_max, theta_space)*np.pi/180  #弧度
# theta = np.linspace(0, theta_max, 15)*np.pi/180  #弧度
# theta = theta + np.zeros((data.shape[0], len(theta)))
# #theta = 90 * np.ones((data.shape[0], 1)).flatten()

# # # 3 Ruger　横向各向同性　
# # vti_appr = forward.Thomsen(data["Vp"], data["Vs"], data["Rho"], theta, data["Epsilon"], data["Delta"], data["Gamma"])
# # reflect_3 = vti_appr.ruger_approximate()
# # reflect_3[np.isnan(reflect_3)] = 0
# # np.save("reflect_ruger.npy", reflect_3)

# # 7 Thomsen 横向各向同性
# vti_appr = forward.Thomsen(data["Vp"], data["Vs"], data["Rho"], theta, data["Epsilon"], data["Delta"], data["Gamma"])
# reflect_7 = vti_appr.iso_approximate()
# reflect_7[np.isnan(reflect_7)] = 0
# np.save("reflect_ruger.npy", reflect_7)

r = np.load("reflect_ruger.npy") # 反射系数
s = np.load("7260_7275_cdp888.npy") # 角道集

dt = 0.002  # sample rate in seconds
tar_the = 5 # 提取其中一阁角度
r = (r[:, tar_the]).flatten()
s = (s[:-2, tar_the]).flatten() # 由于反射系数采样点与角道集不匹配故裁剪
t = np.arange(s.size) * dt # 采集时间
print(s.size)

fig01 = plt.figure(figsize=(3,8))

# Reflectivity track
ax = fig01.add_subplot(121)
ax.plot(r, t, 'k')
ax.invert_yaxis()
ax.set_xticks([])
# ax.set_xlim(-0.5,0.5)
# ax.set_ylim(1.5,0)
ax.set_ylabel('two-way time (s)')

# Seismic track
ax2 = fig01.add_subplot(122)
ax2.plot(s, t, 'k')
ax2.invert_yaxis()
ax2.fill_betweenx(t, s, 0, s > 0, color='k', alpha=1.0)
# ax2.set_ylim(1.5,0)
# ax2.set_xlim(-0.25,0.25)
ax2.set_xticks([])
ax2.set_yticks([])

# # Seismic track2
# ax2 = fig01.add_subplot(133)
# ax2.plot(s, t, 'k')
# ax2.invert_yaxis()
# ax2.fill_betweenx(t, s, 0, s > 0, color='k', alpha=1.0)
# # ax2.set_ylim(1.5,0)
# # ax2.set_xlim(-0.25,0.25)
# ax2.set_xticks([])
# ax2.set_yticks([])
plt.show()

# freqs = [5, 80, 130, 160]
# c = 1.0
# points = c*np.array([-50,-5,-5,-50])

# amp_spec = np.abs(np.fft.rfft(s))
# f = np.fft.rfftfreq(len(s), d=dt)
# P = 20 * np.log10(amp_spec)   #Power in Decibel Scale

# fig02 = plt.figure()
# ax = fig02.add_subplot(111)
# ax.plot(f, P,'m')
# # ax.plot(freqs, points, 'ko-', lw=2, zorder=2, ms=6)
# # for fr in freqs:
# #     ax.text(fr,1.20*points[0], fr, ha='center', va='top', fontsize=15)
# ax.set_xlabel('frequency (Hz)', fontsize=14)
# ax.set_ylabel('power (dB)', fontsize=14)

# # Uncomment this next line if you want to save the figure
# # fig02.savefig('figure_1.png', dpi=500)
# # plt.show()

# phase_spec = np.angle(np.fft.rfft(s))
# plt.plot(f, phase_spec, 'm')
# plt.yticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi],
#            [r'$-\pi$', r'$-\pi/2$', r'$0$', r'$+\pi/2$', r'$+\pi$'])
# plt.xlabel('frequency (Hz)', fontsize=14)
# plt.ylabel('phase', fontsize=14)

# ###
# ###Wavelet estimation by autocorrelation
# dw = 64 # number of samples to display on either side of zero
# acorr = np.correlate(s, s, mode='same')
# w1 = acorr[len(s)//2-dw//2:len(s)//2+dw//2]

# def norm(data):
#     return data/np.amax(data)

# fig04 =plt.figure(figsize=(8,3))
# ax = fig04.add_axes([0.1, 0.15, 0.8, 0.7])
# ax.plot(tw, orms, 'blue', lw=2, alpha=0.75, 
#          label='Ormsby (%i-%i-%i-%i Hz)' %(freqs[0],freqs[1],freqs[2],freqs[3]))
# ax.plot(np.arange(0,len(w1))*dt-0.064, norm(w1), 'k', lw=2, alpha=0.75,
#          label='Autocorrelation')
# ax.legend(loc=1)
# ax.set_xlim(-0.064,0.064)
# ax.set_xlabel('time (s)', fontsize=14)
# ax.set_ylabel('amplitude', fontsize=14)
# ax.grid()
# fig04.tight_layout

# # Uncomment this next line if you want to save the figure
# # fig04.savefig('figure_2.png', dpi=500)

# ###
# ###Wavelet estimation by spectral division
# def spectral_division(reflectivity, data):
    
#     seis_fft = np.fft.fft(data)
#     ref_fft = np.fft.fft(reflectivity)

#     wavelet_spec = seis_fft / ref_fft
#     wavelet_div = np.fft.ifft(wavelet_spec)
    
#     return wavelet_div

# spec_div = spectral_division(r, s)

# fig05 = plt.figure()
# ax = fig05.add_subplot(111)
# ax.plot(t, np.real(spec_div), 'k', lw=2)
# ax.set_xlim([0,0.256])

# def wigner(rpp, seismic):
#     opConvolve = la.toeplitz(rpp)
#     wavelet = lstsq(opConvolve, seismic)[0]
#     return wavelet     

# wigner_wave = wigner(r, s)

# fig06 = plt.figure()
# ax = fig06.add_subplot(111)
# ax.plot(t, wigner_wave, 'k', lw=2)
# ax.set_xlim([0,0.128])
# ax.set_ylim([-1,1])

# Y1 = 20 * np.log10(np.abs(np.fft.rfft(s))) 
# R1 = 20 * np.log10(np.abs(np.fft.rfft(r))) 
# W1 = 20 * np.log10(np.abs(np.abs(np.fft.rfft(s) / np.fft.rfft(r)))) 
# W2 = 20 * np.log10(np.fft.rfft(wigner_wave)) 

# fig07 = plt.figure(figsize=(15,3))
# ax = fig07.add_axes([0.1, 0.15, 0.25, 0.7])
# ax.plot(f,Y1, 'k', lw=1)
# ax.set_title('data', fontsize=14)
# ax.set_ylabel('power (dB)', fontsize=14)
# ax.set_xlabel('frequency (Hz)', fontsize=14)
# ax.set_ylim(-80,10)

# ax2 = fig07.add_axes([0.1 + 1*0.8/3, 0.15, 0.25, 0.7])
# ax2.plot(f,R1, 'k', lw=1)
# ax2.set_title('reflectivity', fontsize=14)
# ax2.set_xlabel('frequency (Hz)', fontsize=14)
# ax2.set_ylim(-80,10)
# ax2.set_yticklabels([])

# ax3 = fig07.add_axes([0.1 + 2*0.8/3, 0.15, 0.25, 0.7])
# ax3.plot(f,W1, 'k', lw=1)
# # ax3.plot(X,W2, 'dark blue', lw=1)
# ax3.set_title(' "wavelet" ', fontsize=14)
# ax3.set_xlabel('frequency (Hz)', fontsize=14)
# ax3.set_ylim(-80,10)
# ax3.set_yticklabels([])

# # Uncomment this next line if you want to save the figure
# # fig07.savefig('figure_3.png', dpi=500)

# ###
# ###Wavelet estimation by least squares
# clf = linear_model.Ridge(alpha = 0.5, fit_intercept=False)
# R = la.toeplitz(r)
# clf.fit(R, s)
# wavelet = clf.coef_

# Y2 = 20* np.log10(np.abs(np.fft.rfft(wavelet)))  

# fig08 = plt.figure(figsize = (10,3))
# ax = fig08.add_subplot(121)
# ax.plot(t, wavelet, 'k', lw=2)
# ax.set_xlim([0,0.128])
# ax.set_title('wavelet', fontsize=14)
# ax.set_ylabel('amplitude', fontsize=14)
# ax.set_xlabel('time (s)', fontsize=14)

# # Check the spectra
# ax2 = fig08.add_subplot(122)
# ax2.plot(f,Y2, 'k', lw=2)
# ax2.set_title('spectrum of wavelet', fontsize=14)
# ax2.set_ylabel('power (dB)', fontsize=14)
# ax2.set_xlabel('frequency (Hz)', fontsize=14)

# # modelled seismic
# fig09 = plt.figure(figsize=(15,3))
# ax = fig09.add_subplot(111) 
# ax.plot(t, np.dot(R, wavelet), 'k', lw=2)
# ax.set_title('synthetic', fontsize=14)
# ax.set_ylabel('amplitude', fontsize=14)
# ax.set_xlabel('time (s)', fontsize=14)

# wavelet_size = 15 # samples
# opProj = np.zeros((r.size, r.size))
# opProj[:wavelet_size, :wavelet_size] = np.eye(wavelet_size)

# op  = np.dot(R, opProj)
# wavelet = lstsq(op, s)[0]

# Y3 = 20* np.log10(np.abs(np.fft.rfft(wavelet)))  

# fig10 = plt.figure(figsize = (10,3))
# ax = fig10.add_axes([0.1, 0.15, 0.3, 0.7])
# ax.plot(t, wavelet, 'k', lw=2)
# ax.set_xlim([0,0.128])
# ax.set_title('wavelet', fontsize=14)
# ax.set_ylabel('amplitude', fontsize=14)
# ax.set_xlabel('time (s)', fontsize=14)

# # Check the spectra
# ax2 = fig10.add_axes([0.5, 0.15, 0.35, 0.7])
# ax2.plot(f, Y3, 'k', lw=2)
# ax2.set_title('spectrum of wavelet', fontsize=14)
# ax2.set_ylabel('power (dB)', fontsize=14)
# ax2.set_xlabel('frequency (Hz)', fontsize=14)

# # Uncomment this next line if you want to save the figure
# # fig10.savefig('figure_4.png', dpi=500)

