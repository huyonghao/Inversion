import pylab as pl 
import pandas as pd
import numpy as np
# x = pl.array([0.25,	0.3, 0.35, 0.4, 0.45, 0.5])
# e = pl.array([0.1247, 0.1382, 0.1489, 0.1646, 0.187, 0.19])
# d = pl.array([0.048, 0.046, 0.05, 0.038, 0.051, 0.042])
# g = pl.array([0.0986, 0.1128, 0.121, 0.148, 0.158, 0.166])

# pl.plot(x, e, label="Epsilon")
# pl.plot(x, d, label="Delta")
# pl.plot(x, g, label="Gamma")
# pl.legend()
# pl.title("$Clay$ $content$")
# pl.show()

# import pylab as pl 

# x = pl.array([2.1, 3.8, 4.7, 6.5, 11.5, 16.4])
# e = pl.array([0.253, 0.241, 0.223, 0.198, 0.167, 0.139])
# d = pl.array([0.048, 0.048, 0.048, 0.048, 0.048, 0.048])
# g = pl.array([0.234, 0.218, 0.191, 0.174, 0.148, 0.126])

# pl.plot(x, e, label="Epsilon")
# pl.plot(x, d, label="Delta")
# pl.plot(x, g, label="Gamma")
# pl.legend()
# pl.title("$Porosity$")
# pl.show()

data_r = (pd.read_csv("D:\\line1_05.csv")).iloc[990:1901, :]
data = pd.read_csv("D:\\1.csv")
data.index = list(np.arange(990,1901))

vp_r = data_r.iloc[:, 5]/1000
vs_r = data_r.iloc[:, 6]/1000
rho_r = data_r.iloc[:, 4]
vp = data.iloc[:, 0]
vs = data.iloc[:, 1]
rho = data.iloc[:, 2]
vp_ani = data.iloc[:, 3]
vs_ani = data.iloc[:, 4]
rho_ani = data.iloc[:, 5]
vp_iso = data.iloc[:, 6]
vs_iso = data.iloc[:, 7]
rho_iso = data.iloc[:, 8]

pl.figure()
pl.plot(vp_r, label="$Vp$")
pl.plot(vp_iso, label="$iso$")
pl.legend()
pl.title("Vp")

pl.figure()
pl.plot(vs_r, label="$Vs$")
pl.plot(vs_iso, label="$iso$")
pl.legend()
pl.title("Vs")

pl.figure()
pl.plot(rho_r, label="$Rho$")
pl.plot(rho_iso, label="$iso$")
pl.legend()
pl.title("Rho")


pl.figure()
pl.plot(vp_ani, label="$ani$")
pl.plot(vp, label="$real$")
pl.legend()
pl.title("Vp * rho")

pl.figure()
pl.plot(vs_ani, label="$ani$")
pl.plot(vs, label="$real$")
pl.legend()
pl.title("Vs^2 * rho * exp((epsilon-delta)/k)")

pl.figure()
pl.plot(rho_ani, label="$ani$")
pl.plot(rho, label="$real$")
pl.legend()
pl.title("Vp * exp(epsilon)")

pl.show()