#2PCFx10_512.dat
#2PCFx10_1G.dat
#2PCFx10_2G.dat

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

data_512 = pd.read_csv("2PCFx10_512.dat")
data_1G = pd.read_csv("2PCFX10_1G.dat")
data_2G = pd.read_csv("2PCFx10_2G.dat")

dmaxs = np.linspace(20,160,8)

fig, (ax1,ax2,ax3) = plt.subplots(3,1, figsize=(8, 11.5), dpi=720)

for dmax in dmaxs:
    data = data_512[data_512['dmax']==dmax][['partitions','time [ms]']].groupby('partitions').mean()
    p = data.index.to_numpy() 
    t = data['time [ms]'].to_numpy()
    ax1.plot(p,t, label=str(dmax))

    data = data_1G[data_1G['dmax']==dmax][['partitions','time [ms]']].groupby('partitions').mean()
    p = data.index.to_numpy() 
    t = data['time [ms]'].to_numpy()
    ax2.plot(p,t, label=str(dmax))

    data = data_2G[data_2G['dmax']==dmax][['partitions','time [ms]']].groupby('partitions').mean()
    p = data.index.to_numpy() 
    t = data['time [ms]'].to_numpy()
    ax3.plot(p,t, label=str(dmax))


ax1.set_title('data_512MPc')
ax2.set_title('data_1GPC')
ax3.set_title('data_2GPc')

ax1.set_ylabel('Time [ms]')
ax2.set_ylabel('Time [ms]')
ax3.set_ylabel('Time [ms]')
ax3.set_xlabel('Partitions')

ax1.grid()
ax2.grid()
ax3.grid()
fig.savefig("2pcfiso_opt.png")