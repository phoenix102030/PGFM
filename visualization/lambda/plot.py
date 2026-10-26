from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
P=Path(__file__).resolve().parent
r=pd.read_csv(P/'lambda_latitude.csv').sort_values('latitude')
z=np.load(P/'lambda_maps.npz')
plt.rcParams.update({'font.family':'serif','font.size':12,'pdf.fonttype':42})
fig,axes=plt.subplots(1,2,figsize=(11,3.8))
color='#AA3377'
axes[0].fill_between(r.latitude,r.p10,r.p90,color=color,alpha=.1)
axes[0].fill_between(r.latitude,r.p25,r.p75,color=color,alpha=.2)
axes[0].plot(r.latitude,r['median'],color=color,label=r'$\lambda$ median')
axes[0].plot(r.latitude,r['mean'],'--',color=color,label=r'$\lambda$ mean')
axes[0].set(xlim=(-90,90),ylim=(0,1),xlabel='Latitude (degree)',ylabel=r'$\lambda$')
axes[0].legend(frameon=False)
im=axes[1].pcolormesh(z['longitude'],z['latitude'],z['lambda'].mean(axis=0),shading='nearest',vmin=0,vmax=1,cmap='viridis',rasterized=True)
axes[1].set(xlabel='Longitude (degree)',ylabel='Latitude (degree)')
fig.colorbar(im,ax=axes[1],label=r'$\lambda$')
fig.tight_layout()
for ext in ['png','pdf']:fig.savefig(P/('lambda.'+ext),dpi=300)
