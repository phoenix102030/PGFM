from pathlib import Path
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
P=Path(__file__).resolve().parent
DATA=P.parents[1]/'data'
parser=argparse.ArgumentParser();parser.add_argument('--frame',type=int,default=0);args=parser.parse_args()
z=np.load(DATA/'velocity_fields.npz');f=args.frame
if not 0<=f<24:raise ValueError('frame must be 0..23')
plt.rcParams.update({'font.family':'serif','font.size':12,'pdf.fonttype':42})
weight=1-z['lambda_map'][f]
fields=[('Target velocity',z['u_target'][f]),('Data contribution',(1-weight)*z['u_data'][f]),('Physical contribution',weight*z['u_phys'][f]),('Fused velocity',z['u_pred'][f])]
fig,axes=plt.subplots(2,2,figsize=(11,6))
vmax=max(float(np.max(np.abs(v))) for _,v in fields)
for ax,(title,v) in zip(axes.flat,fields):
    im=ax.pcolormesh(z['longitude'],z['latitude'],v,shading='nearest',vmin=-vmax,vmax=vmax,cmap='coolwarm',rasterized=True)
    ax.set(title=title,xlabel='Longitude (degree)',ylabel='Latitude (degree)')
fig.subplots_adjust(left=.07,right=.86,bottom=.10,top=.93,hspace=.45,wspace=.25)
cax=fig.add_axes([.89,.16,.02,.67]);fig.colorbar(im,cax=cax,label='Normalized flow-matching velocity')
for ext in ['png','pdf']:fig.savefig(P/f'velocity_frame{f:02d}.{ext}',dpi=300)
