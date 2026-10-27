from pathlib import Path
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA = ROOT.parents[1] / 'data'
COLORS = {'PGFM':'#EE6677','FMB':'#4477AA','Oracle':'#AA3377','ConvLSTM':'#66CCEE','STGP':'#228833'}
MARKERS = {'PGFM':'D','FMB':'o','Oracle':'s','ConvLSTM':'+','STGP':'^'}

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--input',type=Path,default=DATA/'all_models_24h_error.csv')
    parser.add_argument('--output',type=Path,default=None)
    args=parser.parse_args()
    frame=pd.read_csv(args.input)
    stem=args.output or ROOT/'comparison'
    stem.parent.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.family':'serif','font.serif':['STIXGeneral','DejaVu Serif'],
        'font.size':12,'axes.labelsize':14,'xtick.labelsize':11,'ytick.labelsize':11,
        'legend.fontsize':12,'axes.linewidth':0.8,'lines.linewidth':1.7,
        'svg.fonttype':'none','pdf.fonttype':42,'text.usetex':False})
    fig, axes=plt.subplots(1,3,figsize=(12.4,4.2))
    for i,(ax,key,label) in enumerate(zip(axes,['rmse_tecu','mae_tecu','nmae_percent'],['RMSE (TECU)','MAE (TECU)','nMAE (%)'])):
        for model in ['PGFM','FMB','Oracle','ConvLSTM','STGP']:
            rows=frame[frame.model.eq(model)].sort_values('lead_hour')
            ax.plot(rows.lead_hour, rows[key], label=model,color=COLORS[model],
                    marker=MARKERS[model],markersize=3.5,markevery=2,
                    linestyle='--' if model=='STGP' else '-',linewidth=2 if model=='STGP' else 1.5)
        ax.set(xlim=(1,24),ylim=(0,float(frame[key].max())*1.12),xlabel='Lead time (hour)',ylabel=label)
        ax.set_xticks([1,6,12,18,24]);ax.grid(alpha=.18,linewidth=.5)
        ax.text(.02,.94,chr(65+i),transform=ax.transAxes,weight='bold',fontsize=14)
        ax.spines[['top','right']].set_visible(False)
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,ncol=5,loc='upper center',bbox_to_anchor=(.5,.91),frameon=False)
    fig.subplots_adjust(left=.065,right=.985,bottom=.18,top=.75,wspace=.30)
    fig.savefig(stem.with_suffix('.png'),dpi=300,facecolor='white')
    plt.close(fig)

if __name__=='__main__':
    main()
