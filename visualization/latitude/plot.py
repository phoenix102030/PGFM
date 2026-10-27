from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
P=Path(__file__).resolve().parent
DATA=P.parents[1]/'data'
d=pd.read_csv(DATA/'latitude_metrics.csv')
plt.rcParams.update({'font.family':'serif','font.size':12,'pdf.fonttype':42})
colors={'PGFM':'#EE6677','FMB':'#4477AA','ConvLSTM':'#66CCEE','STGP':'#228833'}
fig,axes=plt.subplots(1,3,figsize=(12,3.8))
for i,(ax,metric,limit) in enumerate(zip(axes,['RMSE','MAE','nMAE'],[10,7,40])):
    rows=d[d.metric.eq(metric)]
    for model,color in colors.items():
        r=rows[rows.model.eq(model)].sort_values('latitude')
        ax.plot(r.latitude,r.value,color=color,label=model)
    ax.set(xlim=(-90,90),ylim=(0,limit),xlabel='Latitude (degree)',ylabel=metric+(' (%)' if metric=='nMAE' else ' (TECU)'))
    ax.set_xticks(range(-90,91,30));ax.grid(alpha=.15);ax.text(.02,.94,chr(65+i),transform=ax.transAxes)
    if i<2:
        r=d[d.metric.eq('background_tec') & d.panel.eq('AB'[i])].sort_values('latitude')
        right=ax.twinx();right.plot(r.latitude,r.value,'--',color='.45',label='Background TEC')
        right.set(ylim=(0,36.7424),ylabel='Background TEC (TECU)');right.tick_params(colors='.45')
handles,labels=axes[0].get_legend_handles_labels()
handles+=axes[0].figure.axes[3].get_legend_handles_labels()[0]
labels+=['Background TEC']
fig.legend(handles,labels,ncol=5,loc='upper center',frameon=False)
fig.subplots_adjust(left=.06,right=.98,bottom=.18,top=.82,wspace=.55)
for ext in ['png','pdf']:fig.savefig(P/('latitude.'+ext),dpi=300)
