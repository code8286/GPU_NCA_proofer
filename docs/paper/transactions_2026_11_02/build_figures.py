from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
D=Path(__file__).resolve().parent/'figures'; R=D.parents[3]
plt.rcParams.update({'font.family':'Arial','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':350})
s=json.loads((R/'docs/results/confirmation_v11_run1/summary.json').read_text())
fig,axs=plt.subplots(1,2,figsize=(7.1,2.7),layout='constrained')
for ax,key,title in zip(axs,['mean_native_nll','mean_changed_nll'],['(a) Native first patch','(b) Neighbor intervention']):
 for i,obj in enumerate(['standard','weighted']):
  gs=[next(g for g in s['groups'] if g['level']==lv and g['objective']==obj and g['arm']=='coherent') for lv in [512,4096]]
  ax.plot([0,1],[g[key] for g in gs],marker='o',label=obj.capitalize(),color=['#4a4a4a','#145e9a'][i])
 ax.set_xticks([0,1],['512','4096']);ax.set_xlabel('Available training conditions');ax.set_ylabel('NLL (nats per target)');ax.set_title(title);ax.grid(axis='y',alpha=.2)
axs[0].legend(frameon=False,fontsize=8)
fig.savefig(D/'confirmation.png');plt.close(fig)
s=json.loads((R/'docs/results/scientific_transfer_v2/CHECKS.json').read_text())
fig,axs=plt.subplots(1,3,figsize=(7.1,2.6),layout='constrained')
for name,case in s['cases'].items():
 q=case['reference'];x=q['eta'];y=q['state']
 for ax,idx in zip(axs,[1,3,5]):ax.plot(x,y[idx],label=name.replace('_',' '),lw=1.5)
for ax,title in zip(axs,['(a) Radial velocity f′','(b) Rotation profile g','(c) Temperature θ']):
 ax.set_title(title,fontsize=10);ax.set_xlabel('Similarity coordinate η');ax.grid(alpha=.2)
axs[0].legend(frameon=False,fontsize=7)
fig.savefig(D/'scientific_profiles.png');plt.close(fig)
print('Generated two figures from retained public summaries/states; no solver or model run.')
