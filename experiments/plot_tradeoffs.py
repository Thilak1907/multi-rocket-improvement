"""Publication-sized trade-off plot with direct labels and unjittered zooms."""
import argparse
import json
import os
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

LABELS = {
    'baseline50': '50k reference',
    'buffered50': '50k buffers',
    'fused50': '50k sklearn bundle',
    'fast50': '50k blocked',
    'fast30': 'Retained 30k blocked',
    'auto30': '30k routed',
    'fused30': '30k sklearn',
    'fused20': '20k bundle',
    'fused10': '10k bundle',
    'pair50': 'LSPV+MPV pair',
    'hydra64': 'HYDRA only (64 groups)',
    'hybrid20h32': '20k + HYDRA (32 groups)',
}
COLORS = {'fast30':'#c44e00', 'auto30':'#7b4ab5', 'fused30':'#2166ac',
          'baseline50':'#111111', 'hydra64':'#377e72', 'hybrid20h32':'#377e72',
          'fused10':'#2166ac', 'fused20':'#2166ac', 'pair50':'#7b4ab5'}


def pareto_points(frame):
    best = -np.inf
    points = []
    for row in frame.sort_values(['speedup','accuracy'], ascending=[False,False]).itertuples():
        if row.accuracy > best:
            points.append((row.speedup, row.accuracy * 100))
            best = row.accuracy
    return np.array(sorted(points))


def plot_tradeoffs(frame, report, n_datasets=55):
    os.environ.setdefault('MPLCONFIGDIR', '/tmp/multirocket-matplotlib')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FormatStrFormatter

    report = Path(report); report.mkdir(parents=True, exist_ok=True)
    if set(frame.variant) != set(LABELS) or frame.variant.duplicated().any():
        raise ValueError('Expected one row for each of the twelve recorded configurations')
    if 'datasets' in frame and not (frame.datasets == n_datasets).all():
        raise ValueError('All plotted configurations must use the same dataset count')
    rows = frame.set_index('variant')
    reference_accuracy = float(rows.loc['baseline50', 'accuracy']) * 100
    with plt.rc_context({'font.size':10, 'axes.labelsize':10, 'xtick.labelsize':9,
                         'ytick.labelsize':9, 'axes.titlesize':10, 'pdf.fonttype':42,
                         'svg.fonttype':'none'}):
        fig = plt.figure(figsize=(7.16, 5.1), layout='constrained')
        grid = fig.add_gridspec(2, 2, width_ratios=(1.12, 1), hspace=.12, wspace=.08)
        overview = fig.add_subplot(grid[:, 0])
        thirty = fig.add_subplot(grid[0, 1])
        fifty = fig.add_subplot(grid[1, 1])

        def point(ax, name, suffix=''):
            row = rows.loc[name]
            marker = '*' if name=='fast30' else ('s' if name=='baseline50' else ('D' if name=='auto30' else 'o'))
            size = 165 if name=='fast30' else 42
            item = ax.scatter(row.speedup, 100*row.accuracy, s=size, marker=marker,
                              color=COLORS.get(name, '#555555'), edgecolors='white',
                              linewidths=.6, zorder=5 if name=='fast30' else 4)
            item.set_gid(name + suffix)

        def label(ax, name, position, text=None, ha='center', coords='data'):
            row=rows.loc[name]
            ax.annotate(text or LABELS[name], (row.speedup, 100*row.accuracy), xytext=position,
                        textcoords=coords, ha=ha, va='center', fontsize=9.5,
                        color=COLORS.get(name, '#333333'), weight='bold' if name=='fast30' else 'normal',
                        arrowprops=dict(arrowstyle='-', color=COLORS.get(name, '#666666'),
                                        linewidth=.7, shrinkA=3, shrinkB=5),
                        annotation_clip=False)

        for name in LABELS:
            point(overview, name)
        frontier = pareto_points(frame)
        overview.plot(frontier[:,0], frontier[:,1], ls='--', color='#555555', lw=1.2, zorder=1)
        overview.axhline(reference_accuracy, color='#777777', ls=':', lw=1)
        overview.axvline(1, color='#777777', ls='-', lw=.9)
        overview.set(xlim=(min(.43,float(frame.speedup.min())*.7),max(4.98,float(frame.speedup.max())*1.085)),
                     ylim=(float(frame.accuracy.min())*100-.20,float(frame.accuracy.max())*100+.33),
                     xlabel='Speed-up over 50k reference', ylabel='Mean test accuracy (%)',
                     title='(a) All configurations')
        overview.set_xticks([1,2,3,4])
        overview.grid(alpha=.15)
        label(overview, 'hydra64', (1.25,86.60), 'HYDRA only\n64 groups')
        label(overview, 'pair50', (1.60,87.10), 'LSPV+MPV\npair')
        label(overview, 'hybrid20h32', (1.07,88.00), '20k + HYDRA\n32 groups')
        label(overview, 'fused20', (3.12,87.60), '20k bundle')
        label(overview, 'fused10', (4.15,86.98), '10k bundle')
        overview.annotate('30k and 50k points:\nsee (b) and (c)',
                          (1.50,reference_accuracy+.01), xytext=(3.00,87.88),
                          ha='center', va='center', fontsize=9.5,
                          arrowprops=dict(arrowstyle='-', color='#666666', lw=.7))

        for name in ('fast30','auto30','fused30'):
            point(thirty,name,'_zoom')
        thirty.axhline(reference_accuracy, color='#777777', ls=':', lw=1)
        x30=rows.loc[['fast30','auto30','fused30'],'speedup'];y30=100*rows.loc[['fast30','auto30','fused30'],'accuracy']
        pad30=max(float(x30.max()-x30.min())*.45,.012)
        thirty.set(xlim=(float(x30.min())-pad30,float(x30.max())+pad30),
                   ylim=(min(reference_accuracy,float(y30.min()))-.005,max(reference_accuracy,float(y30.max()))+.020),
                   xlabel='Speed-up', title='(b) 30k comparison (zoom)')
        from matplotlib.ticker import MaxNLocator
        thirty.xaxis.set_major_locator(MaxNLocator(3))
        thirty.yaxis.set_major_locator(MaxNLocator(3,steps=[1,2,5,10]))
        thirty.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        thirty.grid(alpha=.15)
        label(thirty,'fast30',(.18,.83),'Retained\n30k blocked',coords='axes fraction')
        label(thirty,'auto30',(.42,.24),'30k routed',coords='axes fraction')
        label(thirty,'fused30',(.82,.83),'30k\nsklearn',coords='axes fraction')

        for name in ('baseline50','buffered50','fast50','fused50'):
            point(fifty,name,'_zoom')
        fifty.axhline(reference_accuracy, color='#777777', ls=':', lw=1)
        fifty.axvline(1, color='#777777', ls='-', lw=.9)
        x50=rows.loc[['baseline50','buffered50','fast50','fused50'],'speedup']
        y50=100*rows.loc[['baseline50','buffered50','fast50','fused50'],'accuracy']
        pad50=max(float(x50.max()-x50.min())*.2,.012)
        fifty.set(xlim=(float(x50.min())-pad50,float(x50.max())+pad50),
                  ylim=(float(y50.min())-.021,float(y50.max())+.028),
                  xlabel='Speed-up', title='(c) 50k controls (zoom)')
        fifty.xaxis.set_major_locator(MaxNLocator(3))
        fifty.yaxis.set_major_locator(MaxNLocator(3,steps=[1,2,5,10]))
        fifty.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        fifty.grid(alpha=.15)
        label(fifty,'baseline50',(.19,.17),'50k\nreference',coords='axes fraction')
        label(fifty,'buffered50',(.34,.83),'50k\nbuffers',coords='axes fraction')
        label(fifty,'fast50',(.80,.83),'50k\nblocked',coords='axes fraction')
        label(fifty,'fused50',(.75,.18),'50k sklearn\nbundle',coords='axes fraction')
        fig.savefig(report/'tradeoffs_all.png',dpi=300)
        fig.savefig(report/'tradeoffs_all.pdf')
        fig.savefig(report/'tradeoffs_all.svg')
        plt.close(fig)

    svg=ET.parse(report/'tradeoffs_all.svg');namespace='{http://www.w3.org/2000/svg}'
    for element in svg.getroot().iter():
        name=element.get('id','').removesuffix('_zoom')
        if name in rows.index:
            r=rows.loc[name];tip=ET.Element(namespace+'title')
            tip.text=f'{LABELS[name]} [{name}]: accuracy {100*r.accuracy:.3f}%; speed-up {r.speedup:.3f}x'
            element.insert(0,tip)
    svg.write(report/'tradeoffs_all.svg',encoding='utf-8',xml_declaration=True)
    pd.DataFrame([dict(identifier=name,label=label,retained=name=='fast30')
                  for name,label in LABELS.items()]).to_csv(report/'plot_labels.csv',index=False)
    metadata=dict(datasets=n_datasets,retained='fast30',timing='one common first run',
                  coordinates_jittered=False,reference_accuracy_pct=reference_accuracy,
                  points={name:dict(speedup=float(rows.loc[name,'speedup']),accuracy_pct=float(100*rows.loc[name,'accuracy']))
                          for name in LABELS})
    (report/'plot_metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,default=Path('reports/review_revision/all_configurations.csv'))
    parser.add_argument('--output',type=Path,default=Path('reports/review_revision'))
    args=parser.parse_args()
    frame=pd.read_csv(args.input)
    plot_tradeoffs(frame,args.output,n_datasets=int(frame.datasets.min()))


if __name__=='__main__':main()
