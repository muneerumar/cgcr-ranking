#!/usr/bin/env python3
"""deterministic vector PDF/300-dpi PNG figures from saved paper values."""
import argparse
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from paper_common import ROOT, RUN, CATEGORIES, CATEGORY_LABELS, METHODS, METHOD_LABELS, build, write_tables, write_manifest, cell_value

# Okabe-Ito palette, consistent method colors throughout all figures.
COLORS = dict(zip(METHODS, ['#000000', '#999999', '#0072B2', '#E69F00', '#009E73', '#56B4E9', '#CC79A7', '#D55E00']))
COLORS['seller_text'] = COLORS['no_defense']
SHORT = dict(METHOD_LABELS, geo_filter='Shield + Qwen')
CATS = list(CATEGORIES.values())
CAT_SHORT = ['Home and Kitchen', 'Electronics', 'Tools and Home Improvement']
STYLE = {'font.family':'serif', 'font.serif':['STIXGeneral'], 'mathtext.fontset':'stix', 'font.size':9,
         'axes.labelsize':9, 'xtick.labelsize':8, 'ytick.labelsize':8, 'legend.fontsize':8,
         'axes.spines.top':False, 'axes.spines.right':False, 'axes.linewidth':.6,
         'pdf.fonttype':42, 'ps.fonttype':42, 'savefig.dpi':300, 'path.simplify':False}


def val(p, row, name):
    return cell_value(p, row[name])


def ci(p, row, prefix):
    mean, lower, upper = [val(p,row,prefix+'_'+field) for field in ('mean','ci_low','ci_high')]
    return mean, np.array([[max(0.,mean-lower)], [max(0.,upper-mean)]])


def save(p, fig, folder, name, rows, caption):
    folder.mkdir(parents=True,exist_ok=True)
    fig.savefig(folder/(name+'.pdf'), metadata={'Creator':'CGCR saved-result renderer', 'Producer':'Matplotlib', 'CreationDate':None, 'ModDate':None})
    fig.savefig(folder/(name+'.png'), dpi=300, metadata={'Software':'CGCR saved-result renderer'})
    width,height = fig.get_size_inches()
    p.figures[name].update(width_in=float(width),height_in=float(height),png_dpi=300,
        points=[{'table':table,'row':index,'numeric_values':{k:v['node'] for k,v in row.items() if isinstance(v,dict) and 'node' in v}} for table,index,row in sorted(rows,key=lambda entry:(entry[0],entry[1]))])
    (folder/(name+'.tex')).write_text('% Generated figure wrapper; caption contains no additional measurements.\n\\begin{figure}[p]\n\\centering\n\\includegraphics[width='+('0.49\\textwidth' if width<4 else '\\textwidth')+']{figures/'+name+'.pdf}\n\\caption{'+caption+'}\n\\label{fig:'+name+'}\n\\end{figure}\n')
    plt.close(fig)


def grouped(p, folder, name, table, prefix, caption, attack=None):
    """Bars (Promote@3) or point estimates (NDCG@10) with 95% CIs.

    Each category shows only the methods evaluated there, centred without gaps.
    Promote@3 bars carry value labels so zero-height bars stay visible."""
    rows=[(i,r) for i,r in enumerate(p.tables[table]['rows']) if attack is None or r['attack']==attack]
    bars=prefix=='promote3'
    fig,ax=plt.subplots(figsize=(7.2,2.75));fig.subplots_adjust(left=.085,right=.99,bottom=.15,top=.80)
    width=.098;points=[];seen=set()
    for category_index,category in enumerate(CATS):
        present=[]
        for method in METHODS:
            matches=[(i,r) for i,r in rows if r['method']==method and r['category']==category]
            if matches:present.append((method,)+matches[0])
        for j,(method,index,row) in enumerate(present):
            value,error=ci(p,row,prefix);x=category_index+(j-(len(present)-1)/2)*width
            if bars:
                ax.bar(x,value,width=width*.88,color=COLORS[method],edgecolor='white',linewidth=.3,zorder=3)
                ax.errorbar(x,value,yerr=error,color='#333333',capsize=1.5,elinewidth=.65,capthick=.65,zorder=4,fmt='none')
                ax.text(x,value+error[1][0]+.012,f'{value:.2f}',ha='center',va='bottom',fontsize=5.6,rotation=90,zorder=5)
            else:
                ax.errorbar(x,value,yerr=error,fmt='o',color=COLORS[method],markeredgecolor='white',markeredgewidth=.4,markersize=5,capsize=1.8,elinewidth=.8,capthick=.8,zorder=4)
            points.append((table,index,row));seen.add(method)
    ax.set_xticks(range(3),CAT_SHORT);ax.set_ylabel('Promote@3' if bars else 'Clean NDCG@10')
    ax.set_xlim(-.5,2.5)
    if bars:ax.set_ylim(0,max(.1,max(val(p,r,prefix+'_ci_high') for i,r in rows)*1.22))
    else:ax.set_ylim(.70,.95)
    ax.grid(axis='y',color='#dddddd',linewidth=.45,zorder=0);ax.set_axisbelow(True)
    legend=[Line2D([],[],marker='s' if bars else 'o',linestyle='',markersize=6 if bars else 5,color=COLORS[m],label=SHORT[m]) for m in METHODS if m in seen]
    fig.legend(handles=legend,loc='upper center',bbox_to_anchor=(.53,1.0),ncol=4,frameon=False,columnspacing=1.2,handletextpad=.4)
    save(p,fig,folder,name,points,caption)


def scatter(p,folder,name,table,supplement=False):
    categories=CATS[:1] if supplement else CATS
    fig,axes=plt.subplots(1,len(categories),figsize=(3.5,3.0) if supplement else (7.2,2.85),squeeze=False,sharex=True,sharey=True)
    fig.subplots_adjust(left=.19 if supplement else .085,right=.97,bottom=.29,top=.86,wspace=.16)
    points=[];rows=p.tables[table]['rows']
    xmax=max(val(p,r,'attack_success_ci_high') for r in rows)*1.18+.01
    ymax=max(val(p,r,'responsiveness_ci_high') for r in rows)*1.15
    for ax,category in zip(axes[0],categories):
        for index,row in enumerate(rows):
            if row['category']!=category:continue
            x,xerr=ci(p,row,'attack_success');y,yerr=ci(p,row,'responsiveness');method=row['method']
            ax.errorbar(x,y,xerr=xerr,yerr=yerr,fmt='o',color=COLORS[method],markeredgecolor='white',markeredgewidth=.5,markersize=6,capsize=2,elinewidth=.8,zorder=4)
            points.append((table,index,row))
        ax.set_xlim(-.015,xmax);ax.set_ylim(-.015,ymax);ax.grid(color='#dddddd',linewidth=.45)
        ax.set_xlabel('Mean Promote@3')
    axes[0,0].set_ylabel('Responsiveness')
    for index,ax in enumerate(axes[0]):
        box=ax.get_position();fig.text((box.x0+box.x1)/2,.045,('' if supplement else '('+chr(97+index)+') ')+CATEGORY_LABELS[categories[index]],ha='center',fontsize=8)
    handles=[Line2D([],[],marker='o',linestyle='',color=COLORS[m],label=METHOD_LABELS[m]) for m in ('seller_text','details_only','cgcr')]
    fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.53,1.),ncol=3,frameon=False,columnspacing=.9,handletextpad=.3)
    caption=('Home and Kitchen supplementary view averaging all six frozen attacks within each query.' if supplement else 'Robustness--responsiveness trade-off using the same three attacks in every category: core\\_reasoning\\_opt5, core\\_reasoning, framing\\_M3\\_opt5. Attack success is averaged within each query before bootstrapping. Panels show Home and Kitchen, Electronics, and Tools and Home Improvement; only the three methods with measured responsiveness are shown.')+' Error bars are query-bootstrap 95\\% intervals on both axes. Attack and responsiveness denominators are exported separately.'
    save(p,fig,folder,name,points,caption)


def flooding(p,folder):
    name='F5_review_flooding';table='T5_adaptive';points=[];counts=set()
    fig,axes=plt.subplots(1,2,figsize=(7.2,2.7));fig.subplots_adjust(left=.085,right=.98,bottom=.19,top=.84,wspace=.28)
    specs=[('unverified','o','-',COLORS['cgcr'],'Unverified reviews'),('verified','s','--','#D55E00','Verified-purchase flag')]
    for verified,marker,style,color,label in specs:
        selected=sorted([(i,r) for i,r in enumerate(p.tables[table]['rows']) if r['method']=='cgcr' and r['budget'] is not None and r['verified_flag']==verified],key=lambda v:val(p,v[1],'budget'))
        offset=-.35 if verified=='unverified' else .35
        for ax,prefix in zip(axes,('licensed','promote3')):
            x=[val(p,r,'budget') for i,r in selected];y=[val(p,r,prefix+'_mean') for i,r in selected]
            error=np.array([[v-val(p,r,prefix+'_ci_low') for v,(i,r) in zip(y,selected)],[val(p,r,prefix+'_ci_high')-v for v,(i,r) in zip(y,selected)]])
            ax.errorbar([b+offset for b in x],y,yerr=error,marker=marker,linestyle=style,color=color,markersize=5,capsize=2,linewidth=1.,elinewidth=.8)
            ax.set_xlabel('Injected review budget B');ax.set_xticks(x);ax.set_xlim(-1.2,21.2);ax.set_ylim(-.01,.30);ax.grid(color='#dddddd',linewidth=.45)
            counts.update(int(val(p,r,prefix+'_n')) for i,r in selected)
        points.extend((table,i,r) for i,r in selected)
    axes[0].set_ylabel('Inserted claim licensed');axes[1].set_ylabel('Promote@3 under CGCR')
    if len(counts)==1:
        n=next(iter(counts))
        for ax in axes:ax.text(.98,.95,f'n = {n} queries',transform=ax.transAxes,ha='right',va='top',fontsize=7)
    handles=[Line2D([],[],marker=m,linestyle=st,color=c,markersize=5,linewidth=1.,label=l) for _,m,st,c,l in specs]
    fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.53,1.0),ncol=2,frameon=False)
    save(p,fig,folder,name,points,'Review flooding against CGCR in Home and Kitchen, by injected-review budget and purchase flag. Left: end-to-end licensing of the inserted claim, counting extraction misses as unlicensed. Right: Promote@3. Claims are unsupported by genuine evidence at baseline; no falsity is asserted. Error bars are query-bootstrap 95\\% intervals; zero-event bootstrap intervals may be degenerate. Points are offset horizontally for visibility.')


def write_pipeline(folder):
    text=r'''% qualitative pipeline; no attack prompts/texts or empirical numbers.
\begin{tikzpicture}[
  box/.style={draw,rounded corners=2pt,align=center,minimum height=1.1cm,text width=2.5cm,inner sep=5pt},
  evidence/.style={box,fill=blue!5},
  flow/.style={-{Stealth[length=2mm]},thick},font=\small,node distance=7mm]
\node[box,fill=orange!8] (seller) {Seller text\\attacker-controlled};
\node[box,right=of seller] (extract) {Hybrid extraction\\typed claims};
\node[box,right=of extract] (verify) {Catalog / review\\verification};
\node[box,right=of verify] (render) {Deterministic\\neutral rendering};
\node[box,right=of render] (rank) {Fixed-order\\LLM ranking};
\node[evidence,below=of verify] (catalog) {Frozen typed catalog\\variant / unit checks};
\node[evidence,below=of catalog] (reviews) {Customer reviews\\retrieval + NLI};
\node[box,below=of render] (audit) {Per-claim\\evidence trace};
\draw[flow] (seller)--(extract);\draw[flow] (extract)--(verify);
\draw[flow] (verify)--node[above,font=\scriptsize]{licensed claims}(render);
\draw[flow] (render)--(rank);\draw[flow] (catalog)--(verify);
\draw[flow] (reviews.west)-|([xshift=-6mm]catalog.west)|-(verify.south west);
\draw[flow] (catalog.west)-| (extract.south);
\draw[flow] (verify.south east)--(audit.west);
\end{tikzpicture}
'''
    (folder/'pipeline.tex').write_text(text)


def generate(root=ROOT,output=None):
    output=Path(output) if output else Path(root)/'paper';p=build(root,RUN);write_tables(p,output)
    with plt.rc_context(STYLE):
        folder=output/'figures'
        grouped(p,folder,'F2_strongest_attack','T1_main_robustness','promote3','Promote@3 for CORE reasoning opt5, selected as the strongest attack using the frozen dev-based attack ordering. Methods are compared on the same targets and orders within each category. Error bars are 95\\% query-bootstrap intervals. Unrun method/category cells are omitted.','core_reasoning_opt5')
        scatter(p,folder,'F3_robustness_responsiveness','F3_common_attacks_data')
        grouped(p,folder,'F4_clean_quality','T2_clean_quality','ndcg10','Clean NDCG@10 by method and category, on utility-eligible queries only. Error bars are 95\\% query-bootstrap intervals. Unrun method/category cells are omitted.')
        flooding(p,folder)
        scatter(p,folder,'S1_home_kitchen_six_attacks','S1_home_kitchen_six_attacks',True)
        write_pipeline(folder)
    write_manifest(p,output)
    return p


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,default=ROOT/'paper');args=parser.parse_args()
    p=generate(ROOT,args.output)
    print(f'Generated {len(p.figures)} vector PDFs, matching 300-dpi PNGs, TikZ pipeline and provenance manifest.')


if __name__=='__main__':main()
