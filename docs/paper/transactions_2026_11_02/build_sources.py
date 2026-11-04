"""Regenerate the readable and LaTeX companions from the updated manuscript."""
from pathlib import Path
import json,re
D=Path(__file__).resolve().parent
def tex(s):
    s=str(s).translate(str.maketrans({'–':'--','—':'---','’':"'",'‘':"'",'“':'``','”':"''",'…':'...','−':'-'}))
    chars={'\\':r'\textbackslash{}','&':r'\&','%':r'\%','$':r'\$','#':r'\#','_':r'\_','{':r'\{','}':r'\}','~':r'\textasciitilde{}','^':r'\textasciicircum{}'}
    s=''.join(chars.get(c,c) for c in s)
    math={'²':r'\textsuperscript{2}','³':r'\textsuperscript{3}','⁴':r'\textsuperscript{4}','′':r'\ensuremath{^{\prime}}','″':r'\ensuremath{^{\prime\prime}}','‴':r'\ensuremath{^{\prime\prime\prime}}','θ':r'\ensuremath{\theta}','Λ':r'\ensuremath{\Lambda}','Δ':r'\ensuremath{\Delta}','ℰ':r'\ensuremath{\mathcal E}','⁻':r'\textsuperscript{-}','⁹':r'\textsuperscript{9}','¹':r'\textsuperscript{1}','⁰':r'\textsuperscript{0}','⁵':r'\textsuperscript{5}','×':r'\ensuremath{\times}'}
    s=''.join(math.get(c,c) for c in s)
    for c,v in {'Π':r'\Pi','ω':r'\omega','δ':r'\delta','η':r'\eta','λ':r'\lambda','α':r'\alpha','ⱼ':'_j','₀':'_0','ₛ':'_s','ᵧ':'_y','ₙ':'_n'}.items():
        s=s.replace(c,r'\ensuremath{'+v+'}')
    return s.replace('/',r'/\allowbreak{}').replace(r'\_',r'\_\allowbreak{}')
def main():
    m=json.loads((D/'manuscript.json').read_text(encoding='utf-8'))
    out=[r'\documentclass[10pt,letterpaper,twocolumn]{article}',r'\usepackage[T1]{fontenc}',r'\usepackage[utf8]{inputenc}',r'\usepackage[left=.65in,right=.65in,top=.7in,bottom=.7in]{geometry}',r'\usepackage{mathptmx,amsmath,amssymb,graphicx,booktabs,longtable,array,hyperref}',r'\setlength{\columnsep}{.2in}',r'\setlength{\parindent}{.14in}',r'\setlength{\emergencystretch}{2em}',r'\sloppy',r'\title{'+tex(m['title'])+'}',r'\author{'+tex(m['author'])+r'\thanks{'+tex(m['affiliation'])+'. '+tex(m['email'])+'}}',r'\date{'+tex(m['date'])+'}',r'\begin{document}',r'\maketitle',r'\begin{abstract}'+tex(m['abstract'])+r'\end{abstract}',r'\textbf{Index Terms---}'+tex(', '.join(m['keywords']))+'.']
    md=['# '+m['title'],'',m['author'],m['affiliation'],m['email'],m['date'],'Evidence cutoff: '+m['evidence_cutoff'],'','## Abstract','',m['abstract'],'','Index Terms: '+', '.join(m['keywords']),'']
    eq=tb=fg=0
    for b in m['blocks']:
        k=b['type']
        if k=='heading':
            if b['text'].startswith('Appendix B.'):out.append(r'\clearpage\onecolumn')
            out.append(('\\section*' if b['level']==1 else '\\subsection*')+'{'+tex(b['text'])+'}')
            md.extend([('#'*(b['level']+1))+' '+b['text'],''])
        elif k=='paragraph':out.append(tex(b['text'])+'\n');md.extend([b['text'],''])
        elif k=='equation':
            eq+=1
            out.extend([r'\begin{equation}',r'\resizebox{.98\columnwidth}{!}{$\displaystyle '+b['tex']+'$}',r'\end{equation}'])
            md.extend(['$$',b['tex'],'$$',f'({eq})',''])
        elif k=='table':
            tb+=1;n=len(b['headers']);widths=[.18]+[.82/(n-1)]*(n-1)
            spec=''.join(r'>{\raggedright\arraybackslash}p{\dimexpr '+f'{w:.4f}'+r'\linewidth-2\tabcolsep\relax}' for w in widths)
            head=' & '.join(r'\textbf{'+tex(v)+'}' for v in b['headers'])+r'\\'
            out.extend([r'{\footnotesize\setlength{\tabcolsep}{4pt}\renewcommand{\arraystretch}{1.2}',r'\begin{longtable}{'+spec+'}',r'\caption{'+tex(b['title'])+r'}\\\toprule',head,r'\midrule\endfirsthead',r'\toprule',head,r'\midrule\endhead',r'\bottomrule\endlastfoot'])
            out.extend(' & '.join(tex(v) for v in row)+r'\\' for row in b['rows']);out.extend([r'\end{longtable}}',r'{\footnotesize Source: '+tex(b['source'])+r'\par}'])
            md.extend([f'**Table {tb}. '+b['title']+'**','','| '+' | '.join(b['headers'])+' |','| '+' | '.join(['---']*n)+' |']);md.extend('| '+' | '.join(str(c) for c in row)+' |' for row in b['rows']);md.extend(['','Source: '+b['source'],''])
        elif k=='figure':
            fg+=1;out.extend([r'\begin{figure}[htbp]\centering',r'\includegraphics[width=.96\linewidth,height=3.65in,keepaspectratio]{'+b['file']+'}',r'\caption{'+tex(b['title']+'. '+b['caption'])+r'}\end{figure}'])
            md.extend([f'![Figure {fg}. '+b['title']+']('+b['file']+')',b['caption'],'Source: '+b['source'],''])
        elif k=='reference':out.append('['+str(b['number'])+'] '+tex(b['text'])+(r' \url{'+b['url']+'}' if b.get('url') else '')+r'\par');md.extend(['['+str(b['number'])+'] '+b['text']+(' '+b['url'] if b.get('url') else ''),''])
    out.append(r'\end{document}')
    (D/'main.tex').write_text('\n'.join(out)+'\n',encoding='utf-8')
    (D/'MANUSCRIPT.md').write_text('\n'.join(md)+'\n',encoding='utf-8')
    print(f'Generated LaTeX/Markdown companions: {eq} equations, {tb} tables, {fg} figures.')
if __name__=='__main__':main()
