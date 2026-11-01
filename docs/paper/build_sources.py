"""Regenerate editable LaTeX and a readable text manuscript from manuscript.json."""
from pathlib import Path
import json,re
D=Path(__file__).resolve().parent

def tex(s):
    escaped=''.join({'\\':r'\textbackslash{}','&':r'\&','%':r'\%','$':r'\$','#':r'\#','_':r'\_','{':r'\{','}':r'\}','~':r'\textasciitilde{}','^':r'\textasciicircum{}'}.get(c,c) for c in str(s))
    return escaped.replace('/',r'/\allowbreak{}').replace(r'\_',r'\_\allowbreak{}')

def math(s):
    s=re.sub(r'\\(mathbb|mathcal)\s+([A-Za-z])',lambda m:'\\'+m[1]+'{'+m[2]+'}',s)
    return s.replace(r'\frac12',r'\frac{1}{2}').replace(r'\frac1m',r'\frac{1}{m}').replace(r'\overline L',r'\overline{L}')

def main():
    m=json.loads((D/'manuscript.json').read_text(encoding='utf-8'))
    out=[r'\documentclass[11pt,a4paper]{article}',r'\usepackage[T1]{fontenc}',r'\usepackage[utf8]{inputenc}',r'\usepackage[margin=25mm]{geometry}',r'\usepackage{mathptmx,amsmath,amssymb,graphicx,booktabs,longtable,array,hyperref}',r'\hypersetup{colorlinks=true,urlcolor=blue,linkcolor=black}',r'\setlength{\parindent}{0pt}',r'\setlength{\parskip}{6pt}',r'\renewcommand{\baselinestretch}{1.12}',r'\setlength{\tabcolsep}{4pt}',r'\renewcommand{\arraystretch}{1.2}',r'\setlength{\emergencystretch}{2em}',r'\sloppy',r'\title{\bfseries '+tex(m['title'])+'}',r'\author{'+tex(m['author'])+r'\\'+tex(m['affiliation'])+r'\\\href{mailto:'+m['email']+'}{'+tex(m['email'])+'}}',r'\date{'+tex(m['date'])+'}',r'\begin{document}',r'\maketitle',r'\section*{ABSTRACT}']
    plain=['# '+m['title'],'',m['author'],m['affiliation'],m['email'],m['date'],'','## ABSTRACT','']
    for label,text in m['abstract']:
        out.append(r'\textbf{'+tex(label)+':} '+tex(text)+r'\par')
        plain.extend([label+': '+text,''])
    out.append(r'\textbf{Keywords:} '+tex('; '.join(m['keywords']))+r'\par\clearpage')
    plain.extend(['Keywords: '+'; '.join(m['keywords']),''])
    for block in m['blocks']:
        kind=block['type']
        if kind=='heading':
            command='section' if block['level']==1 else 'subsection'
            if block['text'].startswith('Appendix A.'):out.append(r'\clearpage')
            out.append('\\'+command+'*{'+tex(block['text'])+'}')
            plain.extend([('#'* (block['level']+1))+' '+block['text'],''])
        elif kind=='paragraph':out.append(tex(block['text'])+'\n');plain.extend([block['text'],''])
        elif kind=='equation':
            out.extend([r'\begin{equation}',math(block['tex']),r'\end{equation}'])
            plain.extend(['$$',math(block['tex']),'$$',''])
        elif kind=='table':
            n=len(block['headers']);widths=block.get('widths') or [1/n]*n
            spec=''.join('>{\\raggedright\\arraybackslash}p{\\dimexpr '+f'{w:.4f}'+r'\linewidth-2\tabcolsep\relax}' for w in widths)
            header=' & '.join(r'\textbf{'+tex(c)+'}' for c in block['headers'])+r' \\'
            out.extend([r'{\small',r'\begin{longtable}{'+spec+'}',r'\caption{'+tex(block['title'])+r'}\\',r'\toprule',header,r'\midrule\endfirsthead',r'\toprule',header,r'\midrule\endhead',r'\bottomrule\endlastfoot'])
            for row in block['rows']:out.append(' & '.join(tex(c) for c in row)+r' \\')
            out.extend([r'\end{longtable}',r'}'])
            if block.get('source'):out.append(r'{\footnotesize Source: '+tex(block['source'])+r'\par}')
            plain.extend(['**'+block['title']+'**','', '| '+' | '.join(block['headers'])+' |','| '+' | '.join(['---']*n)+' |'])
            plain.extend('| '+' | '.join(str(c) for c in row)+' |' for row in block['rows']);plain.append('')
        elif kind=='figure':
            out.extend([r'\begin{figure}[htbp]',r'\centering',r'\includegraphics[width=\linewidth]{'+block['file']+'}',r'\caption{'+tex(block['title'])+'. '+tex(block['caption'])+'}',r'\end{figure}'])
            plain.extend(['!['+block['title']+']('+block['file']+')',block['caption'],''])
        elif kind=='reference':
            out.append('['+str(block['number'])+'] '+tex(block['text'])+(' '+r'\url{'+block['url']+'}' if block['url'] else '')+r'\par')
            plain.extend(['['+str(block['number'])+'] '+block['text']+(' '+block['url'] if block['url'] else ''),''])
    out.append(r'\end{document}')
    (D/'main.tex').write_text('\n'.join(out)+'\n',encoding='utf-8',newline='\n')
    (D/'MANUSCRIPT.md').write_text('\n'.join(plain)+'\n',encoding='utf-8',newline='\n')
    print('Generated main.tex and MANUSCRIPT.md from',len(m['blocks']),'content blocks.')

if __name__=='__main__':main()
