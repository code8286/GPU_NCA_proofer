from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
from copy import deepcopy
from lxml import etree as E
import json,hashlib,re
from docx import Document
from docx.shared import Inches
from io import BytesIO

import argparse
parser=argparse.ArgumentParser(description='Rebuild the Word article from the unchanged author-supplied IEEE template.')
parser.add_argument('--template',type=Path,required=True)
args=parser.parse_args()
D=Path(__file__).resolve().parent; R=D.parents[2]
REF=args.template.resolve()
assert hashlib.sha256(REF.read_bytes()).hexdigest()=='eea69bea29ff9c6e72eabc7717cd0cb58a39ea187894d73d09d70817773d817b'
M=json.loads((D/'manuscript.json').read_text(encoding='utf-8'))
W='http://schemas.openxmlformats.org/wordprocessingml/2006/main'; A='http://schemas.openxmlformats.org/officeDocument/2006/relationships'; O='http://schemas.openxmlformats.org/officeDocument/2006/math'; NS={'w':W,'r':A,'m':O}
def node(tag,**attrs):return E.Element('{'+W+'}'+tag,{'{'+W+'}'+k:str(v) for k,v in attrs.items()})
def run(text,bold=False,size=None,font=None):
 r=node('r');rp=node('rPr')
 if bold:rp.append(node('b'))
 if size:rp.append(node('sz',val=int(size*2)))
 if font:rp.append(node('rFonts',ascii=font,hAnsi=font))
 if len(rp):r.append(rp)
 t=node('t');t.set('{http://www.w3.org/XML/1998/namespace}space','preserve');t.text=text;r.append(t);return r
def para(text='',style='Text',keep=False,size=None):
 p=node('p');pr=node('pPr');pr.append(node('pStyle',val=style))
 if style in ['Heading2','References']:
  num=node('numPr');num.append(node('numId',val=0));pr.append(num)
 if keep:pr.append(node('keepNext'))
 p.append(pr)
 if text:p.append(run(text,size=size))
 return p
with ZipFile(REF) as z:parts={n:z.read(n) for n in z.namelist()}
root=E.fromstring(parts['word/document.xml']);body=root.find('w:body',NS)
original_pars=list(body.findall('w:p',NS));secs=root.xpath('//w:sectPr',namespaces=NS)
titlepr=deepcopy(original_pars[0].find('w:pPr',NS));authorpr=deepcopy(original_pars[2].find('w:pPr',NS))
for child in list(body):body.remove(child)
p=node('p');p.append(titlepr);p.append(run(M['title']));body.append(p)
p=node('p');p.append(authorpr);p.append(run(M['author']));rr=node('r');rr.append(node('footnoteReference',id=1));p.append(rr);body.append(p)
# Full-width title boundary from the actual source. No artificial title page.
p=para('',style='Normal');p.find('w:pPr',NS).append(deepcopy(secs[0]));body.append(p)
body.append(para('Abstract—'+M['abstract'],style='Abstract'))
body.append(para('Index Terms—'+', '.join(M['keywords'])+'.',style='IndexTerms'))
# Unnumbered supplied author/correspondence footnote, no invented funding.
eq=0;tables=0;figures=0
def mathrun(text):
 r=E.Element('{'+O+'}r');rp=node('rPr');rp.append(node('rFonts',ascii='Cambria Math',hAnsi='Cambria Math'));rp.append(node('sz',val=20));r.append(rp)
 t=E.SubElement(r,'{'+O+'}t');t.text=text;return r
def equation(lines,number):
 p=para('',style='Equation');pr=p.find('w:pPr',NS);pr.append(node('keepNext',val=0));pr.append(node('spacing',before=80,after=80));pr.append(node('keepLines'))
 ac=E.SubElement(p,'{http://schemas.openxmlformats.org/markup-compatibility/2006}AlternateContent')
 choice=E.SubElement(ac,'{http://schemas.openxmlformats.org/markup-compatibility/2006}Choice',Requires='w16cex')
 for i,line in enumerate(lines):
  if i:r=node('r');r.append(node('br'));choice.append(r)
  om=E.SubElement(choice,'{'+O+'}oMath');om.append(mathrun(line))
 fallback=E.SubElement(ac,'{http://schemas.openxmlformats.org/markup-compatibility/2006}Fallback')
 from PIL import Image
 mathfile=D/'math'/f'eq{number:02}.png';im=Image.open(mathfile);width=im.width/600
 ip=image_doc.add_paragraph();ip.add_run().add_picture(str(mathfile),width=Inches(min(width,3.1)))
 for child in ip._p:
  if child.tag=='{'+W+'}r':fallback.append(deepcopy(child))
 r=node('r');r.append(node('tab'));p.append(r);p.append(run('('+str(number)+')'))
 return p
def roman(n):
 s=''
 for v,c in [(10,'X'),(9,'IX'),(5,'V'),(4,'IV'),(1,'I')]:
  while n>=v:s+=c;n-=v
 return s
def table(b,n):
 p=para('TABLE '+roman(n)+'\n'+b['title'],style='TableTitle',keep=True);body.append(p)
 tbl=node('tbl');pr=node('tblPr');pr.append(node('tblW',w=10368,type='dxa'));pr.append(node('tblLayout',type='fixed'))
 borders=node('tblBorders')
 for k in ['top','left','bottom','right','insideH','insideV']:borders.append(node(k,val='single',sz=4,color='D9D9D9'))
 pr.append(borders);mar=node('tblCellMar')
 for k in ['top','bottom']:mar.append(node(k,w=65,type='dxa'))
 for k in ['left','right']:mar.append(node(k,w=80,type='dxa'))
 pr.append(mar);tbl.append(pr)
 count=len(b['headers']);ratios=([.14,.47,.39] if n==1 else [.23,.27,.50] if n==2 else [.18]+[.82/(count-1)]*(count-1))
 widths=[int(10368*v) for v in ratios];grid=node('tblGrid')
 for width in widths:grid.append(node('gridCol',w=width))
 tbl.append(grid)
 for ri,row in enumerate([b['headers']]+b['rows']):
  tr=node('tr');trpr=node('trPr');trpr.append(node('cantSplit'))
  if ri==0:trpr.append(node('tblHeader'))
  tr.append(trpr)
  for ci,value in enumerate(row):
   tc=node('tc');cp=node('tcPr');cp.append(node('tcW',w=widths[ci],type='dxa'));cp.append(node('vAlign',val='center'))
   if ri==0:cp.append(node('shd',fill='EEEEEE'))
   tc.append(cp);p=para('',style='Normal');pp=p.find('w:pPr',NS);pp.append(node('spacing',before=0,after=0,line=240,lineRule='auto'));pp.append(node('ind',firstLine=0));pp.append(node('jc',val='left'))
   p.append(run(str(value),bold=(ri==0),size=8));tc.append(p);tr.append(tc)
  tbl.append(tr)
 body.append(tbl)
 p=para('Source: '+b['source'],style='FigureCaption');pp=p.find('w:pPr',NS);pp.append(node('spacing',after=160));body.append(p)

image_doc=Document();image_info=[]
for b in M['blocks']:
 typ=b['type']
 if typ=='heading':
  if b['text'].startswith('Appendix B.'):
   p=para('',style='Normal');sp=deepcopy(secs[-1]);pg=sp.find('w:pgNumType',NS)
   if pg is not None:sp.remove(pg)
   p.find('w:pPr',NS).append(sp);body.append(p)
  body.append(para(b['text'],style='Heading1' if b['level']==1 else 'Heading2'))
 elif typ=='paragraph':body.append(para(b['text']))
 elif typ=='equation':eq+=1;body.append(equation(b['word'],eq))
 elif typ=='table':tables+=1;table(b,tables)
 elif typ=='figure':
  figures+=1
  from PIL import Image
  im=Image.open(D/b['file']);width=7.1;height=width*im.height/im.width
  if height>3.65:width*=3.65/height
  ip=image_doc.add_paragraph();ip.add_run().add_picture(str(D/b['file']),width=Inches(width))
  xml=deepcopy(ip._p);pr=xml.find('w:pPr',NS)
  if pr is None:pr=node('pPr');xml.insert(0,pr)
  pr.append(node('keepNext'));pr.append(node('jc',val='center'));pr.append(node('spacing',before=120,after=0))
  body.append(xml);body.append(para(f'Fig. {figures}. '+b['title']+'. '+b['caption'],style='FigureCaption'))
 elif typ=='reference':
  p=para(f'[{b["number"]}] '+b['text']+(' [Online]. Available: '+b['url'] if b.get('url') else ''),style='References');pp=p.find('w:pPr',NS);pp.append(node('ind',left=280,hanging=280));pp.append(node('spacing',after=80));body.append(p)
# New full-width appendix is the cloned source full-width geometry.
sp=deepcopy(secs[0]);pg=sp.find('w:pgNumType',NS)
if pg is not None:sp.remove(pg)
sp.find('w:type',NS).set('{'+W+'}val','nextPage')
body.append(sp)
# Import just new picture payloads/relationships from a temporary in-memory doc.
rels=E.fromstring(parts['word/_rels/document.xml.rels']);RNS='http://schemas.openxmlformats.org/package/2006/relationships'
ct=E.fromstring(parts['[Content_Types].xml']);CT='http://schemas.openxmlformats.org/package/2006/content-types'
bio=BytesIO();image_doc.save(bio)
with ZipFile(bio) as z:
 ir=E.fromstring(z.read('word/_rels/document.xml.rels'));mapping={}
 for j,rel in enumerate(ir):
  if rel.get('Type','').endswith('/image'):
   rid='rIdResearchFigure'+str(j);oldid=rel.get('Id');target='media/research_'+Path(rel.get('Target')).name
   parts['word/'+target]=z.read('word/'+rel.get('Target'));mapping[oldid]=rid
   E.SubElement(rels,'{'+RNS+'}Relationship',Id=rid,Type=rel.get('Type'),Target=target)
 for blip in root.xpath('//*[local-name()="blip"]'):
  val=blip.get('{'+A+'}embed')
  if val in mapping:blip.set('{'+A+'}embed',mapping[val])
if not any(x.get('Extension')=='png' for x in ct):E.SubElement(ct,'{'+CT+'}Default',Extension='png',ContentType='image/png')
header=E.fromstring(parts['word/header1.xml'])
kept_title=False
for p in header.findall('w:p',NS):
 fields=p.xpath('.//w:instrText/text()',namespaces=NS)
 if any('PAGE' in s for s in fields):continue
 if kept_title:header.remove(p);continue
 kept_title=True
 pr=deepcopy(p.find('w:pPr',NS));p.clear()
 if pr is not None:p.append(pr)
 p.append(run('Graph Neural Cellular Automata and Controlled Transfer',size=8))
fn=E.fromstring(parts['word/footnotes.xml'])
for x in list(fn):
 if int(x.get('{'+W+'}id','-1'))>0:fn.remove(x)
x=node('footnote',id=1);x.append(para(M['author']+' is with '+M['affiliation']+'. Correspondence: '+M['email']+'.',style='Normal',size=8));fn.append(x)
settings=E.fromstring(parts['word/settings.xml']);uf=settings.find('w:updateFields',NS)
if uf is None:uf=node('updateFields',val='true');settings.append(uf)
else:uf.set('{'+W+'}val','true')
core=E.fromstring(parts['docProps/core.xml']);dns={'dc':'http://purl.org/dc/elements/1.1/','cp':'http://schemas.openxmlformats.org/package/2006/metadata/core-properties'}
for path,value in [('dc:title',M['title']),('dc:creator',M['author']),('cp:lastModifiedBy',M['author'])]:
 el=core.find(path,dns)
 if el is None:prefix,tag=path.split(':');el=E.SubElement(core,'{'+dns[prefix]+'}'+tag)
 el.text=value
for n,el in [('word/document.xml',root),('word/header1.xml',header),('word/footnotes.xml',fn),('word/settings.xml',settings),('word/_rels/document.xml.rels',rels),('[Content_Types].xml',ct),('docProps/core.xml',core)]:parts[n]=E.tostring(el,encoding='UTF-8',xml_declaration=True,standalone=True)
out=D/'Graph_NCA_Updated_Transactions_2026_11_02.docx'
with ZipFile(out,'w',ZIP_DEFLATED) as z:
 for n,data in parts.items():z.writestr(n,data)
print('Created',out.relative_to(R),';',eq,'native equations,',tables,'tables,',figures,'figures.')
print('Preserved source styles, numbering, theme and opaque package members unchanged.')
