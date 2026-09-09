from pathlib import Path
import subprocess,json,re,hashlib,sys
import fitz
from PIL import Image,ImageDraw
Q=Path(__file__).resolve().parents[1]/'qa';P=Q.parent.parent/'paper_en_partc_cv5';R=Q/'renders';R.mkdir(parents=True,exist_ok=True)
report={'builds':{}}
with (Q/'build.log').open('a',encoding='utf-8') as log:
    def run(args):
        p=subprocess.run(args,cwd=P,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
        log.write(p.stdout.decode('utf-8',errors='replace'));log.flush()
        assert p.returncode==0, f'{args}; see build.log'
    for stem in ['main_en','main_en_anonymous']:
        if '--verify-only' not in sys.argv:
            cmd=['pdflatex','-interaction=nonstopmode','-halt-on-error',stem+'.tex']
            run(cmd);run(['bibtex',stem])
            for _ in range(5):
                run(cmd)
                txt=(P/(stem+'.log')).read_text(errors='replace')
                if not any(x in txt for x in ['Label(s) may have changed','There were undefined','Rerun to get cross-references right']):break
        txt=(P/(stem+'.log')).read_text(errors='replace')
        warnings=[x for x in txt.splitlines() if any(w in x for w in ['Overfull','undefined','Missing character:','Label(s) may have changed'])]
        with fitz.open(P/(stem+'.pdf')) as d:
            text='\n'.join(pg.get_text() for pg in d)
            for i,pg in enumerate(d):pg.get_pixmap(matrix=fitz.Matrix(1.5,1.5)).save(R/f'{stem}_p{i+1:02}.png')
            report['builds'][stem]={'pages':len(d),'warnings':warnings,'unknown_references':'??' in text}
            (Q/(stem+'_text.txt')).write_text(text,encoding='utf-8')
        print(stem,report['builds'][stem],flush=True)
def body(path):
    with fitz.open(path) as d:s='\n'.join(p.get_text(clip=fitz.Rect(0,50,p.rect.width,p.rect.height-52)) for p in d)
    s=s[s.index('1. Introduction'):]
    return re.sub(r'\s+','',re.sub(r'-\s*\n','',s))
report['named_anonymous_body_equal']=body(P/'main_en.pdf')==body(P/'main_en_anonymous.pdf')
for stem in ['main_en','main_en_anonymous']:
    files=[R/f'{stem}_p{i:02}.png' for i in range(1,report['builds'][stem]['pages']+1)]
    for start in range(0,len(files),6):
        sheet=Image.new('RGB',(1440,2070),'#ddd');draw=ImageDraw.Draw(sheet)
        for j,p in enumerate(files[start:start+6]):
            im=Image.open(p).convert('RGB');im.thumbnail((680,645))
            x=j%2*720+(720-im.width)//2;y=j//2*690+30
            sheet.paste(im,(x,y));draw.text((j%2*720+15,j//2*690+8),p.name,fill='black')
        sheet.save(R/f'{stem}_contact_{start//6+1}.jpg')
(Q/'build_checks.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
assert report['named_anonymous_body_equal']
assert all(not r['warnings'] and not r['unknown_references'] for r in report['builds'].values()),report

