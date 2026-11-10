import csv, io, json, zipfile, requests
from pathlib import Path
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed
root=Path(__file__).parent
src=root/'datas/nips2017_1000'; tgt=root/'datas/coco_val2017_1000'
src.mkdir(parents=True,exist_ok=True); tgt.mkdir(parents=True,exist_ok=True)
with zipfile.ZipFile('/root/autodl-tmp/nips2017_official.zip') as archive:
 names=sorted(n for n in archive.namelist() if n.startswith('images/') and n.endswith('.png'))
 if len(names)!=1000: raise RuntimeError(f'NIPS archive has {len(names)} images')
 manifest={}
 for i,name in enumerate(names):
  out=src/f'{i:04d}.png'
  manifest[out.name]=name
  if not out.exists():
   Image.open(io.BytesIO(archive.read(name))).convert('RGB').resize((224,224)).save(out)
 (root/'datas/nips2017_1000_manifest.json').write_text(json.dumps(manifest,indent=2))
print('NIPS source count',len(list(src.glob('*.png'))),flush=True)
if len(list(src.glob('*.png')))==1000 and len(list(tgt.glob('*.png')))==1000:
 print('FINAL DATA COUNTS 1000 1000',flush=True)
 raise SystemExit(0)
proxy='http://127.0.0.1:10808'; proxies={'http':proxy,'https':proxy}
r=requests.get('https://huggingface.co/api/datasets/merve/coco/tree/main/val2017?limit=1000',proxies=proxies,timeout=30); r.raise_for_status()
paths=[x['path'] for x in r.json() if x['path'].endswith('.jpg')]
paths.append('val2017/000000577539.jpg')
if len(paths)<1000: raise RuntimeError(f'COCO listing only {len(paths)}')
def fetch(item):
 i,p=item
 out=tgt/f'{i:04d}.png'
 if out.exists(): return i
 for attempt in range(5):
  try:
   r=requests.get('http://images.cocodataset.org/'+p,proxies=proxies,timeout=30); r.raise_for_status()
   Image.open(io.BytesIO(r.content)).convert('RGB').resize((224,224)).save(out)
   return i
  except Exception:
   if attempt==4: raise
with ThreadPoolExecutor(max_workers=12) as ex:
 for j,f in enumerate(as_completed([ex.submit(fetch,x) for x in enumerate(paths[:1000])]),start=1):
  f.result()
  if j%100==0: print('COCO targets',j,flush=True)
a=len(list(src.glob('*.png'))); b=len(list(tgt.glob('*.png')))
print('FINAL DATA COUNTS',a,b,flush=True)
if a!=1000 or b!=1000: raise RuntimeError('data incomplete')
