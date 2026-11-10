import io, zipfile, requests
from pathlib import Path
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed
root=Path(__file__).parent
src=root/'datas/source_1000'; tgt=root/'datas/target_1000'
archive=zipfile.ZipFile('/root/autodl-tmp/source1000.zip')
names=sorted(n for n in archive.namelist() if n.startswith('dataset/images/') and n.endswith('.png'))
if len(names)!=1000: raise RuntimeError(f'archive has {len(names)} images')
for i,n in enumerate(names[:800],start=200):
 out=src/f'{i:04d}.png'
 if not out.exists():
  Image.open(io.BytesIO(archive.read(n))).convert('RGB').resize((224,224)).save(out)
 if (i+1)%100==0: print('source',i+1,flush=True)
proxy='http://127.0.0.1:10808'; proxies={'http':proxy,'https':proxy}
r=requests.get('https://huggingface.co/api/datasets/merve/coco/tree/main/val2017?limit=1000',proxies=proxies,timeout=30); r.raise_for_status()
paths=[x['path'] for x in r.json() if x['path'].endswith('.jpg')]
paths.append('val2017/000000581929.jpg')
if len(paths)<1000: raise RuntimeError(f'only {len(paths)} target paths')
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
  if j%100==0: print('targets checked',j,flush=True)
a=len(list(src.glob('*.png'))); b=len(list(tgt.glob('*.png')))
print('counts',a,b,flush=True)
if a<1000 or b<1000: raise RuntimeError('data incomplete')
