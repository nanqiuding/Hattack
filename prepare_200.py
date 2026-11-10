import csv, io, requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from PIL import Image
from threading import Lock
root=Path(__file__).parent
src=root/'datas/source_1000'; tgt=root/'datas/target_1000'
src.mkdir(parents=True,exist_ok=True); tgt.mkdir(parents=True,exist_ok=True)
proxy='http://127.0.0.1:10808'
proxies={'http':proxy,'https':proxy}
r=requests.get('https://raw.githubusercontent.com/cleverhans-lab/cleverhans/master/cleverhans_v3.1.0/examples/nips17_adversarial_competition/dataset/dev_dataset.csv',proxies=proxies,timeout=30); r.raise_for_status()
rows=list(csv.DictReader(io.StringIO(r.text)))
def source_fetch(row):
 try:
  r=requests.get(row['URL'],proxies=proxies,timeout=12); r.raise_for_status()
  im=Image.open(io.BytesIO(r.content)).convert('RGB'); w,h=im.size
  return im.crop((int(float(row['x1'])*w),int(float(row['y1'])*h),int(float(row['x2'])*w),int(float(row['y2'])*h))).resize((224,224))
 except Exception: return None
n=len(list(src.glob('*.png')))
with ThreadPoolExecutor(max_workers=16) as ex:
 for future in as_completed([ex.submit(source_fetch,row) for row in rows[:400]]):
  if n>=200: break
  im=future.result()
  if im is None: continue
  im.save(src/f'{n:04d}.png'); n+=1
  if n%20==0: print('source',n,flush=True)
print('source_count',n,flush=True)
if n<200: raise RuntimeError('fewer than 200 source images')
r=requests.get('https://huggingface.co/api/datasets/merve/coco/tree/main/val2017?limit=250',proxies=proxies,timeout=30); r.raise_for_status()
paths=[x['path'] for x in r.json() if x['path'].endswith('.jpg')]
if len(paths)<200: raise RuntimeError('fewer than 200 COCO paths')
def target_fetch(item):
 i,p=item; out=tgt/f'{i:04d}.png'
 if out.exists(): return i
 r=requests.get('http://images.cocodataset.org/'+p,proxies=proxies,timeout=30); r.raise_for_status()
 Image.open(io.BytesIO(r.content)).convert('RGB').resize((224,224)).save(out)
 return i
with ThreadPoolExecutor(max_workers=16) as ex:
 for future in as_completed([ex.submit(target_fetch,x) for x in enumerate(paths[:200])]):
  future.result()
print('target_count',len(list(tgt.glob('*.png'))),flush=True)
