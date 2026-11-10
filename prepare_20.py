import csv, io, requests
from pathlib import Path
from PIL import Image
s=requests.Session()
root=Path(__file__).parent
src=root/'datas/source_1000'; tgt=root/'datas/target_1000'
src.mkdir(parents=True,exist_ok=True); tgt.mkdir(parents=True,exist_ok=True)
rows=list(csv.DictReader(io.StringIO(s.get('https://raw.githubusercontent.com/cleverhans-lab/cleverhans/master/cleverhans_v3.1.0/examples/nips17_adversarial_competition/dataset/dev_dataset.csv',timeout=30).text)))
n=len(list(src.glob('*.png')))
for row in rows:
 if n>=20: break
 try:
  r=s.get(row['URL'],timeout=8); r.raise_for_status()
  im=Image.open(io.BytesIO(r.content)).convert('RGB'); w,h=im.size
  im=im.crop((int(float(row['x1'])*w),int(float(row['y1'])*h),int(float(row['x2'])*w),int(float(row['y2'])*h))).resize((224,224))
  im.save(src/f'{n:04d}.png'); n+=1; print('source',n,flush=True)
 except Exception: pass
r=s.get('https://huggingface.co/api/datasets/merve/coco/tree/main/val2017?limit=25',timeout=30); r.raise_for_status()
paths=[x['path'] for x in r.json() if x['path'].endswith('.jpg')][:20]
for i,p in enumerate(paths):
 out=tgt/f'{i:04d}.png'
 if out.exists(): continue
 url='http://images.cocodataset.org/'+p
 r=s.get(url,timeout=30); r.raise_for_status()
 im=Image.open(io.BytesIO(r.content)).convert('RGB').resize((224,224))
 im.save(out); print('target',i+1,flush=True)
print('counts',len(list(src.glob('*.png'))),len(list(tgt.glob('*.png'))),flush=True)
