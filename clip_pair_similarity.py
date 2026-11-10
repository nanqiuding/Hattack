"""Compare source-target cosine scores across three CLIP final-CLS encoders."""
import json
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from scipy.stats import spearmanr, pearsonr
from torchvision import transforms
from extractors import BACKBONE_MAP
from utils import to_tensor

names=('B16','B32','Laion')
preprocess=transforms.Compose([
 transforms.Resize(224,interpolation=transforms.InterpolationMode.BICUBIC),
 transforms.CenterCrop(224),
])
source=Path('datas/nips2017_1000')
target=Path('datas/coco_val2017_1000')
ids=[f'{i:04d}' for i in range(1000)]
scores={}
for name in names:
 model=BACKBONE_MAP[name]().eval().cuda().requires_grad_(False)
 values=[]
 with torch.inference_mode():
  for k,image_id in enumerate(ids,1):
   features=[]
   for folder in (source,target):
    with Image.open(folder/f'{image_id}.png') as opened:
     image=preprocess(opened.convert('RGB'))
    x=to_tensor(image).unsqueeze(0).cuda()
    features.append(model.global_features(x)[0])
   values.append(torch.nn.functional.cosine_similarity(features[0],features[1],dim=1).item())
   if k%200==0: print(name,k,flush=True)
 scores[name]=np.array(values)
 del model
 torch.cuda.empty_cache()
result={'pairs':'NIPS2017 source image and same-ID COCO target image',
        'features':'L2-normalized final CLS before CLIP projection',
        'count':len(ids),'models':{},'comparisons':{},'largest_disagreements':[]}
for name in names:
 a=scores[name]
 result['models'][name]={'mean':float(a.mean()),'std':float(a.std()),
   'median':float(np.median(a)),'p10':float(np.percentile(a,10)),
   'p90':float(np.percentile(a,90)),'min':float(a.min()),'max':float(a.max())}
for i,left in enumerate(names):
 for right in names[i+1:]:
  a,b=scores[left],scores[right]
  d=np.abs(a-b)
  result['comparisons'][f'{left}-{right}']={
   'mean_absolute_difference':float(d.mean()),
   'median_absolute_difference':float(np.median(d)),
   'p90_absolute_difference':float(np.percentile(d,90)),
   'fraction_difference_gt_0.1':float(np.mean(d>0.1)),
   'fraction_difference_gt_0.2':float(np.mean(d>0.2)),
   'spearman_across_pairs':float(spearmanr(a,b).statistic),
   'pearson_across_pairs':float(pearsonr(a,b).statistic)}
spread=np.stack([scores[n] for n in names]).max(axis=0)-np.stack([scores[n] for n in names]).min(axis=0)
result['overall_spread']={'mean':float(spread.mean()),'median':float(np.median(spread)),
 'p90':float(np.percentile(spread,90)),'fraction_gt_0.1':float(np.mean(spread>0.1)),
 'fraction_gt_0.2':float(np.mean(spread>0.2))}
for index in np.argsort(-spread)[:10]:
 result['largest_disagreements'].append({'id':ids[index],
  'spread':float(spread[index]),**{name:float(scores[name][index]) for name in names}})
result['per_pair']=[{'id':ids[i],**{name:float(scores[name][i]) for name in names},
                    'spread':float(spread[i])} for i in range(len(ids))]
out=Path('cache/analysis/clip_pair_similarity_1000.json')
out.parent.mkdir(parents=True,exist_ok=True)
out.write_text(json.dumps(result,indent=2))
print(json.dumps({k:v for k,v in result.items() if k!='per_pair'},indent=2),flush=True)
print('OUTPUT',out,flush=True)
