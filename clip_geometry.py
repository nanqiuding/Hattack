"""Compare pairwise image geometry across the three attack CLIP encoders."""
import argparse, json
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from scipy.stats import spearmanr
from sklearn.manifold import TSNE
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from extractors import BACKBONE_MAP
from utils import to_tensor

def main():
 p=argparse.ArgumentParser()
 p.add_argument('--images',type=Path,default=Path('datas/nips2017_1000'))
 p.add_argument('--limit',type=int,default=200)
 p.add_argument('--output',type=Path,default=Path('cache/analysis/clip_geometry_200.json'))
 a=p.parse_args()
 paths=sorted(a.images.glob('*.png'))[:a.limit]
 if len(paths)!=a.limit: raise ValueError(f'expected {a.limit} images, found {len(paths)}')
 names=['B16','B32','Laion']
 embeddings={}
 for name in names:
  print('MODEL',name,flush=True)
  encoder=BACKBONE_MAP[name]().eval().cuda().requires_grad_(False)
  rows=[]
  with torch.inference_mode():
   for i,path in enumerate(paths):
    image=Image.open(path).convert('RGB').resize((224,224))
    x=to_tensor(image).unsqueeze(0).cuda()
    vector=encoder.global_features(x)[0].squeeze(0)
    rows.append(vector.cpu().float().numpy())
    if (i+1)%50==0: print(name,i+1,flush=True)
  embeddings[name]=np.stack(rows)
  del encoder
  torch.cuda.empty_cache()
 selected=[0,1,2,3,20,50,100,150]
 selected=[i for i in selected if i<len(paths)]
 colors=plt.get_cmap('tab10')
 plot_dir=a.output.parent/'tsne'
 plot_dir.mkdir(parents=True,exist_ok=True)
 for name in names:
  coordinates=TSNE(n_components=2,perplexity=30,metric='cosine',
                   init='random',random_state=2026).fit_transform(embeddings[name])
  fig,ax=plt.subplots(figsize=(8,7),dpi=170)
  ax.scatter(coordinates[:,0],coordinates[:,1],s=13,c='#b9c1ca',alpha=0.55)
  for j,index in enumerate(selected):
   x,y=coordinates[index]
   ax.scatter([x],[y],s=82,c=[colors(j)],edgecolors='black',linewidths=0.5,zorder=3)
   ax.annotate(paths[index].stem,(x,y),xytext=(5,5),textcoords='offset points',
               fontsize=8,color=colors(j),weight='bold')
  ax.set_title(f'{name}: t-SNE of {len(paths)} NIPS 2017 images')
  ax.set_xlabel('t-SNE 1 (model-specific)')
  ax.set_ylabel('t-SNE 2 (model-specific)')
  ax.grid(alpha=0.15)
  fig.tight_layout()
  fig.savefig(plot_dir/f'{name}_tsne.png')
  plt.close(fig)
  np.save(plot_dir/f'{name}_tsne.npy',coordinates)
  print('PLOT',plot_dir/f'{name}_tsne.png',flush=True)
 similarities={name:embeddings[name]@embeddings[name].T for name in names}
 tri=np.triu_indices(len(paths),1)
 rng=np.random.default_rng(2026)
 result={'image_count':len(paths),'image_ids':[x.stem for x in paths],
         'representation':'normalized final CLS token before projection',
         'pairs':{}}
 for i,left in enumerate(names):
  for right in names[i+1:]:
   s1,s2=similarities[left],similarities[right]
   observed=float(spearmanr(s1[tri],s2[tri]).statistic)
   perm=[]
   for _ in range(200):
    order=rng.permutation(len(paths))
    perm.append(float(spearmanr(s1[tri],s2[np.ix_(order,order)][tri]).statistic))
   k=10
   neighbors1=np.argsort(-s1,axis=1)[:,1:k+1]
   neighbors2=np.argsort(-s2,axis=1)[:,1:k+1]
   overlap=float(np.mean([len(set(x)&set(y))/k for x,y in zip(neighbors1,neighbors2)]))
   result['pairs'][f'{left}-{right}']={
    'spearman_pairwise_cosine':observed,
    'permutation_null_mean':float(np.mean(perm)),
    'permutation_p_one_sided':float((1+sum(x>=observed for x in perm))/(len(perm)+1)),
    'top10_neighbor_overlap':overlap,
    'random_overlap_expectation':k/(len(paths)-1),
   }
 a.output.parent.mkdir(parents=True,exist_ok=True)
 a.output.write_text(json.dumps(result,indent=2),encoding='utf-8')
 print(json.dumps(result['pairs'],indent=2),flush=True)
if __name__=='__main__': main()
