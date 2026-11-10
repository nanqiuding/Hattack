"""Five-pair pixel-constrained DDIM inverse pilot with B16 CLIP guidance."""
import json
from pathlib import Path
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image
from diffusers import UNet2DConditionModel, AutoencoderKL, DDIMScheduler
from transformers import CLIPTextModel, CLIPTokenizer
from extractors import BACKBONE_MAP
from utils import to_tensor

torch.manual_seed(2026)
device='cuda'
model_dir='/root/autodl-tmp/sd_v1_4'
unet=UNet2DConditionModel.from_pretrained(model_dir,subfolder='unet',torch_dtype=torch.float16,local_files_only=True).to(device).eval()
vae=AutoencoderKL.from_pretrained(model_dir,subfolder='vae',torch_dtype=torch.float16,local_files_only=True).to(device).eval()
text_model=CLIPTextModel.from_pretrained(model_dir,subfolder='text_encoder',torch_dtype=torch.float16,local_files_only=True).to(device).eval()
tokenizer=CLIPTokenizer.from_pretrained(model_dir,subfolder='tokenizer',local_files_only=True)
scheduler=DDIMScheduler.from_pretrained(model_dir,subfolder='scheduler',local_files_only=True)
scheduler.set_timesteps(50)
timesteps=[int(t) for t in scheduler.timesteps if int(t)<=100]
clip=BACKBONE_MAP['B16']().eval().to(device).requires_grad_(False)
empty=tokenizer([''],padding='max_length',max_length=tokenizer.model_max_length,truncation=True,return_tensors='pt')
with torch.no_grad(): context=text_model(empty.input_ids.to(device))[0]
pre=transforms.Compose([transforms.Resize(224,interpolation=transforms.InterpolationMode.BICUBIC),transforms.CenterCrop(224)])
def load(path):
 with Image.open(path) as im: image=pre(im.convert('RGB'))
 return (to_tensor(image)/255).unsqueeze(0).to(device)
def encode(x):
 with torch.no_grad():
  return vae.encode((x*2-1).half()).latent_dist.mode()*vae.config.scaling_factor
def decode(z):
 with torch.no_grad():
  return ((vae.decode(z/vae.config.scaling_factor).sample.float()+1)/2).clamp(0,1)
def project(x,src):
 return x.clamp(0,1)
def feature(x):
 return clip.global_features(x*255)[0]
def attack(src,target,mode):
 zsrc=encode(src)
 noise=torch.randn_like(zsrc)
 first=timesteps[0]
 alpha=scheduler.alphas_cumprod[first].to(device)
 z=alpha.sqrt()*zsrc+(1-alpha).sqrt()*noise
 tgt_feat=feature(target).detach()
 history=[]
 for k,t in enumerate(timesteps):
  t_tensor=torch.tensor(t,device=device)
  with torch.no_grad(): eps=unet(z,t_tensor,encoder_hidden_states=context).sample
  a=scheduler.alphas_cumprod[t].to(device)
  x0=(z-(1-a).sqrt()*eps)/a.sqrt()
  image=decode(x0)
  before=float((image-src).abs().max())
  if mode=='guided':
   image=image.detach().requires_grad_(True)
   loss=1-F.cosine_similarity(feature(image),tgt_feat,dim=1).mean()
   grad=torch.autograd.grad(loss,image)[0]
   guided=image.detach()-0.5/255*grad.sign()
  else: guided=image
  projected=project(guided,src)
  z0=encode(projected)
  eps_new=(z-a.sqrt()*z0)/(1-a).sqrt()
  if k+1<len(timesteps):
   prev=timesteps[k+1]
   a_prev=scheduler.alphas_cumprod[prev].to(device)
   z=a_prev.sqrt()*z0+(1-a_prev).sqrt()*eps_new
  else: z=z0
  history.append({'t':t,'pre_projection_linf':before,
   'post_projection_linf':float((projected-src).abs().max()),
   'target_cosine':float(F.cosine_similarity(feature(projected),tgt_feat,dim=1).item())})
 return project(projected.detach(),src),history

out=Path('cache/analysis/diffusion_inverse_unconstrained')
out.mkdir(parents=True,exist_ok=True)
results=[]
for i in range(5):
 name=f'{i:04d}.png'
 src=load(Path('datas/nips2017_1000')/name)
 target=load(Path('datas/coco_val2017_1000')/name)
 target_feat=feature(target).detach()
 row={'id':name,'source_cosine':float(F.cosine_similarity(feature(src),target_feat,dim=1).item())}
 baseline=load(Path('cache/adv_images/SHARED_B16_NIPS100')/name)
 row['ifgsm_cosine']=float(F.cosine_similarity(feature(baseline),target_feat,dim=1).item())
 for mode in ('denoise','guided'):
  image,history=attack(src,target,mode)
  save_image(image,out/f'{mode}_{name}')
  row[mode+'_cosine']=float(F.cosine_similarity(feature(image),target_feat,dim=1).item())
  row[mode+'_linf']=float((image-src).abs().max().item())
  row[mode+'_history']=history
 results.append(row)
 (out/'results.json').write_text(json.dumps({'model':'CompVis/stable-diffusion-v1-4',
  'timesteps':timesteps,'epsilon':None,'guidance_step':0.5/255,'results':results},indent=2))
 print('RESULT',i,{k:v for k,v in row.items() if not k.endswith('history')},flush=True)
print('COMPLETE',out/'results.json',flush=True)
