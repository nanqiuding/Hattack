import os,time
from huggingface_hub import snapshot_download
os.environ['HTTPS_PROXY']='http://127.0.0.1:10808'
os.environ['HTTP_PROXY']='http://127.0.0.1:10808'
os.environ['HF_HUB_DISABLE_XET']='1'
for attempt in range(10):
 try:
  path=snapshot_download('CompVis/stable-diffusion-v1-4',
   local_dir='/root/autodl-tmp/sd_v1_4',
   allow_patterns=['model_index.json','unet/config.json','unet/diffusion_pytorch_model.safetensors',
    'vae/config.json','vae/diffusion_pytorch_model.safetensors',
    'text_encoder/config.json','text_encoder/model.safetensors',
    'tokenizer/*','scheduler/scheduler_config.json'],max_workers=2)
  print('COMPLETE',path,flush=True); break
 except Exception as e:
  print('ERROR',attempt,repr(e),flush=True); time.sleep(15)
else: raise RuntimeError('download failed')
