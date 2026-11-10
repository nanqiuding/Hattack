from torchvision import transforms
from torch.utils.data import DataLoader
from utils import to_tensor, set_seed
from config import TomlLoader, Config
from methods import TRAINER_MAP, ImageFolderDataset

cfg = TomlLoader('config_mattack_1000.toml', Config)
transform = transforms.Compose([
    transforms.Lambda(lambda img: img.convert('RGB')),
    transforms.Resize(cfg.model.resize, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(cfg.model.resize),
    transforms.Lambda(to_tensor),
])
set_seed()
src = ImageFolderDataset(cfg.data.src_data_path, cfg.data.num_samples, transform)
tgt = ImageFolderDataset(cfg.data.tgt_data_path, cfg.data.num_samples, transform)
trainer = TRAINER_MAP[cfg.method](cfg)
for source, target in zip(DataLoader(src, shuffle=False), DataLoader(tgt, shuffle=False)):
    trainer.attack_image(source, target)
