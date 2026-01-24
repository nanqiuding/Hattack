from torchvision import transforms
from torch.utils.data import DataLoader

from utils import to_tensor, set_seed
from config import TomlLoader, Config
from methods import TRAINER_MAP, ImageFolderDataset

cfg = TomlLoader("config.toml", Config)
data = cfg.data

transform_fn = transforms.Compose([
    transforms.Lambda(lambda img: img.convert("RGB")),
    transforms.Resize(
        cfg.model.resize,
        interpolation=transforms.InterpolationMode.BICUBIC,
    ),
    transforms.CenterCrop(cfg.model.resize),
    transforms.Lambda(lambda img: to_tensor(img)),
])

if __name__ == "__main__":
    set_seed()

    src_dataset = ImageFolderDataset(data.src_data_path, data.num_samples, transform_fn)
    tgt_dataset = ImageFolderDataset(data.tgt_data_path, data.num_samples, transform_fn)

    src_dataloader = DataLoader(src_dataset, shuffle=False)
    tgt_dataloader = DataLoader(tgt_dataset, shuffle=False)
    
    trainer = TRAINER_MAP[cfg.method](cfg)
    
    for src, tgt in zip(src_dataloader, tgt_dataloader):
        trainer.attack_image(src, tgt)