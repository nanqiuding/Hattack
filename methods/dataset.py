from PIL import Image
from pathlib import Path
from torch.utils.data import Dataset

class ImageFolderDataset(Dataset):
    def __init__(self, img_folder: Path, num_samples, transform=None):
        if isinstance(img_folder, str):
            img_folder = Path(img_folder)
        self.transform = transform
        self.img_paths = sorted(
            img_folder.glob("*"),
            key=lambda x: int(x.stem)
        )
        self.num_samples = min(num_samples, len(self.img_paths))
    
    def __getitem__(self, index):
        img_path = self.img_paths[index]
        image = Image.open(img_path)
        if self.transform:
            image = self.transform(image)
        return {
            "path": str(img_path),
            "img": image
        }
    
    def __len__(self):
        return self.num_samples