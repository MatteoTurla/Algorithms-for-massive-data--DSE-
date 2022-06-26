from torch.utils.data import Dataset as TorchDataset
from torchvision.io import read_image

class Dataset(TorchDataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.img_dir)

    def __getitem__(self, idx):
        img_path = self.img_dir[idx][0]
        image = read_image(str(img_path))
        label = self.img_dir[idx][1]
        if self.transform:
            image = self.transform(image)

        return image, label