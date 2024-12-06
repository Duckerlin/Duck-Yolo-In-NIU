import os
from PIL import Image
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, data_path, transform=None):
        self.data_path = data_path
        self.transform = transform
        self.image_filenames = [f for f in os.listdir(data_path) if f.endswith('.jpg')]

    def __getitem__(self, index):
        image = Image.open(os.path.join(self.data_path, self.image_filenames[index])).convert('RGB')
        mask_path = os.path.join(self.data_path, self.image_filenames[index].replace('.jpg', '_segmentation.png'))
        mask = Image.open(mask_path).convert('L')

        if image.mode == 'L':
            image = image.convert('RGB')

        if mask.mode == 'L':
            mask = mask.convert('RGB')  # 或者保持单通道

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        return image, mask

    def __len__(self):
        return len(self.image_filenames)
