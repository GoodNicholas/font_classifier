# file: font_dataset.py
import glob, os, cv2, torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

class FontFieldDataset(Dataset):
    def __init__(self, root_dir, split='train', img_size=224):
        self.root_dir = root_dir
        self.img_size = img_size
        self.split = split

        # собираем все файлы
        self.samples = []
        for label, cls in enumerate(['printed', 'handwritten']):
            for path in glob.glob(os.path.join(root_dir, cls, '**', '*.png'), recursive=True):
                self.samples.append((path, label))

        # train/val/test разбиение по ratio
        ratios = {'train': 0.8, 'val': 0.1, 'test': 0.1}
        split_point1 = int(ratios['train'] * len(self.samples))
        split_point2 = int((ratios['train'] + ratios['val']) * len(self.samples))
        if split == 'train':
            self.samples = self.samples[:split_point1]
        elif split == 'val':
            self.samples = self.samples[split_point1:split_point2]
        else:
            self.samples = self.samples[split_point2:]

        # аугментации
        self.transform = A.Compose([
            A.RandomBrightnessContrast(0.15, 0.15, p=0.4),
            A.MotionBlur(blur_limit=3, p=0.3),
            A.GaussianBlur(blur_limit=3, p=0.2),
            A.OpticalDistortion(p=0.2),
            A.ShiftScaleRotate(shift_limit=0.02, scale_limit=0.08,
                               rotate_limit=3, border_mode=cv2.BORDER_CONSTANT, value=255, p=0.5),
            A.Resize(img_size, img_size),
            A.Normalize(mean=(0.5,), std=(0.5,)),
            ToTensorV2()
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = cv2.imread(path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.transform(image=image)['image']
        return image, torch.tensor(label, dtype=torch.long)
