import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
import albumentations as A

class SegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.images = os.listdir(image_dir)

        self.transform = A.Compose([
            A.Resize(256, 256),
            A.HorizontalFlip(p=0.5),
        ])

        # IMPORTANT: original mask values
        self.unique_vals = [0, 1, 2, 3, 27, 39]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])
        mask_path = os.path.join(self.mask_dir, self.images[index])

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(mask_path, 0)
        mask=torch.tensor(mask,dtype=torch.long)

        # REMAP MASK VALUES
        new_mask = np.zeros_like(mask)
        for new_val, old_val in enumerate(self.unique_vals):
            new_mask[mask == old_val] = new_val

        mask = new_mask

        augmented = self.transform(image=image, mask=mask)
        image = augmented["image"]
        mask = augmented["mask"]

        image = torch.tensor(image).permute(2, 0, 1).float() / 255.0
        mask = torch.tensor(mask).long()

        return image, mask