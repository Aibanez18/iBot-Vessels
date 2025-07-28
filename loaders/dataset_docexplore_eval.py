import os
import numpy as np
from torchvision import transforms
from PIL import Image, ImageOps

import pandas as pd

from torch.utils.data import Dataset


class DocExploreEval(Dataset):
    def __init__(self, path_docexplore, transform=None):
        print(path_docexplore)
        self.df_docexplore = pd.read_pickle(path_docexplore)
        self.transform = transform

        self.change_path()

        # En caso de querer filtrar
        # self.df_docexplore = self.df_docexplore[self.df_docexplore['score'] >= 0.03]


    def __len__(self):
        return len(self.df_docexplore)
    
    def crop_image(self, df_idx):
        path_image = df_idx['filename'].iloc[0] if isinstance(df_idx['filename'], pd.Series) else df_idx['filename']

        if not os.path.exists(path_image):
            raise FileNotFoundError(f"No se encontró el archivo de imagen: {path_image}")

        img = Image.open(path_image).convert('RGB')
        x1, y1, x2, y2 = map(int, [df_idx['x1'], df_idx['y1'], df_idx['x2'], df_idx['y2']])
        return img.crop((x1, y1, x2, y2))
    
    def change_path(self, new_path_docexplore='/home/data/cstears/'):
        self.df_docexplore['filename'] = self.df_docexplore['filename'].apply(lambda x: x.replace(
            '/home/cloyola/datasets/DocExplore/', new_path_docexplore))
        self.df_docexplore['filename'] = self.df_docexplore['filename'].apply(lambda x: x.replace(
            '/home/cloyola/datasets/DocExplore/', new_path_docexplore))
    
    def transform_image(self, image, transform, pad=True):
        if pad:
            padded_image = ImageOps.pad(image, size=(224, 224))
            return transform(padded_image)
        else:
            return transform(image)
        
    def __getitem__(self, idx):
        df_idx = self.df_docexplore.iloc[idx]

        crop_anchor = self.crop_image(df_idx)
        crop_anchor_tensor = self.transform_image(crop_anchor, self.transform)

        filename_bbox = df_idx['doc_name'].replace("page", "") + f'_{df_idx["x1"]}-{df_idx["y1"]}-{df_idx["x2"]}-{df_idx["y2"]}'
        return crop_anchor_tensor, filename_bbox, df_idx['filename']
    

class DocExploreQueries(Dataset):
    def __init__(self, file, transform=None):
        self.data = []
        self.transform = transform

        with open(file, 'r') as f_images:
            lines = f_images.readlines()
            for line in lines:
                image_path, label = line.strip().split('\t')
                self.data.append((image_path, int(label)))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data_path, data_label = self.data[idx]
        data = Image.open(data_path).convert('RGB')
        data = ImageOps.pad(data, size=(224, 224))

        if self.transform:
            data = self.transform(data)

        return data, data_label, data_path