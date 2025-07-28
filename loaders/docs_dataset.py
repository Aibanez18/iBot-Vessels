# Copyright (c) ByteDance, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import random
import math
import numpy as np
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt

from PIL import Image, ImageOps
from torch.utils.data import Dataset

class HistoricalDocuments(Dataset):
    def __init__(self, pkl_path,
                 max_size=224, transform=None):
        try:
            self.df = pd.DataFrame(pd.read_pickle(pkl_path))

        except Exception as e:
            print(f"No se pudo cargar el archivo pickle: {e}")
            raise e 
        self.max_size = max_size
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        df_idx = self.df.iloc[idx][0]
        crop = Image.open(df_idx).convert('RGB')
        return self.transform_image(crop)
    
    def validation(self):
        for idx in tqdm(range(len(self.df))):
            df_idx = self.df.iloc[idx]

            x1, y1, x2, y2 = map(int, [df_idx['x1'], df_idx['y1'], df_idx['x2'], df_idx['y2']])
            if (x2 - x1) > 500 and (y2 - y1) > 500:
                tqdm.write(f"Se descartó el crop {df_idx['x1']},{df_idx['y1']},{df_idx['x2']},{df_idx['y2']} por ser muy grande")
                self.df.drop(idx, inplace=True)

    def transform_image(self, image):
        padded_image = ImageOps.pad(image, size=(self.max_size, self.max_size))
        return self.transform(padded_image)

    def save_sample(self, original, positive, negative):

        fig, ax = plt.subplots(1, 3)
        ax[0].imshow(original.permute(1, 2, 0))
        ax[0].set_title('Original')
        ax[0].axis('off')

        ax[1].imshow(positive.permute(1, 2, 0))
        ax[1].set_title('Positive')
        ax[1].axis('off')

        ax[2].imshow(negative.permute(1, 2, 0))
        ax[2].set_title('Negative')
        ax[2].axis('off')

        plt.show()

class HistoricalDocFolderMask(HistoricalDocuments):
    def __init__(self, *args, patch_size, pred_ratio, pred_ratio_var, pred_aspect_ratio, 
                 pred_shape='block', pred_start_epoch=0, **kwargs):
        super(HistoricalDocFolderMask, self).__init__(*args, **kwargs)
        self.psz = patch_size
        self.pred_ratio = pred_ratio[0] if isinstance(pred_ratio, list) and \
            len(pred_ratio) == 1 else pred_ratio
        self.pred_ratio_var = pred_ratio_var[0] if isinstance(pred_ratio_var, list) and \
            len(pred_ratio_var) == 1 else pred_ratio_var
        if isinstance(self.pred_ratio, list) and not isinstance(self.pred_ratio_var, list):
            self.pred_ratio_var = [self.pred_ratio_var] * len(self.pred_ratio)
        self.log_aspect_ratio = tuple(map(lambda x: math.log(x), pred_aspect_ratio))
        self.pred_shape = pred_shape
        self.pred_start_epoch = pred_start_epoch

    def get_pred_ratio(self):
        if hasattr(self, 'epoch') and self.epoch < self.pred_start_epoch:
            return 0

        if isinstance(self.pred_ratio, list):
            pred_ratio = []
            for prm, prv in zip(self.pred_ratio, self.pred_ratio_var):
                assert prm >= prv
                pr = random.uniform(prm - prv, prm + prv) if prv > 0 else prm
                pred_ratio.append(pr)
            pred_ratio = random.choice(pred_ratio)
        else:
            assert self.pred_ratio >= self.pred_ratio_var
            pred_ratio = random.uniform(self.pred_ratio - self.pred_ratio_var, self.pred_ratio + \
                self.pred_ratio_var) if self.pred_ratio_var > 0 else self.pred_ratio
        
        return pred_ratio

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __getitem__(self, index):
        output = super(HistoricalDocFolderMask, self).__getitem__(index)
                
        masks = []
        for img in output:
            try:
                H, W = img.shape[1] // self.psz, img.shape[2] // self.psz
            except:
                # skip non-image
                continue
            
            high = self.get_pred_ratio() * H * W
            
            if self.pred_shape == 'block':
                # following BEiT (https://arxiv.org/abs/2106.08254), see at
                # https://github.com/microsoft/unilm/blob/b94ec76c36f02fb2b0bf0dcb0b8554a2185173cd/beit/masking_generator.py#L55
                mask = np.zeros((H, W), dtype=bool)
                mask_count = 0
                while mask_count < high:
                    max_mask_patches = high - mask_count

                    delta = 0
                    for attempt in range(10):
                        low = (min(H, W) // 3) ** 2 
                        target_area = random.uniform(low, max_mask_patches)
                        aspect_ratio = math.exp(random.uniform(*self.log_aspect_ratio))
                        h = int(round(math.sqrt(target_area * aspect_ratio)))
                        w = int(round(math.sqrt(target_area / aspect_ratio)))
                        if w < W and h < H:
                            top = random.randint(0, H - h)
                            left = random.randint(0, W - w)

                            num_masked = mask[top: top + h, left: left + w].sum()
                            if 0 < h * w - num_masked <= max_mask_patches:
                                for i in range(top, top + h):
                                    for j in range(left, left + w):
                                        if mask[i, j] == 0:
                                            mask[i, j] = 1
                                            delta += 1

                        if delta > 0:
                            break

                    if delta == 0:
                        break
                    else:
                        mask_count += delta
            
            elif self.pred_shape == 'rand':
                mask = np.hstack([
                    np.zeros(H * W - int(high)),
                    np.ones(int(high)),
                ]).astype(bool)
                np.random.shuffle(mask)
                mask = mask.reshape(H, W)

            else:
                # no implementation
                assert False

            masks.append(mask)

        return (output, masks)