import os
import pandas as pd
import torch
import clip

from PIL import Image
from torch.utils.data import Dataset
from configs import cfg
from transformers import AutoProcessor, CLIPVisionModel


os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
torch.set_default_dtype(torch.float32)

from configs import cfg

class Custom_Dataset(Dataset):
    def __init__(self, cfg, root_folder, dataset, label, split='train', image_size=224, fast=True):
        super(Custom_Dataset, self).__init__()
        self.cfg = cfg
        self.root_folder = root_folder
        self.dataset = dataset
        self.split = split
        self.label = label

        self.image_size = image_size
        self.fast = fast

        self.info_file = cfg.info_file
        self.df = pd.read_csv(self.info_file)
        self.df = self.df[self.df['split'] == self.split].reset_index(drop=True)

        if self.label == 'target':
            self.df = self.df[self.df['hate'] == 1].reset_index(drop=True)

        float_cols = self.df.select_dtypes(float).columns
        self.df[float_cols] = self.df[float_cols].fillna(-1).astype('Int64')

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        if row['text'] == 'None':
            txt = 'null'
        else:
            txt = row['text']

        image_fn = row['name']
        image = Image.open(f"{self.cfg.img_folder}/{image_fn}").convert('RGB')\
            .resize((self.image_size, self.image_size))
        text = txt

        item = {
            'image': image,
            'text': text,
            'label': row[self.label],
            'idx_meme': row['name'],
            'origin_text': txt
        }

        return item

class Custom_Collator(object):
    def __init__(self, cfg):
        self.cfg = cfg
        self.clip_model, _ = clip.load(self.cfg.clip_variant, device="cuda", jit=False)
        _, self.clip_preprocess = clip.load(self.cfg.clip_variant, device="cuda", jit=False)
        self.clip_model.float().eval()

    def __call__(self, batch):
        labels = torch.LongTensor([item['label'] for item in batch])
        idx_memes = [item['idx_meme'] for item in batch]

        batch_new = {'labels': labels,
                     'idx_memes': idx_memes,
                     }
        
        image_embed_list = []
        text_embed_list = []

        for item in batch:

            pixel_values = self.clip_preprocess(item['image']).unsqueeze(0)
            text = clip.tokenize(item['text'], context_length=77, truncate=True)

            image_features, text_features = self.compute_CLIP_features_without_proj(self.clip_model,
                                                                    pixel_values.to(self.cfg.device),
                                                                    text.to(self.cfg.device))
            text_embed_list.append(text_features.cpu().detach())
            image_embed_list.append(image_features.cpu().detach())

        image_features = torch.cat([item for item in image_embed_list], dim=0)
        text_features = torch.cat([item for item in text_embed_list], dim=0)

        batch_new['image_features'] = image_features
        batch_new['text_features'] = text_features

        return batch_new
    
    def compute_CLIP_features_without_proj(self, clip_model, img_input, text_input):
        image_features = clip_model.visual(img_input.type(clip_model.dtype))

        x = clip_model.token_embedding(text_input).type(clip_model.dtype)
        x = x + clip_model.positional_embedding.type(clip_model.dtype)
        x = x.permute(1, 0, 2)
        x = clip_model.transformer(x)
        x = x.permute(1, 0, 2)
        x = clip_model.ln_final(x).type(clip_model.dtype)
        text_features = x[torch.arange(x.shape[0]), text_input.argmax(dim=-1)]

        return image_features, text_features


def load_dataset(cfg, split):
    dataset = Custom_Dataset(cfg = cfg, root_folder=cfg.root_dir, dataset=cfg.dataset_name, split=split,
                           image_size=cfg.image_size, label = cfg.label, fast=cfg.fast_process)

    return dataset
