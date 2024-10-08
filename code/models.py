import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchmetrics
from clip import clip
from tqdm import tqdm
import os
from functools import partial
import torch.nn.functional as F
from transformers import AutoTokenizer
torch.set_default_dtype(torch.float32)

class Adapter(nn.Module):
    def __init__(self, c_in, reduction=4):
        super(Adapter, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(c_in, c_in // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c_in // reduction, c_in, bias=False),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.fc(x)
        return x
    
class Classifier(nn.Module):
    def __init__(self, feat_dim=768, num_classes=None, dtype=None):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_classes, feat_dim, dtype=dtype))
        self.weight.data.uniform_(-1, 1).renorm_(2, 0, 1e-5).mul_(1e5)

    @property
    def dtype(self):
        return self.weight.dtype

    def forward(self, x):
        raise NotImplementedError
        
    def apply_weight(self, weight):
        self.weight.data = weight.clone()

class CosineClassifier(Classifier):
    def __init__(self, feat_dim=None, num_classes=None, dtype=None, scale=30, **kwargs):
        super().__init__(feat_dim, num_classes, dtype)
        self.scale = scale
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(p=0.2)

    def forward(self, x):
        x = F.normalize(x, dim=-1)
        weight = F.normalize(self.weight, dim=-1)

        return F.linear(x, weight) * self.scale

class LinearClassifier(Classifier):
    def __init__(self, feat_dim=None, num_classes=None, dtype=None, **kwargs):
        super().__init__(feat_dim, num_classes, dtype)
        nn.init.kaiming_normal_(self.weight.data)
        self.bias = nn.Parameter(torch.zeros(num_classes, dtype=dtype))

    def forward(self, x):
        return F.linear(x, self.weight, self.bias)

class LinearProjection(nn.Module):
    def __init__(self, input_dim, output_dim, num_layers, drop_probs):
        super(LinearProjection, self).__init__()

        map_layers = [nn.Linear(input_dim, output_dim),
                      nn.Dropout(p=drop_probs[0])]

        for _ in range(1, num_layers):
            map_layers.extend(
                [nn.ReLU(), nn.Linear(output_dim, output_dim), nn.Dropout(p=drop_probs[0])])

        self.proj = nn.Sequential(*map_layers)

    def __call__(self, *args, **kwargs):
        return super().__call__(*args, **kwargs)

    def forward(self, x):
        return self.proj(x)
    
class CLIP_Text(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.token_embedding = clip_model.token_embedding
        self.positional_embedding = clip_model.positional_embedding
        self.transformer = clip_model.transformer
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, text):
        x = self.token_embedding(text).to(self.dtype)  
        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2).type(torch.float32)  
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  
        x = self.ln_final(x).to(self.dtype)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection.to(self.dtype)

        return x
