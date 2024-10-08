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
from models import LinearClassifier, CosineClassifier, LinearProjection, CLIP_Text, Adapter

class MemeCLIP(pl.LightningModule):

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.acc = torchmetrics.Accuracy(task='multiclass', num_classes = cfg.num_classes)
        self.auroc = torchmetrics.AUROC(task='multiclass', num_classes = cfg.num_classes)
        self.f1 = torchmetrics.F1Score(task='multiclass', num_classes = cfg.num_classes, average='macro')

        self.clip_model, _ = clip.load(self.cfg.clip_variant, device="cuda", jit=False)
        self.clip_model.float()

        pre_output_input_dim = self.cfg.map_dim
        pre_output_layers = [nn.Dropout(p=cfg.drop_probs[1])]
        output_input_dim = pre_output_input_dim

        self.classifier = CosineClassifier(feat_dim = output_input_dim, num_classes=cfg.num_classes, dtype=self.clip_model.dtype)
        self.init_head_text_feat()
        self.text_encoder =  CLIP_Text(self.clip_model)
        self.img_adapter = Adapter(self.map_dim, 4).to(self.clip_model.dtype)
        self.text_adapter = Adapter(self.map_dim, 4).to(self.clip_model.dtype)
        self.clip_model.visual.proj = None

        for _, p in self.clip_model.named_parameters():
            p.requires_grad_(False)
        
        for name, param in self.classifier.named_parameters():
            param.requires_grad_(True)

        self.image_map = LinearProjection(self.cfg.unmapped_dim, self.cfg.map_dim,
                                          self.cfg.num_mapping_layers, self.cfg.drop_probs)
        self.text_map = LinearProjection(self.cfg.unmapped_dim, self.cfg.map_dim,
                                         self.cfg.num_mapping_layers, self.cfg.drop_probs)
        
        self.soft = nn.Softmax(dim=1)
            
        if self.cfg.num_pre_output_layers >= 1:
            pre_output_layers.extend(
                [nn.Linear(pre_output_input_dim, self.cfg.map_dim), nn.ReLU(), nn.Dropout(p=cfg.drop_probs[2])])
            output_input_dim = self.cfg.map_dim

        for _ in range(1, self.cfg.num_pre_output_layers):
            pre_output_layers.extend(
                [nn.Linear(self.cfg.map_dim, self.cfg.map_dim), nn.ReLU(), nn.Dropout(p=cfg.drop_probs[2])])

        self.pre_output = nn.Sequential(*pre_output_layers)
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss(reduction='mean')

    def forward(self, batch):
        pass
    
    def init_head_text_feat(self):

        print("Initialize head with text features")
        template = "a photo of a {}."
        prompts = [template.format(c.replace("_", " ")) for c in self.cfg.class_names]
        prompts = clip.tokenize([p for p in prompts], context_length=77, truncate=True).to(self.cfg.device)
        text_features = self.clip_model.encode_text(prompts)
        text_features = F.normalize(text_features, dim=-1)
        text_features = text_features @ self.clip_model.visual.proj.t()
        text_features = F.normalize(text_features, dim=-1)
        self.classifier.apply_weight(text_features)

    def common_step(self, batch):

        image_embeds = batch['image_features']
        text_embeds = batch['text_features']

        image_projection = self.image_map(image_embeds)
        txt_projection = self.text_map(text_embeds)

        image_features = self.img_adapter(image_projection)
        text_features = self.text_adapter(txt_projection)

        text_features = self.cfg.ratio  * text_features + (1 - self.cfg.ratio ) * txt_projection
        image_features = self.cfg.ratio  * image_features + (1 - self.cfg.ratio ) * image_projection

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        features = torch.mul(image_features, text_features)

        features_pre_output = self.pre_output(features)
        logits = self.classifier(features_pre_output).squeeze(dim=1) 
        preds_proxy = torch.sigmoid(logits)
        _ , preds = logits.data.max(1)

        output = {}
        output['loss'] = self.cross_entropy_loss(logits, batch['labels'])
        output['accuracy'] = self.acc(preds, batch['labels'])
        output['auroc'] = self.auroc(preds_proxy, batch['labels'])
        output['f1'] = self.f1(preds, batch['labels'])

        return output
    
    def training_step(self, batch, batch_idx):
        output = self.common_step(batch)

        total_loss = output['loss']

        self.log('train/total_loss', total_loss)
        self.log('train/loss', output['loss'])
        self.log('train/accuracy', output['accuracy'])
        self.log(f'train/auroc', output['auroc'], on_step=False, on_epoch=True, prog_bar=True)

        return total_loss

    def validation_step(self, batch, batch_idx):
        output = self.common_step(batch)

        total_loss = output['loss']

        self.log(f'val/total_loss', total_loss)
        self.log(f'val/loss', output['loss'])
        self.log(f'val/accuracy', output['accuracy'], on_step=False, on_epoch=True, prog_bar=True)
        self.log(f'val/auroc', output['auroc'], on_step=False, on_epoch=True, prog_bar=True)
        self.log(f'val/f1', output['f1'], on_step=False, on_epoch=True, prog_bar=True)


        return total_loss

    def test_step(self, batch, batch_idx):

        output = self.common_step(batch)
        self.log(f'test/accuracy', output['accuracy'])
        self.log(f'test/auroc', output['auroc'])
        self.log(f'test/f1', output['f1'])

        return output

    def on_train_epoch_end(self):
        self.acc.reset()
        self.auroc.reset()
        self.f1.reset()
        
    def on_validation_epoch_end(self):
        self.acc.reset()
        self.auroc.reset()
        self.f1.reset()

    def on_test_epoch_end(self):
        self.acc.reset()
        self.auroc.reset()
        self.f1.reset()

    def configure_optimizers(self):
        param_dicts = [
            {"params": [p for n, p in self.named_parameters() if p.requires_grad]}
        ]
        optimizer = torch.optim.AdamW(param_dicts, lr=self.cfg.lr, weight_decay=self.cfg.weight_decay)

        return optimizer

def create_model(cfg):
    model = MemeCLIP(cfg)
    return model
