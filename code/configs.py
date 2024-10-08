import os
from yacs.config import CfgNode 

cfg = CfgNode()
cfg.root_dir = ''
cfg.img_folder = ''
cfg.info_file = ''
cfg.checkpoint_path = os.path.join(cfg.root_dir, 'checkpoints')
cfg.checkpoint_file = os.path.join(cfg.checkpoint_path,'model.ckpt')

cfg.clip_variant = "ViT-L/14"
cfg.dataset_name = 'Pride'
cfg.name = 'MemeCLIP' 
cfg.label = 'hate'
cfg.seed = 42
cfg.test_only = False
cfg.device = 'cuda'
cfg.gpus = [0]

if cfg.label =='hate':
    cfg.class_names = ['Benign Meme', 'Harmful Meme']
elif cfg.label == 'humour':
    cfg.class_names = ['No Humour', 'Humour']
elif cfg.label == 'target':
    cfg.class_names = ['No particular target', 'Individual', 'Community', 'Organization']
elif cfg.label == 'stance':
    cfg.class_names = ['Neutral', 'Support', 'Oppose']
  
cfg.batch_size = 16
cfg.image_size = 224
cfg.num_mapping_layers = 1
cfg.unmapped_dim = 768
cfg.map_dim = 1024
cfg.num_pre_output_layers = 1
cfg.drop_probs = [0.1, 0.4, 0.2]
cfg.lr = 1e-4
cfg.max_epochs = 10
cfg.weight_decay = 1e-4
cfg.num_classes = len(cfg.class_names)
cfg.scale = 30 
cfg.print_model = True
