import torch
import torch.nn as nn
import torchvision

import numpy as np
from sacred import Experiment
from datasets import ds
from datasets import StaticHdF5Dataset
from model import net
from model import GENESIS
from tqdm import tqdm
from pathlib import Path
from loss import image_batch_gmm_log_prob
import shutil

ex = Experiment('GENESIS-EVAL', ingredients=[ds, net])

@ex.config
def cfg():
    test = {
            'batch_size': 1,
            'num_workers': 0,
            'num_test_images': 5,
            'checkpoint_dir': './weights',
            'checkpoint': '',
            'metric': 'image_decomposition' 
        }

def restore_from_checkpoint(test, checkpoint):
    state = torch.load(checkpoint)
    genesis = GENESIS(batch_size=test['batch_size']).to('cuda')
    genesis.load_state_dict(state['model'])
    return genesis

def create_images_as_numpy(idx, out_dir, model_outs, K):
    """
    Store numpy arrays of results
    """
    masks = model_outs['pis'].exp()
    sub_images = model_outs['x_loc']

    images = []
    all_masks = []
    all_subis = []
    for i in range(K):
        images += [masks[i,0] * sub_images[i,0]]
        all_masks += [masks[i,0]]
        all_subis += [sub_images[i,0]]

    images = torch.stack(images)
    all_masks = torch.stack(all_masks)
    all_subis = torch.stack(all_subis)
    whole_image = images.sum(0)

    all_masks_grid = torchvision.utils.make_grid(all_masks, nrow=K)
    all_subis_grid = torchvision.utils.make_grid(all_subis, nrow=K)
    all_images_grid = torchvision.utils.make_grid(images, nrow=K)
    
    filepath = out_dir / f'whole_image_{idx}'
    np.save(filepath, whole_image.data.cpu().numpy())
    filepath = out_dir / f'all_images_{idx}'
    np.save(filepath, all_images_grid.data.cpu().numpy())
    filepath = out_dir / f'masks_{idx}'
    np.save(filepath, all_masks_grid.data.cpu().numpy())
    filepath = out_dir / f'sub_images_{idx}'
    np.save(filepath, all_subis_grid.data.cpu().numpy())

@ex.automain
def run(_run, test, seed):
    
    # Fix random seed
    print(f'setting random seed to {seed}')
    torch.manual_seed(seed)
    
    # Data
    te_dataset = StaticHdF5Dataset(d_set='test')
    te_dataloader = torch.utils.data.DataLoader(te_dataset, batch_size=1, shuffle=True, num_workers=test['num_workers'])
    genesis = restore_from_checkpoint(test, Path(test['checkpoint_dir'], test['checkpoint']))
   
    genesis.eval()
    
    out_dir = Path('results', test['checkpoint'])
    
    if not out_dir.exists():
        out_dir.mkdir(parents=True)
    
    with torch.no_grad():
        for i,batch in enumerate(tqdm(te_dataloader)):
            batch = batch.to('cuda')
            if test['metric'] == 'image_decomposition':
                model_outs = genesis(batch)
                create_images_as_numpy(i, out_dir, model_outs, genesis.K)
                if i == test['num_test_images']:
                    break
    print('done eval')
