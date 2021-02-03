import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import torchvision

import numpy as np
from sacred import Experiment
#from datasets import ds
#from datasets import StaticHdF5Dataset
from datasets_clevr import ds
from datasets_clevr import StaticHdF5Dataset
from model import net
from model import GENESIS
from loss import genesis_loss
from tqdm import tqdm
from pathlib import Path
import shutil
from utils import GECO

import numpy as np
from collections import deque
import time

ex = Experiment('GENESIS', ingredients=[ds, net])

@ex.config
def cfg():
    training = {
            'batch_size': 32,
            'num_workers': 8,
            'iters': 500000,
            'lr': 10e-5,
            'GECO_EMA_alpha': 0.99,
            'beta_step_size_1': 1e-5,
            'beta_warm_start': 1000,
            'beta_step_freq': 10,
            'reconstruction_target': 27500,
            'tensorboard_dir': './tb',
            'tensorboard_freq': 100,
            'tensorboard_delete_prev': False,
            'checkpoint_freq': 10000,
            'load_from_checkpoint': False,
            'checkpoint_dir': './weights',
            'checkpoint': '',
            'debug': True,
            'run_suffix': 'debug'
        }

def save_checkpoint(step, model, model_opt, geco_ema, filepath):
    state = {
        'step': step,
        'model': model.state_dict(),
        'model_opt': model_opt.state_dict(),
        'geco_ema': geco_ema.state_dict(),
        'geco_C_ema': geco_ema.C_ema
    }
    torch.save(state, filepath)

def restore_from_checkpoint(training, checkpoint):
    state = torch.load(checkpoint)

    genesis = GENESIS(batch_size=training['batch_size']).to('cuda')
    genesis.load_state_dict(state['model'])

    geco_ema = GECO(training['reconstruction_target'], 0, training['GECO_EMA_alpha']).to('cuda')
    geco_ema.load_state_dict(state['geco_ema'])
    geco_ema.C_ema = state['geco_C_ema'].to('cuda')

    genesis_opt = torch.optim.Adam(genesis.parameters(), lr=training['lr'])
    genesis_opt.load_state_dict(state['model_opt'])
    
    step = state['step']
    return genesis, genesis_opt, geco_ema, step

def visualize_slots(writer, batch_data, model_outs, step):
    """
    Render images for each mask and slot reconstruction,
    as well as mask*slot 
    """

    with torch.no_grad():
        # [batch_size * K, 3, H, W]
        batch_size, C, H, W = batch_data.shape
        
        x_loc = model_outs['x_loc'].view(-1, batch_size, C, H, W)
        K, _, _, _, _ = x_loc.shape
        x_loc_np = x_loc.data.cpu().numpy()
        # [batch_size * K, 1, H, W]
        # from log scale via exp()
        pis = model_outs['pis'].exp().view(-1, batch_size, 1, H, W)
        pis_np = pis.data.cpu().numpy()

        masks = []
        contents = []
        reconstruction = np.zeros((C,H,W))
        for i in range(K):
            contents += [x_loc_np[i,0]]
            masks += [pis_np[i,0]]
            reconstruction += (masks[-1] * contents[-1])
        img = batch_data[0].data.cpu().numpy()

        comp_grid = torchvision.utils.make_grid(x_loc[:,0])
        mask_grid = torchvision.utils.make_grid(pis[:,0])
        
        writer.add_image('image', img, step)
        writer.add_image('masks', mask_grid.data.cpu().numpy(), step)
        writer.add_image('components', comp_grid.data.cpu().numpy(), step)
        writer.add_image('reconstruction', reconstruction, step)

@ex.automain
def run(training, seed):
    
    # 1. Set up logging and viz
    if training['debug']:
        # Delete if exists
        tb_dbg = Path(training['tensorboard_dir'], training['run_suffix'])
        if training['tensorboard_delete_prev'] and tb_dbg.exists():
            shutil.rmtree(tb_dbg)
            tb_dbg.mkdir()
        
        writer = SummaryWriter(tb_dbg, flush_secs=15)
    else:
        # TODO
        pass

    # Fix random seed
    print(f'setting random seed to {seed}')
    torch.manual_seed(seed)
    
    # Data
    tr_dataset = StaticHdF5Dataset(d_set='train')
    tr_dataloader = torch.utils.data.DataLoader(tr_dataset, batch_size=training['batch_size'], shuffle=True, num_workers=training['num_workers'])
    reconstruction_target = training['reconstruction_target']
    
    if not training['load_from_checkpoint']:
        # Models
        genesis = GENESIS(batch_size=training['batch_size']).to('cuda')

        # Optimization
        genesis_opt = torch.optim.Adam(genesis.parameters(), lr=training['lr'])
        geco_ema = None        
        step = 0
    else:
        genesis, genesis_opt, geco_ema, step = \
                restore_from_checkpoint(training, Path(training['checkpoint_dir'], training['checkpoint']))
        geco_ema.C = reconstruction_target
    
    forward_queue = deque(maxlen=10000)
    backward_queue = deque(maxlen=10000)

    while step <= training['iters']:
        for batch in tqdm(tr_dataloader):
            
            batch = batch['imgs'].to('cuda')
            
            start = time.time()
            out_dict = genesis_loss(batch, genesis, geco_ema, reconstruction_target)
            forward_queue.append(time.time() - start)

            # descend the geco gradient
            genesis_opt.zero_grad()
            
            start = time.time()
            (out_dict['loss']).backward()
            backward_queue.append(time.time()-start)
            #for param in genesis.parameters():
            #    if param.grad is not None:
            #        if torch.isnan(param.grad).any() or param.grad.gt(1e6).any():
            #            import pdb; pdb.set_trace()
            #            print(param.grad)
            #            print(param.grad.shape)
            
            torch.nn.utils.clip_grad_norm_(genesis.parameters(), max_norm=1e3)

            genesis_opt.step()
            
            if geco_ema is None:
                geco_ema = GECO(reconstruction_target, out_dict['reconstruction'], 
                        training['GECO_EMA_alpha']).to('cuda')

            # update beta
            if step > training['beta_warm_start'] and step % training['beta_step_freq'] == 0:
                geco_ema.step_beta(training['beta_step_size_1'])

            # logging
            if step % training['tensorboard_freq'] == 0:
                writer.add_scalar('train/geco_loss', out_dict['loss'].data.cpu().numpy(), step)
                writer.add_scalar('train/elbo', out_dict['elbo'].data.cpu().numpy(), step)
                writer.add_scalar('train/KL', out_dict['KL'].data.cpu().numpy(), step)
                writer.add_scalar('train/reconstruction', out_dict['reconstruction'].data.cpu().numpy(), step)
                writer.add_scalar('train/geco_beta', geco_ema.beta.data.cpu().numpy(), step)
                writer.add_scalar('train/geco_C_ema', geco_ema.C_ema.data.cpu().numpy(), step)
                writer.add_scalar('train/reconstruction_target', reconstruction_target, step)
                visualize_slots(writer, batch, out_dict['model_outs'], step)
                
                print('forward time (ms) : {}'.format(np.mean(forward_queue) * 1000.))
                print('backward time (ms): {}'.format(np.mean(backward_queue) * 1000.))

            if step > 0 and step % training['checkpoint_freq'] == 0:
                prefix = training['run_suffix']
                save_checkpoint(step, genesis, genesis_opt, geco_ema,
                       Path(training['checkpoint_dir'], f'{prefix}-state-{step}.pth'))
            
            if step == training['iters']:
                break
            step += 1
