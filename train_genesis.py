import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import torchvision
from torch.utils.data.distributed import DistributedSampler

import numpy as np
from sacred import Experiment, cli_option

from lib.sequential_datasets import ds as sds
from lib.sequential_datasets import SequentialHdF5Dataset
from lib.datasets_clevr import ds
from lib.datasets_clevr import StaticHdF5Dataset
from lib.model import net
from lib.model import GENESIS
from lib.geco import GECO

from tqdm import tqdm
from pathlib import Path
import shutil
import os 

import numpy as np
from collections import deque
import time

@cli_option('-r','--local_rank')
def local_rank_option(args, run):
    run.info['local_rank'] = args

ex = Experiment('GENESIS', ingredients=[ds, net, sds], additional_cli_options=[local_rank_option])

@ex.config
def cfg():
    training = {
            'DDP_port': 29500,
            'batch_size': 32,
            'num_workers': 8,
            'iters': 500000,
            'lr': 1e-4,
            'clip_grad_norm': True,
            'kl_beta': 0.5,
            'data_type': 'video',
            'mode': 'train',
            'geco_reconstruction_target': -23000,  # GECO C
            'geco_ema_alpha': 0.99,  # GECO EMA step parameter
            'geco_beta_stepsize': 1e-6,  # GECO Lagrange parameter beta
            'tensorboard_dir': './tb',
            'tensorboard_freq': 100,
            'tensorboard_delete_prev': False,
            'checkpoint_freq': 10000,
            'load_from_checkpoint': False,
            'checkpoint': '',
            'run_suffix': 'debug',
            'out_dir': ''
        }

def save_checkpoint(step, kl_beta, model, model_opt, filepath):
    state = {
        'step': step,
        'model': model.state_dict(),
        'model_opt': model_opt.state_dict(),
        'kl_beta': kl_beta
    }
    torch.save(state, filepath)

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
        writer.add_image('masks', mask_grid, step)
        writer.add_image('components', comp_grid, step)
        writer.add_image('reconstruction', reconstruction, step)

@ex.automain
def run(training, seed, _run):
    
    # maybe create
    run_dir = Path(training['out_dir'], 'runs')
    checkpoint_dir = Path(training['out_dir'], 'weights')
    tb_dir = Path(training['out_dir'], 'tb')
    
    for dir_ in [run_dir, checkpoint_dir, tb_dir]:
        if not dir_.exists():
            #dir_.mkdir()
            print(f'Create {dir_} before running!')
            exit(1)

    # Delete if exists
    tb_dbg = tb_dir / training['run_suffix']
    #if training['tensorboard_delete_prev'] and tb_dbg.exists():
    #    shutil.rmtree(tb_dbg)
    #    tb_dbg.mkdir()
    
    local_rank = 'cuda:{}'.format(_run.info['local_rank'])
    if local_rank == 'cuda:0':
        print(f'Creating SummarWriter! ({local_rank})')
        writer = SummaryWriter(tb_dbg)
    
    # Fix random seed
    print(f'setting random seed to {seed}')
    # Auto-set by sacred
    # torch.manual_seed(seed)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=False
    # Auto-set by sacred 
    #np.random.seed(seed)
        
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = str(training['DDP_port'])
    #num_gpus = torch.cuda.device_count()
    torch.distributed.init_process_group(backend='nccl')
    #device = torch.device(f'cuda:{local_rank}')
    torch.cuda.set_device(local_rank)
    
    model = GENESIS(batch_size=training['batch_size'])
    model = model.to(local_rank)

    model_geco = GECO(training['geco_reconstruction_target'], training['geco_ema_alpha'])

    model = torch.nn.parallel.DistributedDataParallel(model,
         device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
    model.train()

    # Optimization
    model_opt = torch.optim.Adam(model.parameters(), lr=training['lr'])
    
    if not training['load_from_checkpoint']:    
        step = 0 
        kl_beta = training['kl_beta']
        #model = torch.nn.DataParallel(model).to('cuda')
        checkpoint_step = 0
    else:
        checkpoint = checkpoint_dir / training['checkpoint']
        map_location = {'cuda:0': local_rank}
        state = torch.load(checkpoint, map_location=map_location)
        #num_gpus = torch.cuda.device_count()
        model.load_state_dict(state['model'])
        model_opt.load_state_dict(state['model_opt'])
    
        kl_beta = state['kl_beta']
        step = state['step']
        checkpoint_step = step

     # Data
    if training['data_type'] == 'image':
        tr_dataset = StaticHdF5Dataset(d_set=training['mode'])
    elif training['data_type'] == 'video':
        tr_dataset = SequentialHdF5Dataset(d_set=training['mode'])

    batch_size = training['batch_size']
    tr_sampler = DistributedSampler(dataset=tr_dataset)
    # N.b. Need to set shuffle=True to reshuffle every epoch
    tr_dataloader = torch.utils.data.DataLoader(tr_dataset,
             batch_size=batch_size, sampler=tr_sampler,
             num_workers=training['num_workers'], drop_last=True)
    
    max_iters = training['iters']

    #forward_queue = deque(maxlen=10000)
    #backward_queue = deque(maxlen=10000)
    print('Num parameters: {}'.format(sum(p.numel() for p in model.parameters())))
    epoch_idx = 0

    while step <= max_iters:
        tr_sampler.set_epoch(epoch_idx)

        for batch in tqdm(tr_dataloader):
            
            batch = batch['imgs'].to(local_rank)
            
            #start = time.time()
            out_dict = model(batch, model_geco, step, kl_beta)
            #forward_queue.append(time.time() - start)
            kl = out_dict['KL']
            nll = -out_dict['reconstruction']
            # descend the geco gradient
            model_opt.zero_grad()
            
            #start = time.time()
            (out_dict['loss']).backward()
            #backward_queue.append(time.time()-start)
            #for param in genesis.parameters():
            #    if param.grad is not None:
            #        if torch.isnan(param.grad).any() or param.grad.gt(1e6).any():
            #            import pdb; pdb.set_trace()
            #            print(param.grad)
            #            print(param.grad.shape)
            if training['clip_grad_norm']:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)

            model_opt.step()
                
            if step == model.module.geco_warm_start:
                model.module.geco_C_ema = model_geco.init_ema(model.module.geco_C_ema, nll)
            elif step > model.module.geco_warm_start:
                model.module.geco_C_ema = model_geco.update_ema(model.module.geco_C_ema, nll)
                model.module.geco_beta = model_geco.step_beta(model.module.geco_C_ema,
                        model.module.geco_beta, training['geco_beta_stepsize'])

            # logging
            if step % training['tensorboard_freq'] == 0 and local_rank == 'cuda:0':
                writer.add_scalar('train/geco_loss', out_dict['loss'], step)
                writer.add_scalar('train/elbo', out_dict['elbo'], step)
                writer.add_scalar('train/KL', kl, step)
                writer.add_scalar('train/NLL', nll, step)
                writer.add_scalar('train/geco_beta', model.module.geco_beta, step)
                writer.add_scalar('train/geco_C_ema', model.module.geco_C_ema, step)
                
                visualize_slots(writer, batch, out_dict['model_outs'], step)
                
                #print('forward time (ms) : {}'.format(np.mean(forward_queue) * 1000.))
                #print('backward time (ms): {}'.format(np.mean(backward_queue) * 1000.))

            if step > 0 and step % training['checkpoint_freq'] == 0 and local_rank == 'cuda:0':
                prefix = training['run_suffix']
                save_checkpoint(step, kl_beta, model, model_opt,
                       Path(checkpoint_dir, f'{prefix}-state-{step}.pth'))
            
            if step >= max_iters:
                step += 1
                break
            step += 1
        epoch_idx += 1

    if local_rank == 'cuda:0':
        writer.close()
