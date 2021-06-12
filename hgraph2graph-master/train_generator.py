import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader
import re
import rdkit
import math, random, sys
import numpy as np
import argparse
import os
from tqdm.auto import tqdm
import time

from hgraph import *

lg = rdkit.RDLogger.logger() 
lg.setLevel(rdkit.RDLogger.CRITICAL)

parser = argparse.ArgumentParser()
parser.add_argument('--train', required=True)
parser.add_argument('--vocab', required=True)
parser.add_argument('--atom_vocab', default=common_atom_vocab)
parser.add_argument('--save_dir', required=True)
parser.add_argument('--load_model', default=None)
parser.add_argument('--seed', type=int, default=7)

parser.add_argument('--rnn_type', type=str, default='LSTM')
parser.add_argument('--hidden_size', type=int, default=250)
parser.add_argument('--embed_size', type=int, default=250)
parser.add_argument('--batch_size', type=int, default=50)
parser.add_argument('--latent_size', type=int, default=32)
parser.add_argument('--depthT', type=int, default=15)
parser.add_argument('--depthG', type=int, default=15)
parser.add_argument('--diterT', type=int, default=1)
parser.add_argument('--diterG', type=int, default=3)
parser.add_argument('--dropout', type=float, default=0.0)

parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--clip_norm', type=float, default=5.0)
parser.add_argument('--step_beta', type=float, default=0.001)
parser.add_argument('--max_beta', type=float, default=1.0)
parser.add_argument('--warmup', type=int, default=10000)
parser.add_argument('--kl_anneal_iter', type=int, default=2000)

parser.add_argument('--epoch', type=int, default=20)
parser.add_argument('--anneal_rate', type=float, default=0.9)
parser.add_argument('--anneal_iter', type=int, default=25000)
parser.add_argument('--print_iter', type=int, default=50)
parser.add_argument('--save_iter', type=int, default=5000)

args = parser.parse_args()
print(args)

def time_elapsed(start_time):
    elapsed = time.time() - start_time
    hours = int(elapsed / 3600)
    minutes = int(int(elapsed / 60) % 60)
    seconds = int(elapsed % 60)
    return hours, minutes, seconds

torch.manual_seed(args.seed)
random.seed(args.seed)

vocab = [x.strip("\r\n ").split() for x in open(args.vocab)] 
args.vocab = PairVocab(vocab)

model = HierVAE(args).cuda()
print("Model #Params: %dK" % (sum([x.nelement() for x in model.parameters()]) / 1000,))

for param in model.parameters():
    if param.dim() == 1:
        nn.init.constant_(param, 0)
    else:
        nn.init.xavier_normal_(param)

optimizer = optim.Adam(model.parameters(), lr=args.lr)
scheduler = lr_scheduler.ExponentialLR(optimizer, args.anneal_rate)

if args.load_model:
    print('continuing from checkpoint ' + args.load_model)
    model_state, optimizer_state, total_step, beta = torch.load(args.load_model)
    model.load_state_dict(model_state)
    optimizer.load_state_dict(optimizer_state)

    with open(os.path.join(args.save_dir, 'Training')) as f:
        last_record = f.readlines()[-1]
        count = int(re.findall('Epochs:(.*?)\|', last_record)[0].strip())
        t_final = [int(item) for item in re.findall('Time :(.*?)h(.*?)m(.*?)s', last_record)[0]]
        t_final = t_final[0] * 3600 + t_final[1] * 60 + t_final[2]   
else:
    total_step = beta = 0
    count = 0
    t_final = 0

param_norm = lambda m: math.sqrt(sum([p.norm().item() ** 2 for p in m.parameters()]))
grad_norm = lambda m: math.sqrt(sum([p.grad.norm().item() ** 2 for p in m.parameters() if p.grad is not None]))
  
progress = open(os.path.join(args.save_dir, 'Training'), 'a')

meters = np.zeros(6)
start_time = time.time() - t_final
for epoch in range(args.epoch):

    # dataset = DataFolder(args.train, args.batch_size)
    # for batch in tqdm(dataset):
    #     total_step += 1
    #     model.zero_grad()
    #     loss, kl_div, wacc, iacc, tacc, sacc = model(*batch, beta=beta)
    #     loss.backward()
    #     nn.utils.clip_grad_norm_(model.parameters(), args.clip_norm)
    #     optimizer.step()
    #     meters = meters + np.array([kl_div, loss.item(), wacc * 100, iacc * 100, tacc * 100, sacc * 100])
    #     if total_step % args.anneal_iter == 0:
    #         scheduler.step()
    #         print("learning rate: %.6f" % scheduler.get_lr()[0])
    #     if total_step >= args.warmup and total_step % args.kl_anneal_iter == 0:
    #         beta = min(args.max_beta, beta + args.step_beta)

    for fn in os.listdir(args.train):
        if fn != '.ipynb_checkpoints':
            dataset = DataFile(data_file=os.path.join(args.train, fn))
            for batch in tqdm(dataset):
                total_step += 1
                model.zero_grad()
                loss, kl_div, wacc, iacc, tacc, sacc = model(*batch, beta=beta)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), args.clip_norm)
                optimizer.step()
                meters = meters + np.array([kl_div, loss.item(), wacc * 100, iacc * 100, tacc * 100, sacc * 100])
                if total_step % args.anneal_iter == 0:
                    scheduler.step()
                    print("learning rate: %.6f" % scheduler.get_lr()[0])
                if total_step >= args.warmup and total_step % args.kl_anneal_iter == 0:
                    beta = min(args.max_beta, beta + args.step_beta)
    
        ckpt = (model.state_dict(), optimizer.state_dict(), total_step, beta)
        torch.save(ckpt, os.path.join(args.save_dir, f"model.ckpt.{epoch + 1 + count}"))

        meters = meters / len(dataset)
        del dataset
        print("[%d] Beta: %.3f, KL: %.2f, loss: %.3f, Word: %.2f, %.2f, Topo: %.2f, Assm: %.2f, PNorm: %.2f, GNorm: %.2f" % (total_step, beta, meters[0], meters[1], meters[2], meters[3], meters[4], meters[5], param_norm(model), grad_norm(model)))
        # Print training info
        hours, minutes, seconds = time_elapsed(start_time)
        progress.write(f'Loss: {meters[1]:7.3f} | KL:{meters[0]:7.3f} | Iterations: {total_step:6d} | Epochs: {epoch + 1 + count:5d} | Time : {hours:03d} h {minutes:02d} m {seconds:02d} s \n')
        progress.flush()
        sys.stdout.flush()
        meters *= 0
progress.close()
