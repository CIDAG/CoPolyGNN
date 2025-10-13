import json
import os
import random
import shutil
from pathlib import Path

import numpy as np
import torch
import torch.multiprocessing as mp
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from arguments import arg_parse
from data import *
from kfold import *
from model import *


def seed_everything(seed=1234):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train_fold(
    fold_id, train_loader, val_loader, test_loader, tgt_tasks_id,
    has_features, std, device, seed, epochs, lr, output_path,
    batch_size, dropout, weight_decay, hidden_channels
):
    
    output = '{}_fold_{}_dropout_{}_hidden_channel_{}_bz_{}'.format(
        output_path, fold_id, dropout, hidden_channels, batch_size
    )    
    output = Path(output)
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)

    # k-fold split.
    k = 10
    for target_task in tgt_tasks_id:
        dataset = Polymers('../datasets/dataset{}'.format(target_task), target_task)
        folds = get_folds(n_samples=len(dataset), k=k, random_state=seed)
        train_fold_idx, val_fold_idx, test_fold_idx = get_indices(k=k)

        test_fold_idx = [folds[test_fold_idx[fold_id]] for fold_id in range(k)]
        val_fold_idx = [folds[val_fold_idx[fold_id]] for fold_id in range(k)]
        train_fold_idx = [np.hstack([folds[i] for i in train_fold_idx[fold_id]]) for fold_id in range(k)]

        test_index = test_fold_idx[fold_id]
        val_index = val_fold_idx[fold_id]
        train_index = train_fold_idx[fold_id]

        # standardizing experimental target
        _idx = np.hstack((train_index, val_index))
        mean = dataset[_idx].y.mean(dim=0, keepdim=True)
        std[target_task] = dataset[_idx].y.std(dim=0, keepdim=True)
        dataset.data.y = (dataset.data.y - mean) / std[target_task]
        if target_task in has_features:
            feats_mean = dataset[_idx].feats.mean(dim=0, keepdim=True)
            feats_std = dataset[_idx].feats.std(dim=0, keepdim=True)
            dataset.data.feats = (dataset.data.feats - feats_mean) / feats_std

        test_loader[target_task] = DataLoader(dataset[test_index], batch_size=batch_size, shuffle=False)
        val_loader[target_task] = DataLoader(dataset[val_index], batch_size=batch_size, shuffle=False)
        train_loader[target_task] = DataLoader(dataset[train_index], batch_size=batch_size, shuffle=True)

    # training process starts here
    num_features = dataset[0].x.shape[1]
    num_outputs = len(train_loader)
    model = Net(num_features, hidden_channels, num_outputs, dropout, has_features)
    model.to(device)
    std = std.to(device)

    task_num_iters = [len(train) for train in train_loader]
    task_ave_iter_list = sorted(task_num_iters)
    max_num_iter = task_ave_iter_list[-1]
    task_iter_train = {i: None for i in range(num_outputs)}
    task_count = {i: 0 for i in range(num_outputs)}

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    best_val = {target_task : None for target_task in tgt_tasks_id}
    norm_test_loss = {target_task : None for target_task in tgt_tasks_id}
    test_loss = {target_task : None for target_task in tgt_tasks_id}

    gradient_metrics = {}
    all_tasks = np.arange(44)
    step_interval = 10
    div = (max_num_iter // step_interval) + 1

    for epochId in tqdm(range(1, epochs + 1)):
        model.train()
        batch_train_loss = None
        batch_grad_metrics = None
        for step in tqdm(range(max_num_iter)):
            if step % step_interval == 0:
                # training step
                losses, task_gains = TAG(
                    task_count, task_iter_train, train_loader,
                    all_tasks, tgt_tasks_id, model, optimizer, device
                )
                
                if batch_grad_metrics is None:
                    batch_grad_metrics = {}
                    for combined_task, task_gain_map in task_gains.items():
                        if combined_task not in batch_grad_metrics:
                            batch_grad_metrics[combined_task] = {}
                        for task, gain in task_gain_map.items():
                            batch_grad_metrics[combined_task][task] = float(gain.cpu().numpy()) / div
                else:
                    for combined_task, task_gain_map in task_gains.items():
                        for task, gain in task_gain_map.items():
                          batch_grad_metrics[combined_task][task] += float(gain.cpu().numpy()) / div
                            
                if batch_train_loss is None:
                    batch_train_loss = {}
                    for task, loss in losses.items():
                        batch_train_loss[str(task)] = float(gain.cpu().numpy()) / div
                else:
                    for task, loss in losses.items():
                        batch_train_loss[str(task)] += float(gain.cpu().numpy()) / div

            else:
                for task_id in np.random.permutation(all_tasks):
                    model.zero_grad()
                    loss = train(task_id, task_count, task_iter_train, train_loader, model, device)
                    loss.backward()
                    optimizer.step()

            for target_task in tgt_tasks_id:
                if task_count[target_task] % len(train_loader[target_task]) == 0:
                    best_val[target_task], norm_test_loss[target_task], test_loss[target_task] = evaluation(
                        train_loader, val_loader, test_loader, target_task, 
                        model, epochId, lr, std, device, seed, output, 
                        best_val[target_task], norm_test_loss[target_task], test_loss[target_task]
                    )

        # Saving results per epoch
        for combined_task, task_gain_map in batch_grad_metrics.items():
            combined_task = '_'.join(map(str, combined_task))
            if combined_task not in gradient_metrics:
                gradient_metrics[combined_task] = []
            gradient_metrics[combined_task].append(task_gain_map)

        with open('{}/batch_train_loss.json'.format(output), 'a') as file:
            json.dump(batch_train_loss, file, indent=4)

    with open('{}/gradient_metrics.json'.format(output), 'w') as file:
        json.dump(gradient_metrics, file, indent=4)

if __name__ == '__main__':
    mp.set_start_method('spawn')
    args = arg_parse()
    seed = args.seed
    seed_everything(seed)
    device = torch.device(args.device)
    epochs = args.epochs
    lr = args.lr
    batch_size = args.batch_size
    dropout = args.dropout
    weight_decay = args.weight_decay
    hidden_channels = args.hidden_channels
    output_path = args.output

    has_features = [11, 31, 35, 36, 37]
    num_of_tasks = 44
    
    test_loader = [[] for _ in range(num_of_tasks)]
    val_loader = [[] for _ in range(num_of_tasks)]
    train_loader = [[] for _ in range(num_of_tasks)]
    
    aux_tasks = np.arange(35)
    tgt_tasks_id = np.arange(35, num_of_tasks)
    std = torch.zeros(num_of_tasks)
    num_of_aux_task = len(aux_tasks)

    # train/val splits.
    for _task_ in aux_tasks:
        dataset = Polymers('../datasets/dataset{}'.format(_task_), _task_)

        # Split datasets.
        mean = dataset.y.mean(dim=0, keepdim=True)
        std[_task_] = dataset.y.std(dim=0, keepdim=True)
        dataset.data.y = (dataset.data.y - mean) / std[_task_]
        
        if _task_ in has_features:
            feats_mean = dataset.feats.mean(dim=0, keepdim=True)
            feats_std = dataset.feats.std(dim=0, keepdim=True)
            dataset.data.feats = (dataset.data.feats - feats_mean) / feats_std

        nsamples = len(dataset) // 10
        val_loader[_task_] = DataLoader(dataset[:nsamples], batch_size=batch_size, shuffle=False)
        train_loader[_task_] = DataLoader(dataset[nsamples:], batch_size=batch_size, shuffle=True)

    # downloading everything before mp
    for _task_ in tgt_tasks_id:
        dataset = Polymers('../datasets/dataset{}'.format(_task_), _task_)
        
    processes = []
    for fold_id in [0, 1, 2, 3, 4]:
        p = mp.Process(target=train_fold, args=(
            fold_id, train_loader, val_loader, test_loader, tgt_tasks_id,
            has_features, std, device, seed, epochs, lr, output_path,
            batch_size, dropout, weight_decay, hidden_channels
        ))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
