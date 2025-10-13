import shutil
import json
import random
import os

import numpy as np
import torch
import torch.multiprocessing as mp
from pathlib import Path
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from arguments import arg_parse
from data import *
from model import *
from kfold import *


def seed_everything(seed=1234):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_fold(
    fold_id, max_num_iter, aux_tasks_per_fold, has_features, device, epochs, lr, 
    batch_size, dropout, weight_decay, hidden_channels, target_task, seed, best_epoch
):

    k = 10
    auxiliary_tasks = np.sort(aux_tasks_per_fold[fold_id])
    num_of_aux_task = len(auxiliary_tasks)
    aux_tasks = [t for t in auxiliary_tasks if t < 35]
    tgt_tasks = [t for t in auxiliary_tasks if t >= 35] + [target_task]
    tgt_to_idx = {i: idx for idx, i in enumerate(aux_tasks + tgt_tasks)}
    all_tasks = np.empty(num_of_aux_task * 2, dtype=int)
    all_tasks[1::2] = np.full(num_of_aux_task, tgt_to_idx[target_task])
    idx_auxiliary_tasks = [tgt_to_idx[i] for i in auxiliary_tasks]

    output = 'fold_{}_target_task_{}'.format(fold_id, target_task)
    output = Path(output)
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)

    test_loader = [[] for _ in range(num_of_aux_task + 1)]
    val_loader = [[] for _ in range(num_of_aux_task + 1)]
    train_loader = [[] for _ in range(num_of_aux_task + 1)]
    std = torch.zeros(num_of_aux_task + 1)

    # train/val splits.
    for idx, i in enumerate(aux_tasks):
        dataset = Polymers('../../datasets/dataset{}'.format(i), i)

        # Split datasets.
        nsamples = len(dataset) // 10
        mean = dataset.y.mean(dim=0, keepdim=True)
        std[idx] = dataset.y.std(dim=0, keepdim=True)
        dataset.data.y = (dataset.data.y - mean) / std[idx]

        if i in has_features:
            feats_mean = dataset.feats.mean(dim=0, keepdim=True)
            feats_std = dataset.feats.std(dim=0, keepdim=True)
            dataset.data.feats = (dataset.data.feats - feats_mean) / feats_std

        val_loader[idx] = DataLoader(dataset[:nsamples], batch_size=batch_size, shuffle=False)
        train_loader[idx] = DataLoader(dataset[nsamples:], batch_size=batch_size, shuffle=True)

    # k-fold split.
    for idx, _target_task in enumerate(tgt_tasks, start=len(aux_tasks)):
        dataset = Polymers('../../datasets/dataset{}'.format(_target_task), _target_task)

        folds = get_folds(n_samples=len(dataset), k=k, random_state=seed)
        train_fold_idx, val_fold_idx, test_fold_idx = get_indices(k=k)

        test_fold_idx = [folds[test_fold_idx[fold_id]] for fold_id in range(k)]
        val_fold_idx = [folds[val_fold_idx[fold_id]] for fold_id in range(k)]
        train_fold_idx = [np.hstack([folds[i] for i in train_fold_idx[fold_id]]) for fold_id in range(k)]

        test_index = test_fold_idx[fold_id] # return the training idx for this specific task (i) and fold (fold_id)
        val_index = val_fold_idx[fold_id] # return the training idx
        train_index = train_fold_idx[fold_id] # return the training idx

        # standardizing experimental target
        _idx = np.hstack((train_index, val_index))
        mean = dataset[_idx].y.mean(dim=0, keepdim=True)
        std[idx] = dataset[_idx].y.std(dim=0, keepdim=True)
        dataset.data.y = (dataset.data.y - mean) / std[idx]

        if _target_task in has_features:
            feats_mean = dataset[_idx].feats.mean(dim=0, keepdim=True)
            feats_std = dataset[_idx].feats.std(dim=0, keepdim=True)
            dataset.data.feats = (dataset.data.feats - feats_mean) / feats_std

        test_loader[idx] = DataLoader(dataset[test_index], batch_size=batch_size, shuffle=False)
        val_loader[idx] = DataLoader(dataset[val_index], batch_size=batch_size, shuffle=False)
        train_loader[idx] = DataLoader(dataset[train_index], batch_size=batch_size, shuffle=True)

    # Initialize the model, optimizer, and other components
    has_features = [tgt_to_idx.get(i, None) for i in has_features]
    num_features = dataset[0].x.shape[1]
    num_outputs = len(train_loader)
    model = Net(num_features, hidden_channels, num_outputs, dropout, has_features)
    model.to(device)
    std = std.to(device)

    weights = torch.load('pre-training_fold_{}/best_model_epoch_{}_task_{}.pth'.format(fold_id, best_epoch, target_task), weights_only=True)
    state_dict = {k: v for k, v in weights.items() if 'output' not in k}

    for target_layer in tgt_to_idx:
        matching_weights = [weight_key for weight_key in weights if f'outputs.{target_layer}.' in weight_key]
        for weight_name in matching_weights:
            updated_key = weight_name.replace(f'outputs.{target_layer}.', f'outputs.{tgt_to_idx[target_layer]}.')
            state_dict[updated_key] = weights[weight_name]

    model.load_state_dict(state_dict)

    task_iter_train = {i: None for i in range(num_outputs)}
    task_count = {i: 0 for i in range(num_outputs)}

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val = None
    norm_test_loss = None
    test_loss = None

    # Training loop
    for epochId in range(1, epochs + 1):
        for step in tqdm(range(max_num_iter)):
            all_tasks = list(np.random.permutation(idx_auxiliary_tasks)) + [tgt_to_idx[target_task]]
            for task_id in all_tasks:
                    model.zero_grad()
                    loss = train(task_id, task_count, task_iter_train, train_loader, model, device)
                    loss.backward()
                    optimizer.step()
                    if (task_id == tgt_to_idx[target_task]) and (task_count[task_id] % len(train_loader[task_id]) == 0):
                        best_val, norm_test_loss, test_loss = evaluation(
                            train_loader, val_loader, test_loader, tgt_to_idx[target_task],
                            model, epochId, lr, std, device, seed, output,
                            best_val, norm_test_loss, test_loss
                        )


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

    # Maximum number of iterations per epoch
    max_num_iter = None

    # Selected auxiliary tasks per target task and fold (from previous pre-training phase)
    # {
    #     target_task_id: [
    #         [aux_task_1, aux_task_2, ...],   # fold 0
    #         [aux_task_1, aux_task_2, ...],   # fold 1
    #         ...
    #     ],
    #     ...
    # }
    # Example:
    # selected_tasks = {
    #     35: [[38, 0, 34, 40], [38, 0, 34, 22], ...],
    #     36: [[0, 34, 14], [13, 29, 39], ...],
    # }
    selected_tasks = {}

    # Best epoch number for each target task and fold (from previous pre-training phase)
    # Structure:
    # {
    #     target_task_id: [best_epoch_fold0, best_epoch_fold1, ..., best_epoch_fold9],
    #     ...
    # }
    # Example:
    # all_best_epochs = {
    #     35: [3, 3, 8, 4, 10, 6, 4, 10, 4, 6],
    #     36: [9, 5, 9, 5, 10, 10, 9, 10, 3, 10],
    # }
    all_best_epochs = {}

    for target_task in [35, 36, 37, 38, 39, 40, 41, 42, 43]:

        has_features = [11, 31, 35, 36, 37]
        aux_tasks_per_fold = selected_tasks[target_task]

        processes = []
        for fold_id in range(10):
            p = mp.Process(target=train_fold, args=(fold_id, max_num_iter, aux_tasks_per_fold, has_features,
                device, epochs, lr, batch_size, dropout, weight_decay, hidden_channels, target_task, seed, all_best_epochs[target_task][fold_id])
            )
            p.start()
            processes.append(p)

        for p in processes:
            p.join()
