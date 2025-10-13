from itertools import combinations
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINConv, MLP
from torch_geometric.utils import scatter, softmax
from tqdm import tqdm


class Net(torch.nn.Module):
    """
    A neural network model based on graph convolution and an attention mechanism for polymer prediction.

    Attributes:
        hidden_channels (int): Number of hidden units in each graph convolution layer.
        encoder (torch.nn.ModuleList): A list of graph convolutional layers.
        fc_q (torch.nn.Linear): Linear transformation for the query vector in the attention mechanism.
        fc_k (torch.nn.Linear): Linear transformation for the key vector in the attention mechanism.
        fc_v (torch.nn.Linear): Linear transformation for the value vector in the attention mechanism.
        dropout (torch.nn.Dropout): Dropout layer to prevent overfitting.
        outputs (torch.nn.ModuleList): A list of sequential modules for predicting outputs for each task.
        has_features (list): List indicating which tasks have additional features.
    """

    def __init__(self, input_feat: int, hidden_channels: int, out_channels: int, dropout: float, has_features: List[int]):
        """
        Initializes the Net class with specific parameters for the graph neural network and attention mechanism.
        
        :param input_feat: Number of input features.
        :param hidden_channels: Size of the hidden layers.
        :param out_channels: Number of different tasks for which outputs are generated.
        :param dropout: Dropout rate used in the dropout layers.
        :param has_features: List indicating which tasks have additional features.
        """
        super(Net, self).__init__()
        self.hidden_channels = hidden_channels
        self.register_buffer('scale', torch.tensor(hidden_channels ** -0.5))

        self.encoder = torch.nn.ModuleList()
        for _ in range(3):
            mlp = MLP([input_feat, hidden_channels, hidden_channels], norm=None)
            self.encoder.append(GINConv(nn=mlp, train_eps=False))
            input_feat = hidden_channels

        self.fc_q = torch.nn.Linear(hidden_channels, hidden_channels)
        self.fc_k = torch.nn.Linear(hidden_channels, hidden_channels)
        self.fc_v = torch.nn.Linear(hidden_channels, hidden_channels)

        self.dropout = torch.nn.Dropout(dropout)
        self.attention_dropout = torch.nn.Dropout(0.1)
        self.has_features = has_features

        self.outputs = torch.nn.ModuleList()
        for task_id in range(out_channels):
            self.outputs.append(
                torch.nn.Sequential(
                    torch.nn.Linear(hidden_channels + int(task_id in has_features), hidden_channels),
                    torch.nn.ReLU(),
                    torch.nn.Linear(hidden_channels, 1)
                )
            )

    def forward(self, loader: DataLoader, task_id: int) -> torch.Tensor:
        """
        Propagates the input through the graph convolution layers, applies the attention mechanism, and outputs the prediction for the specified task ID.

        :param loader: Data loader containing the batch data for graph inputs.
        :param task_id: Identifier for the specific task to generate output.
        :return: Predicted output tensor for the specified task.
        """
        x = loader.x
        fractions = loader.ratio.unsqueeze(1)
        _, monomer = torch.unique_consecutive(loader.monomer_id, return_inverse=True)

        for conv in self.encoder:
            x = conv(x, loader.edge_index).relu()

        # monomer - tem os ids dos atomos que compoe um monomero
        monomer_emb = scatter(x, monomer, reduce='mean')
        # cada atomo aqui tinha as frações do monomero e agr nao preciso mais disso
        monomer_fraction = scatter(fractions, monomer, reduce='min')
        # monomer-level emb # 32
        q = self.fc_q(monomer_emb) * monomer_fraction
        v = self.fc_v(monomer_emb)
        k = self.fc_k(monomer_emb) * monomer_fraction
        print('q', q.shape)
        batch_index = scatter(loader.batch, monomer, reduce='min')
        print('k 1', k.shape)
        print('batch', loader.batch.shape, batch_index.shape)
        k = scatter(k, batch_index, reduce='sum')
        print('k 2', k.shape)

        energy = (q * k[batch_index]) * self.scale
        print('energy', energy.shape)
        attention = softmax(energy, batch_index)
        attention = self.attention_dropout(attention)

        print('v', v.shape, 'att', attention.shape)
        import sys
        sys.exit()
        polymer_embedding = v * attention
        polymer_embedding = scatter(polymer_embedding, batch_index, reduce='sum')
        polymer_embedding = self.dropout(polymer_embedding)
        
        if task_id in self.has_features:
            polymer_embedding = torch.cat((polymer_embedding, loader.feats), dim=1)

        return self.outputs[task_id](polymer_embedding).squeeze(1)

def load_model_state(model: torch.nn.Module, state: Dict[str, torch.Tensor], grad: List[torch.Tensor]):
    """
    Loads the model state and applies gradients to the model's parameters.

    :param model: The neural network model.
    :param state: The initial state of the model.
    :param grad: Gradients to be applied to the model's parameters.
    """
    with torch.no_grad():
        for (name, param), g in zip(model.named_parameters(), grad):
            param.data.copy_(state[name])
            if g is not None:
                param.grad = g.clone()

def compute_combined_gradients(
    grads: Dict[int, List[torch.Tensor]],
    tgt_tasks_id: List[int],
    max_elements: int = 3
) -> Tuple[List[Tuple[int]], List[List[torch.Tensor]]]:
    """
    Computes combined gradients for all possible combinations of auxiliary tasks.
    
    :param grads: Gradients for each task.
    :param tgt_tasks_id: List of target task IDs.
    :param max_elements: Maximum number of tasks to combine.
    :return: A tuple containing task combinations and their corresponding combined gradients.
    """
    tasks_combinations = {}
    combined_gradient = []
    tasks_grouping = [
        [0, 34], [1], [2], [3], [4], [5], [6],
        [7, 30], [8], [9], [10], [11, 31], [12],
        [13, 29], [14], [15], [16], [17], [18],
        [19], [20], [21], [22], [23], [24], [25],
        [26], [27], [28], [32], [33], [35, 38],
        [36], [37], [39], [40], [41], [42], [43]
    ] # I removed CO2/N2 selectivity and CO2/CH4 selectivity tasks

    for tgt in tgt_tasks_id:
        auxiliary_tasks = [task for task in tasks_grouping if tgt not in task]
        for combo in combinations(auxiliary_tasks, max_elements - 1):
            if tgt in [35, 38]:
                key = ([35, 38],) + combo
            else:
                key = ([tgt],) + combo
            unique_key = tuple(np.sort(np.concatenate(key)).tolist())
            if unique_key not in tasks_combinations:
                tasks_combinations[unique_key] = key
                temp_gradient = grads[tgt]
                for tasks in combo:
                    for task in tasks:
                        temp_gradient = [temp_gradient[i] + new_g for i, new_g in enumerate(grads[task])]
                combined_gradient.append(temp_gradient)

    return list(tasks_combinations.values()), combined_gradient

def TAG(
    task_count: Dict[int, int], 
    task_iter_train: Dict[int, iter], 
    task_dataloader: Dict[int, torch.utils.data.DataLoader], 
    tasks: List[int], 
    tgt_tasks_id: List[int], 
    model: torch.nn.Module, 
    optimizer: torch.optim.Optimizer, 
    device: torch.device
) -> Tuple[Dict[int, torch.Tensor], Dict[Tuple[int], Dict[int, float]]]:
    """
    Performs Task Affinity Grouping (TAG) for multi-task learning.
    
    :param task_count: Dictionary with task IDs as keys and their respective counts as values.
    :param task_iter_train: Dictionary with task IDs as keys and their respective data loaders iterator as values.
    :param task_dataloader: Dictionary with task IDs as keys and their respective data loaders as values.
    :param tasks: List of task IDs to be processed.
    :param tgt_tasks_id: List of target task IDs for computing task gains.
    :param model: The neural network model being trained.
    :param optimizer: Optimizer used for updating the model parameters.
    :param device: The device (CPU or GPU) on which the computation will be performed.
    :return: A tuple containing:
             - losses: A dictionary with task IDs as keys and their respective loss values as values.
             - task_gains: A dictionary with combinations of task IDs as keys and their respective TAG values.
    """
    
    # 1. Compute the loss for each task and then sum them all up
    optimizer.zero_grad()
    
    batches = {}
    losses = {}
    grads = {}
    acc_grad = None
    
    for task_id in tasks:
        
        # Round Robin Batching (begin)
        if task_count[task_id] % len(task_dataloader[task_id]) == 0:
            task_iter_train[task_id] = iter(task_dataloader[task_id])
        task_count[task_id] += 1
        batches[task_id] = next(task_iter_train[task_id])
        batches[task_id] = batches[task_id].to(device)
        # Round Robin Batching (end)

        predictions = model(batches[task_id], task_id)
        losses[task_id] = F.mse_loss(predictions, batches[task_id].y)
        raw_grads = torch.autograd.grad(losses[task_id], model.parameters(), allow_unused=True)
        grads[task_id] = [g if g is not None else torch.zeros_like(param) for g, param in zip(raw_grads, model.parameters())]

        if acc_grad is None:
            acc_grad = grads[task_id]
        else:
            acc_grad = [acc_grad[i] + new_g for i, new_g in enumerate(grads[task_id])]

    # 2. Computing the accumulated gradients for all possible combinations of auxiliary tasks
    combined_task_gradients = compute_combined_gradients(grads, tgt_tasks_id)

    # 3. Computing task gain for each combination of auxiliary tasks
    for task_id in losses:
        losses[task_id] = losses[task_id].detach()

    optimizer.zero_grad() 
    initial_state = {name: param.detach().clone() for name, param in model.named_parameters()}
    task_gains = {}

    # Freeze prediction head parameters
    for param in model.outputs.parameters():
        param.requires_grad = False
            
    for combined_task, task_gradient in zip(combined_task_gradients[0], combined_task_gradients[1]):
        optimizer.zero_grad()
        load_model_state(model, initial_state, task_gradient)
        optimizer.step()

        combined_task = tuple(np.concatenate(combined_task).tolist())
        task_gains[combined_task] = {}
        with torch.no_grad():
            for task_id in combined_task:
                task_update_loss = F.mse_loss(model(batches[task_id], task_id), batches[task_id].y).detach()
                task_gains[combined_task][task_id] = 1.0 - (task_update_loss / losses[task_id])

    # 4. Backpropagation of the error of all tasks together
    optimizer.zero_grad() 
    load_model_state(model, initial_state, acc_grad)
    for param in model.outputs.parameters():
        param.requires_grad = True
    optimizer.step()
    
    return losses, task_gains

def train(
    task_id: int, 
    task_count: Dict[int, int], 
    task_iter_train: Dict[int, iter], 
    task_dataloader_train: Dict[int, DataLoader], 
    model: torch.nn.Module, 
    device: torch.device
) -> torch.Tensor:
    """
    Trains the model for a specific task by processing a batch of data.

    :param task_id: Identifier for the current task.
    :param task_count: Dictionary with task IDs as keys and their respective counts as values.
    :param task_iter_train: Dictionary with task IDs as keys and their respective data loaders iterator as values.
    :param task_dataloader_train: Dictionary with task IDs as keys and their respective data loaders as values.
    :param model: The model to be trained.
    :param device: The device (CPU or GPU) on which the computation will be performed.
    :return: The loss value for the current training step.
    """
    model.train()
    if task_count[task_id] % len(task_dataloader_train[task_id]) == 0:
        task_iter_train[task_id] = iter(task_dataloader_train[task_id])
    task_count[task_id] += 1

    batch = next(task_iter_train[task_id])
    predictions = model(batch.to(device), task_id)
    return F.mse_loss(predictions, batch.y)

def evaluate_model(model: torch.nn.Module, dataloader: DataLoader, task_id: int, std: torch.Tensor, device: torch.device) -> Tuple[float, float]:
    """
    Evaluates the model on a specific dataloader and task.

    :param model: The neural network model to evaluate.
    :param dataloader: DataLoader for the data on which to evaluate the model.
    :param task_id: Identifier for the current task.
    :param std: Standard deviation tensor used for denormalizing predictions.
    :param device: Device to run the evaluation on (e.g., 'cuda', 'cpu').
    :return: A tuple containing the normalized loss and the real loss.
    """
    batch_size, loss, znorm_loss = 0, 0, 0
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            predictions = model(batch.to(device), task_id)
            loss += (predictions * std[task_id] - batch.y * std[task_id]).abs().sum().item()
            znorm_loss += (predictions - batch.y).abs().sum().item()
            batch_size += batch.y.size(0)

    return znorm_loss / batch_size, loss / batch_size

def evaluation(
    train_loader: DataLoader, 
    val_loader: DataLoader, 
    test_loader: DataLoader, 
    task_id: int, 
    model: torch.nn.Module, 
    epochId: int,
    lr: float, 
    std: torch.Tensor, 
    device: torch.device, 
    seed: int, 
    output: str, 
    best_val: Optional[float], 
    norm_test_loss: Optional[float], 
    test_loss: Optional[float]
) -> Tuple[float, float, float]:
    """
    Evaluates the model's performance on training, validation, and test sets,
    updating the best model and recording performance metrics.

    :param train_loader: DataLoader for the training set.
    :param val_loader: DataLoader for the validation set.
    :param test_loader: DataLoader for the test set.
    :param task_id: Identifier for the current task.
    :param model: The neural network model to be evaluated.
    :param epochId: Current epoch number.
    :param lr: Learning rate used during training.
    :param std: Standard deviation tensor used for denormalizing predictions.
    :param device: The device (CPU or GPU) on which the evaluation will be performed.
    :param seed: Random seed value.
    :param output: Directory path for saving model and logs.
    :param best_val: The best validation loss observed so far.
    :param norm_test_loss: The normalized test loss observed so far.
    :param test_loss: The real test loss observed so far.
    :return: A tuple containing updated best validation loss, normalized test loss, and test loss.
    """
    # Predict and evaluate on validation and training datasets
    norm_val_loss, val_loss = evaluate_model(model, val_loader[task_id], task_id, std, device)
    norm_train_loss, train_loss = evaluate_model(model, train_loader[task_id], task_id, std, device)

    # Save the model if it has the best validation performance
    model_path = output / f'best_model_epoch_{epochId}_task_{task_id}.pth'
    if best_val is None or val_loss <= best_val:
        best_val = val_loss
        torch.save(model.state_dict(), model_path)
        # If a new best validation loss is achieved, also evaluate on the test set
        norm_test_loss, test_loss = evaluate_model(model, test_loader[task_id], task_id, std, device)

    # Log the evaluation results
    history_path = output / f'history_task_{task_id}.txt'
    with open(history_path, 'a') as f:
        f.write(f'Epoch: {epochId:03d} - '
                f'Task id {task_id:02d}, LR: {lr:.7f}, '
                f'Loss (Norm/Real): {norm_train_loss:.7f}/{train_loss:.7f}, '
                f'Val Loss (Norm/Real): {norm_val_loss:.7f}/{val_loss:.7f}, '
                f'Test Loss (Norm/Real): {norm_test_loss:.7f}/{test_loss:.7f}\n')
    
    return best_val, norm_test_loss, test_loss
