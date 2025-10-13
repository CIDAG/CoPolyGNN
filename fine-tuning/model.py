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

        monomer_emb = scatter(x, monomer, reduce='mean')
        monomer_fraction = scatter(fractions, monomer, reduce='min')

        q = self.fc_q(monomer_emb) * monomer_fraction
        v = self.fc_v(monomer_emb)
        k = self.fc_k(monomer_emb) * monomer_fraction
        batch_index = scatter(loader.batch, monomer, reduce='min')
        k = scatter(k, batch_index, reduce='sum')

        energy = (q * k[batch_index]) * self.scale
        attention = softmax(energy, batch_index)
        attention = self.attention_dropout(attention)

        polymer_embedding = v * attention
        polymer_embedding = scatter(polymer_embedding, batch_index, reduce='sum')
        polymer_embedding = self.dropout(polymer_embedding)
        
        if task_id in self.has_features:
            polymer_embedding = torch.cat((polymer_embedding, loader.feats), dim=1)

        return self.outputs[task_id](polymer_embedding).squeeze(1)


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