from typing import List, Optional, Tuple

import numpy as np
from sklearn.model_selection import KFold


def get_indices(k: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generates train, validation, and test indices for k-fold cross-validation.

    :param k: The number of folds or splits in the dataset.    
    :return: A tuple containing:
        - train_idx (np.ndarray): An array of shape (k, k-2) where each row contains the indices used for training in each fold.
        - val_idx (np.ndarray): An array of shape (k,) containing the indices used for validation in each fold.
        - test_idx (np.ndarray): An array of shape (k,) containing the indices used for testing in each fold.
    """
    val_idx = np.arange(k)
    test_idx = (val_idx + 1) % k

    train_idx = np.arange(k)
    train_idx = np.vstack([np.setdiff1d(train_idx, (test_idx[i], val_idx[i])) for i in range(k)])

    return train_idx, val_idx, test_idx

def get_folds(n_samples: int, k: int, random_state: Optional[int] = None) -> List[np.ndarray]:
    """
    Generates indices for k-fold cross-validation splits.

    :param n_samples: The total number of samples in the dataset.
    :param k: The number of folds or splits to create.
    :param random_state: Controls the shuffling applied to the data before splitting. Pass an int for reproducible output
                         across multiple function calls. Default is None.
    
    :return: A list of numpy arrays, where each array contains the indices for a single fold.
    """
    kf = KFold(n_splits=k, shuffle=True, random_state=random_state)    
    indices = np.arange(n_samples)

    return [fold for _, fold in kf.split(indices)]