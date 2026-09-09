import numpy as np
import pandas as pd

class PurgedKFold:
    """
    Implements Purged and Embargoed K-Fold Cross-Validation.
    Handles overlapping labels in time-series data.
    """
    def __init__(self, n_splits=5, pct_embargo=0.01, purge_horizon=5):
        self.n_splits = n_splits
        self.pct_embargo = pct_embargo
        self.purge_horizon = purge_horizon # number of bars a label 'looks forward'

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits

    def split(self, X, y=None, groups=None):
        if y is None:
            y = pd.Series(np.arange(X.shape[0]))
            
        n_samples = X.shape[0]
        indices = np.arange(n_samples)
        embargo_size = int(n_samples * self.pct_embargo)
        
        # Standard K-Fold indices
        fold_size = n_samples // self.n_splits
        
        for i in range(self.n_splits):
            test_start = i * fold_size
            test_end = (i + 1) * fold_size if i != self.n_splits - 1 else n_samples
            
            test_indices = indices[test_start:test_end]
            
            # 1. Purging: Remove samples from train set whose labels overlap with test set
            # If a label at 't' looks forward to 't + purge_horizon', 
            # we must remove 'purge_horizon' samples BEFORE the test set.
            train_start_indices = indices[0:max(0, test_start - self.purge_horizon)]
            
            # 2. Embargoing: Remove samples immediately AFTER the test set 
            # to handle serial correlation.
            train_end_indices = indices[test_end + embargo_size:]
            
            train_indices = np.concatenate([train_start_indices, train_end_indices])
            
            yield train_indices, test_indices

class PurgedWalkForwardCV:
    """
    Rigorous Walk-Forward with Purging and Embargoing.
    """
    def __init__(self, n_splits=5, pct_embargo=0.02, purge_horizon=5):
        self.n_splits = n_splits
        self.pct_embargo = pct_embargo
        self.purge_horizon = purge_horizon

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits

    def split(self, X, y=None, groups=None):
        n_samples = X.shape[0]
        indices = np.arange(n_samples)
        fold_size = n_samples // (self.n_splits + 1)
        embargo_size = int(n_samples * self.pct_embargo)
        
        for i in range(1, self.n_splits + 1):
            train_end = i * fold_size
            # The test set starts after the train set
            # But wait, in Walk-Forward, the 'leakage' is usually train -> test 
            # if labels look forward.
            
            # Rigorous Walk-Forward:
            # Train: [0 : train_end - purge_horizon]
            # Test:  [train_end + embargo_size : train_end + embargo_size + fold_size]
            
            actual_train_end = train_end - self.purge_horizon
            if actual_train_end <= 0:
                continue
                
            test_start = train_end + embargo_size
            test_end = test_start + fold_size
            
            if test_end > n_samples:
                test_end = n_samples
            
            if test_start >= n_samples:
                break
                
            train_indices = indices[:actual_train_end]
            test_indices = indices[test_start:test_end]
            
            if len(test_indices) > 0:
                yield train_indices, test_indices

if __name__ == "__main__":
    # Test the splitter
    X = np.zeros((1000, 10))
    cv = PurgedWalkForwardCV(n_splits=5, purge_horizon=10)
    for i, (train, test) in enumerate(cv.split(X)):
        print(f"Fold {i+1}: Train {len(train)}, Test {len(test)} | Gap: {test[0] - train[-1]}")
