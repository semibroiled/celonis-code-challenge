""" This module contains all the functions needed for the Data
Preprocessing and Feature Extraction step, that is, to process, clean,
shuffle, normalise and create our training and validations data 
"""

import numpy as np


def shuffle_X_y(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Shuffle our dataset along the rows to reduce noise and user bias

    Returns:
        np.ndarray: Two shuffled numpy arrays of Matrix X with data
        features and vector y with target classifications
    """
    # Normalize dataset to reduce noise or bias and shuffle samples in
    # rows to prevent user bias in ML Model Shuffle Dataset Combine
    # dataset
    combined_dataset = np.hstack((X, y))

    # Create shuffled copy
    shuffled_dataset = np.random.permutation(combined_dataset)

    rand_idx = np.random.permutation(np.arange(combined_dataset.shape[0]))

    # shuffled = combined_dataset[rand_idx]

    # Split into X and y
    X_shuffled = shuffled_dataset[:, : (shuffled_dataset.shape[1] - 1)]  # X
    y_shuffled = shuffled_dataset[:, -1].reshape(-1, 1)  # y

    return X_shuffled, y_shuffled


def normalize_std_mean(X_train: np.ndarray, X_test: np.ndarray) -> np.ndarray:
    """Normalize dataset to get 0 uniform Standard Deviation along Mean

    Args:
        X_train (np.ndarray): Data Matrix with Features that will be
        trained in Logistic Regression X_test (np.ndarray): Data Matrix
        with Features that will be used to validate model

    Returns:
        np.ndarray: Normalized Dataset arrays of the Arguments
    """
    # Normalize Dataset
    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0)

    X_train_norm = (X_train - mean) / (std + 1e-8)

    X_test_norm = (X_test - mean) / (std + 1e-8)

    return X_train_norm, X_test_norm


def robust_scaling(X_train: np.ndarray, X_test: np.ndarray) -> np.ndarray:
    """Robust scaling Normalisation. Courtersy of Statistics Script from
        TUDa. What is important is that the same parameters of iqr are used
        on all datasets.


    Args:
        X_train (np.ndarray): Training Dataset X_test (np.ndarray):
        Validation Dataset

    Returns:
        np.ndarray: Dataset returned are normalised datasets in Argument
    """

    # IMP! Apply same

    median = np.median(X_train, axis=0)

    q1 = np.percentile(X_train, 25, axis=0)
    q3 = np.percentile(X_train, 75, axis=0)

    iqr = q3 - q1

    X_train_scaled = (X_train - median) / iqr
    X_test_scaled = (X_test - median) / iqr

    return X_train_scaled, X_test_scaled


def one_hot_encoding(y: np.ndarray) -> np.ndarray:
    """One Hot Encoding as taken from Kaggle Courses and sklearn
        documentationn. This splits up our Target Column of dimension 1
        into a dimension representing its number of classes. As the
        gestures are already given from 1 to 8, I decided to use it
        true to Python Indexing by subtracting 1.

    Args:
        y (np.ndarray): Target Column Vector

    Returns:
        np.ndarray: Target Matrix, in future considerations
        mathematically denoted as vector even though it is essentially
        a matrix
    """

    # One Hot Code y vector for our multi class classificaiton problem

    # Bring to Start Index 0
    y = y.astype(int)
    y_idxshift = y - 1

    # Get the number of classes
    num_classes = np.max(y_idxshift) + 1

    # Initialize empty numpy array to hold OHE Labels
    y_encoded = np.zeros((len(y_idxshift), num_classes))
    # Set to 1 for corresponding sample
    y_encoded[np.arange(len(y_idxshift)), y_idxshift.flatten()] = 1

    return y_encoded, y_idxshift


def train_test_split(
    X: np.ndarray, y_encoded: np.ndarray, y_indexed: np.ndarray
) -> np.ndarray:
    """Split our dataset with feature matrix, encoded target matrix and
        class denoted target column vector into training and validation
        sets.


    Args:
        X (np.ndarray): Feature Matrix
        y_encoded (np.ndarray): One Hot Encoded Target
        y_indexed (np.ndarray): Target with class from 0 to 7

    Returns:
        np.ndarray: Training featues,
        training encoded target,
        Validation features,
        validation target,
        validation target used in training
    """
    # Make Training Data Set of sample size ratio 8:2
    # Cutoff index 8:2
    index_cutoff = int(np.round(X.shape[0] * 0.8))

    # Segregate Samples for Training

    # Training Dataset Matrix X
    X_train = X[:index_cutoff, :]
    # Training Dataset Matrix y
    y_train = y_encoded[:index_cutoff, :]

    # Add test here that y_train corresponds to y_encoded? that its the
    # same index? Or maybe test with idx back at shuffle to see shuffle
    # is successful along rows and not columns

    # Make Validation Data Set of Sample Size of ratio 8:2

    # Shuffled validation sets
    X_valid = X[index_cutoff:, :]
    y_valid_test = y_indexed[index_cutoff:, :]
    y_valid_train = y_indexed[:index_cutoff, :]

    return X_train, y_train, X_valid, y_valid_test, y_valid_train
