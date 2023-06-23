"""This is the file from which the machine leraning pipeline is executed.
"""
import os
import numpy as np
from typing import Path

# Reach data in nested directory

root_dir = "."  # Root directory where files are located is our current folder

file_paths = []  # Empty List to store all filepaths

# Recursively get all files from current working directory
for root, dirs, files in os.walk(root_dir):
    for file in files:
        if (".txt" in file) and (not "readme" in file.lower()) and (not len(file) < 7):
            # Look for .txt files in the directories but exclude the
            # Readme.txt file and parameter files, which is less than 7
            # character typically
            file_paths.append(
                os.path.join(root, file)
            )  # Append filepaths to empty list

print(len(file_paths))

# Define a extended sigmoid function for multi classification purposes
# Taken from ML Script for TU Darmstadt and Springer Text Book on
# Numerical Analysis


def sigmoid_softmax(input: np.ndarray) -> np.ndarray:
    """The Softmax function is an extension of the Sigmoid function.
    It is commonly used in logistic regressions and expands itself from
    binary classification to be able to classify multiple classes. In
    multi-class classification algorithms, it take a vector of scores and
    transfroms them into a probabiliy distribution over multiple classes.
    The function calculates the predicted class as one with the highest
    probabilit according to output.


    Args:
        input (np.ndarray): _description_

    Returns:
        np.ndarray: _description_
    """
    # Return Maximum along Column
    exp_input = np.exp(input - np.max(input, axis=1, keepdims=True))

    score = exp_input / np.sum(exp_input, axis=1, keepdims=True)

    return score


velocity = 3.0  # m/s

# Reach data in nested directory


def get_file_path(root_dir: Path = "."):
    """_summary_

    Args:
        root_dir (Path, optional): _description_. Defaults to ".".
    """
    file_paths = []  # Empty List to store all filepaths

    # Recursively get all files from current working directory
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            # Look for .txt files in the directories but exclude the
            # Readme.txt file and parameter files, which is less than 7
            # character typically
            if (
                (".txt" in file)
                and (not "readme" in file.lower())
                and (not len(file) < 7)
            ):
                # Append filepaths to empty list
                file_paths.append(os.path.join(root, file))

    print(len(file_paths))


# ---
A = 6
a1 = np.arange(6)

a2 = np.zeros(A)
for i in range(A):
    a2[i] = i

# ---
ges_arrs_padded = []  # Empty List in which to append padded arrays
for arr in ges_arrs:
    ges_arr_padded = np.pad(
        arr,
        pad_width=((0, 0), (0, max_arr_length - arr.shape[1])),
        mode="constant",
        constant_values=0,
    )  # Theoretically I should be able to leave out contant_values argument and it would pad with value None aka 0 by default. Try it out later
    ges_arrs_padded.append(ges_arr_padded)

# make memory more efficient
np_arrs_padded = np.zeros((n_data_set, max_arr_length, 3))

for idx, arr in enumerate(ges_arr):
    np_arrs_padded[idx, : arr.shape[1], :] = arr


# Normalize dataset to reduce noise or bias and shuffle samples in rows to prevent user bias in ML Model
# Shuffle Dataset
combined_dataset = np.hstack((comb_ges_arr, ges_codes_in_np))  # Combine dataset

shuffled_dataset = np.random.permutation(combined_dataset)  # Create shuffled copy

rand_idx = np.random.permutation(np.arange(combined_dataset.shape[0]))

shuffled = combined_dataset[rand_idx]
