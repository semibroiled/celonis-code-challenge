""" This module contains all the functions needed for the Data
Preparations step, that is, to access and load the data from the
files present within the directories
"""
import numpy as np
import os


# Reach data in nested directory
def get_file_paths(root_dir: str = ".") -> list:
    """This function take in your root directory where the
    celonis-code-challenge folder is and returns a list with all the
    path names to the txt files where the data are being stored
    Args:
        root_dir (str, optional): Input is the root directory where we
        are currently working in. Defaults to ".".

    Returns:
        list: The function returns a list with the relevant file paths
        from the uWaveGestureLibrary folder. It looks for .txt files in
        the directories but exclude the Readme.txt file and parameter
        files, which is less than 7 character typically
    """
    # Empty List to store all filepaths
    file_paths = []

    # Recursively get all files from current working directory
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if (".txt" in file) and ("acceleration" in file.lower()):
                # Append filepaths to empty list
                file_paths.append(os.path.join(root, file))

    return file_paths


def create_arrays_from_files(filepaths: list) -> list:
    """Read the data in text files and create arrays in lists. Also read
    Gesture Class from filename

    Args:
        filepaths (list): Input takes in a list with filepaths given in
        strings.

    Returns:
        list: Returns two lists which are multidimensional lists or
        arrays where the data from the paths are read and stored into-
        The first contains the x,y,z acceleration values in each txt
        file, the second return contains the class of the gestures

    """
    # List to store numpy array from text files
    gesture_arrays = []
    # List to store the corresponding Gesture Code from path name
    gesture_codes = []

    for file_path in filepaths:
        # Load the values in a text file, and flatten the values to get
        # all the values to use in our features in a single row and
        # multiple columns for each txt data
        ges_arr = np.loadtxt(file_path).reshape(1, -1)

        # The gesture code is only mentioned in the filename with the
        # schema index gesture-index repetition. Splice the path string
        # to get gesture code, and put in array to use as feature column
        gesture_code = file_path.split("_")[-1].split("-")[0][-1]
        gesture_arrays.append(ges_arr)
        gesture_codes.append(int(gesture_code))

    return gesture_arrays, gesture_codes


def pad_with_0(arrays: list):
    """Take our array of data and normalise column numbers by filling
        missing number with 0s.
    Args:
        arrays (list): Input an array

    Returns:
        list | np.ndarray: returns an array matrix where missing column
        values in rows are appened with 0
    """
    # Max fcn returns the maximum column number in our array
    max_length = max(array.shape[1] for array in arrays)
    # Add additional columns/values to our shorter inner arrays so we
    # can make a numpy array without errors Empty List in which to
    # append padded arrays
    padded_arrays = []
    for array in arrays:
        padded_array = np.pad(
            array,
            pad_width=((0, 0), (0, max_length - array.shape[1])),
            mode="constant",
            constant_values=0,
        )
        # Theoretically I should be able to leave out contant_values
        # argument and it would pad with value None aka 0 by default.
        # Try it out later
        padded_arrays.append(padded_array)

    return padded_arrays
