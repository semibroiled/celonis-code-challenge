"""This module will containg all the functions made for the processes
modelling, training and evaluation. We will definde some important
mathematical functions needed for logistical regression 
on multi class problems. As well as call just a few plots to
humanize our models insight"""

# Define a extended sigmoid function for multi classification purposes
# Taken from ML Script for TU Darmstadt and Springer Text Book on Numerical Analysis

# Import packages for module to run
import numpy as np
import matplotlib.pyplot as plt


def sigmoid_softmax(input: np.ndarray) -> np.ndarray:
    """The Softmax function is an extension of the Sigmoid function.
    It is commonly used in logistic regressions and expands itself from
    binary classification to be able to classify multiple classes. In
    multi-class classification algorithms, it take a vector of scores
    and transfroms them into a probabiliy distribution over multiple
    classes. The function calculates the predicted class as one with the
    highest probabilit according to output.

    Input: score of real value

    Returns:

    Output: probability distribution over multiple classes
        np.ndarray: probability distribution over softmax function
    """

    """

    """
    # Return Maximum along Column
    exp_input = np.exp(input - np.max(input, axis=1, keepdims=True))

    score = exp_input / np.sum(exp_input, axis=1, keepdims=True)

    return score


def cross_entropy(y_predicted: np.ndarray, y_true: np.ndarray) -> np.ndarray | float:
    """_summary_
        Define the loss function # Taken from Springer Text Book on
        Numerical The Cross Entropy Loss fucntion is used in logistic
        regression for multi classification problems. It measures the
        deviance between predicted probabilites and true
     labels

    The goal of our regression will be to minimize the loss represented
    by this function. This will help to classify input sampled to its
    correct classes Returns:
        np.ndarray | int | float: loss function of regerssion
    """

    loss = -np.mean(np.sum(y_true * np.log(y_predicted + 1e-8), axis=1))

    return loss


def predict(X: np.ndarray, weight: np.ndarray) -> np.ndarray:
    """_summary_
        Setup prediction function Self implemented from context

        Take X and learned weight from gradient descent to give a
        prediction on y

    Args:
        X (np.ndarray): X matrix of dataset
        weight (np.ndarray): learned weights

    Returns:
        np.ndarray: get back precited classes aka y_predicted
    """
    scores = np.dot(X, weight)
    probability_dist = sigmoid_softmax(scores)
    predicted_class = np.argmax(probability_dist, axis=1)
    predicted_class = predicted_class.reshape(-1, 1)

    # Return indices from probability
    return predicted_class


def accuracy_of_model(y_predicted: np.ndarray, y: np.ndarray) -> np.ndarray | float:
    """_summary_
        Setup accuracy function Self implemented from context

        Calculate accurarcy of our predictions against true values

    Args:
        y_predicted (np.ndarray): Predicted y from weights
        y (np.ndarray): true y

    Returns:
        np.ndarray | float: get back comparision array and mean accuracy
    """

    acc_pred = y_predicted == y
    acc_pred_mean = np.mean(acc_pred)

    return acc_pred, acc_pred_mean


def gradient_descent(
    X: np.ndarray,
    y: np.ndarray,
    learning_rate: float = 0.1,
    iter: int = 1000,
    print_loss: bool = False,
) -> np.ndarray:
    """_summary_
        Setup Gradient Descent Algorithm to update and calculate weights
        Taken from documentation in sklearn library

        Implement gradient descent algorithm to initialize and update
        weights for our model


    Args:
        X (np.ndarray): input training matrix
        y (np.ndarray): input OHE target feature
        learning_rate (float, optional): _description_. Defaults to 0.1.
        iter (int, optional): _description_. Defaults to 1000.
        print_loss (bool, optional): _description_. Defaults to False.

    Returns:
        np.ndarray: weight distribution
        list: epoch indexing
        list: loss indexing
    """
    # Empty list for loss
    losses = []

    # Get Sample, Feature and Class numbers
    num_samples, num_features = X.shape
    num_classes = y.shape[1]

    # Initialize Parameters
    weight = np.random.randn(num_features, num_classes)

    # Setup Iteration to update weights
    for i in range(iter):
        scores = np.dot(X, weight)  # Dot product of arrays
        # alterntely use @?
        probabiliy_dist = sigmoid_softmax(scores)
        gradient = np.dot(X.T, probabiliy_dist - y) / num_samples

        weight = weight - learning_rate * gradient

        loss = cross_entropy(probabiliy_dist, y)
        losses.append(loss)

        # Print progress and append epoch metrics
        if print_loss and (i + 1) % 100 == 0:
            # Find accuracy for plot
            print(f"Iteration {i+1}, Loss: {loss}")

    return weight, losses


def confusion_matrix_plot(y_true: np.ndarray, y_predicted: np.ndarray) -> None:
    """_summary_

    Plots a confusion matrix to get true postives, true negatives,
    false negatives and false positives densities Courtesy of Nihal
    Barua, Thanks for showing me what a confusion matrix is
    """

    all_classes = np.unique(np.concatenate((y_true, y_predicted)))
    num_classes = len(all_classes)

    confusion_matrix = np.zeros((num_classes, num_classes), dtype=int)

    for i in range(num_classes):
        for j in range(num_classes):
            confusion_matrix[i, j] = np.sum((y_true == i) & (y_predicted == j))

    plt.figure(figsize=(10, 10))
    plt.imshow(confusion_matrix, interpolation="nearest", cmap="Blues")
    plt.colorbar()
    plt.xticks(np.arange(num_classes), range(num_classes))
    plt.yticks(np.arange(num_classes), range(num_classes))
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix on Validation Set")
    plt.show()


def loss_plot(loss: list | np.ndarray) -> None:
    """_summary_
    Plot Loss over Epochs, showing progression

    Args:
        loss (list): How accurate ie how low is our losses
    """
    plt.plot(loss)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Model Loss over Epochs")
    plt.show()


def compare_accuracy(accuracy_test: float, acuracy_train: float) -> None:
    """_summary_
    Plot Loss over Epochs, showing progression

    Args:
        accuracy (list): How accurate ie how high our model accuracy is
    """
    labels = [
        "Accuracy of Predicition on Test Data",
        "Accuracy of Prediction on Train Data",
    ]
    values = [accuracy_test, acuracy_train]

    plt.bar(labels, values)

    plt.ylabel("Accuracy in %")
    plt.title("Comparison of Accuracies")
    plt.show()


def model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_valid: np.ndarray,
    y_valid_test: np.ndarray,
    y_valid_train: np.ndarray,
    iter: int = 1000,
    learning_rate: float = 0.1,
    print_loss: bool = False,
) -> None:
    """_summary_

    Bring everything together into model function to train,
    evalute/validate and represent/infer


    """
    # Gradient descent to retrieve trained parameters
    weight, model_losses = gradient_descent(
        X_train, y_train, learning_rate, iter, print_loss
    )

    # Predict
    y_predicted_train = predict(X_train, weight)
    y_predicted_test = predict(X_valid, weight)

    # Accuracy Inspection
    acc_train_compare, acc_train_mean = accuracy_of_model(
        y_predicted_train, y_valid_train
    )
    acc_test_compare, acc_test_mean = accuracy_of_model(y_predicted_test, y_valid_test)

    # As Percentages
    acc_test = acc_test_mean * 100
    acc_train = acc_train_mean * 100

    print(f"The prediction accuracy of the training dataset is {acc_train}%")
    print("The comparision array is as follows:")
    print(acc_train_compare)

    print(f"The prediction accuracy of the validation dataset is {acc_test}%")
    print("The comparision array is as follows:")
    print(acc_test_compare)

    # Compare accuracies on dataset
    compare_accuracy(acc_test, acc_train)

    # Plot Visualisations

    # Plot Loss Function
    loss_plot(model_losses)

    # Plot Confusion Matrix
    confusion_matrix_plot(y_valid_test, y_predicted_test)
