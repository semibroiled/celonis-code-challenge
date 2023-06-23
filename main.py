# Import relevant libraries to be used
import numpy as np
import data_preparation
import data_preprocessing
import pipe_line


# Define main function
def main() -> None:
    # Set Random Seed to our Pipeline to ensure reproducability
    np.random.seed(13)

    file_paths = data_preparation.get_file_paths()

    ges_arrs, ges_codes = data_preparation.create_arrays_from_files(file_paths)

    X_data = np.vstack(data_preparation.pad_with_0(ges_arrs))

    y_data = np.vstack(ges_codes)

    X_data_new, y_data_new = data_preprocessing.shuffle_X_y(X_data, y_data)

    y_ohe, y_idx_shift = data_preprocessing.one_hot_encoding(y_data_new)

    (
        X_train,
        y_train,
        X_valid,
        y_valid_test,
        y_valid_train,
    ) = data_preprocessing.train_test_split(X_data_new, y_ohe, y_idx_shift)

    # Set hyperparameters in a dictionary
    hyperparameters = {"Learning Rate": 0.1, "Number of Iterations": 1000}

    # Run Model
    pipe_line.model(
        X_train,
        y_train,
        X_valid,
        y_valid_test,
        y_valid_train,
        learning_rate=hyperparameters["Learning Rate"],
        iter=hyperparameters["Number of Iterations"],
        print_loss=True,
    )


if __name__ == "__main__":
    main()
    print("Thank you for your considerations! :)")
