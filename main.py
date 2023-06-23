# Import relevant libraries to be used
def main() -> None:
    import numpy as np
    import matplotlib.pyplot as plt
    import DataPreparation
    import DataPreprocessing
    import Pipeline

    # Set Random Seed to our Pipeline to ensure reproducability
    np.random.seed(13)

    file_paths = DataPreparation.get_file_paths()

    ges_arrs, ges_codes = DataPreparation.create_arrays_from_files(file_paths)

    X_data = np.vstack(DataPreparation.pad_with_0(ges_arrs))

    y_data = np.vstack(ges_codes)

    X_data_new, y_data_new = DataPreprocessing.shuffle_X_y(X_data, y_data)

    y_ohe, y_idx_shift = DataPreprocessing.one_hot_encoding(y_data_new)

    (
        X_train,
        y_train,
        X_valid,
        y_valid_test,
        y_valid_train,
    ) = DataPreprocessing.train_test_split(X_data_new, y_ohe, y_idx_shift)

    # Set hyperparameters in a dictionary
    hyperparameters = {"Learning Rate": 0.1, "Number of Iterations": 5000}

    # Run Model
    Pipeline.model(
        X_train, y_train, X_valid, y_valid_test, y_valid_train, print_loss=True
    )


if __name__ == "__main__":
    main()
    print("Thank you for your considerations! :)")
