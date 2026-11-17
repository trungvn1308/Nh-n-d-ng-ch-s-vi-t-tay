
from sklearn.model_selection import train_test_split


def split_dataset(X, y):
    """
    Chia dữ liệu thành:
    - 70% Train
    - 15% Validation
    - 15% Test
    """

    # Chia 70% train và 30% dữ liệu còn lại
    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=42,
        stratify=y
    )

    # Chia phần 30% còn lại thành 15% validation và 15% test
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.50,
        random_state=42,
        stratify=y_temp
    )

    return (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test
    )


def normalize_pixels(X):
    """
    Chuẩn hóa giá trị pixel từ 0-16 về 0-1.
    """

    return X / 16.0
