from sklearn.datasets import load_digits


def get_dataset():
    """
    Tai bo du lieu chu so viet tay.
    """

    digits = load_digits()

    X = digits.data
    y = digits.target

    images = digits.images

    print("===== THONG TIN DATASET =====")
    print("So luong anh:", len(images))
    print("So dac trung:", X.shape[1])
    print("Kich thuoc anh:", images.shape[1:])
    print("Cac lop:", set(y))

    return X, y, images


if __name__ == "__main__":
    get_dataset()