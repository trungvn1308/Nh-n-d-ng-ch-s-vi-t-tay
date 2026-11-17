
from sklearn.neighbors import KNeighborsClassifier


def train_knn(X_train, y_train, k=5):
    """
    Xây dựng và huấn luyện mô hình K-NN.
    """

    model = KNeighborsClassifier(
        n_neighbors=k,
        metric="euclidean",
        weights="uniform"
    )

    model.fit(X_train, y_train)

    return model
