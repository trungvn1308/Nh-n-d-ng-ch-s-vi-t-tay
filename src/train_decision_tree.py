
from sklearn.tree import DecisionTreeClassifier


def train_decision_tree(X_train, y_train, max_depth=10):
    """
    Huấn luyện mô hình Decision Tree.

    max_depth:
        10   : Cây có độ sâu tối đa 10
        20   : Cây có độ sâu tối đa 20
        None : Không giới hạn độ sâu tối đa
    """

    model = DecisionTreeClassifier(
        criterion="gini",
        max_depth=max_depth,
        random_state=42
    )

    model.fit(X_train, y_train)

    return model
