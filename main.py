
import os
import joblib
import pandas as pd

from src.load_data import get_dataset
from src.preprocessing import split_dataset, normalize_pixels
from src.feature_extraction import extract_hog_features
from sklearn.dummy import DummyClassifier

from src.evaluate import (
    evaluate_model,
    plot_confusion_matrix,
    plot_model_comparison
)


def main():

    print("===== GIAI DOAN 5: DANH GIA MO HINH =====")

    # 1. Load dữ liệu
    X, y, images = get_dataset()

    # 2. Chia dữ liệu giống các giai đoạn trước
    (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test
    ) = split_dataset(X, y)

    # 3. Chuẩn hóa Pixel
    X_test = normalize_pixels(X_test)

    # 4. Trích xuất đặc trưng HOG cho Test
    X_test_hog = extract_hog_features(X_test)

    print("\n===== TEST DATA =====")
    print("Pixel:", X_test.shape)
    print("HOG:", X_test_hog.shape)
    print("So luong mau Test:", len(y_test))

    # 5. Tải các mô hình đã huấn luyện
    models = {
    "K-NN Pixel": (
        joblib.load("models/knn_pixel.pkl"),
        X_test
    ),

    "K-NN HOG": (
        joblib.load("models/knn_hog.pkl"),
        X_test_hog
    ),

    "Decision Tree Pixel": (
        joblib.load("models/decision_tree_pixel.pkl"),
        X_test
    ),

    "Decision Tree HOG": (
        joblib.load("models/decision_tree_hog.pkl"),
        X_test_hog
    )
}

    # 6. Xây dựng mô hình Baseline
    print("\n===== BASELINE MODEL =====")

    baseline_model = DummyClassifier(
        strategy="most_frequent"
    )

    baseline_model.fit(X_train, y_train)

    models["Baseline"] = (
        baseline_model,
        X_test
    )

    # 6. Đánh giá từng mô hình
    all_results = []
    all_predictions = {}

    for model_name, (model, X_input) in models.items():

        result, y_pred = evaluate_model(
            model,
            X_input,
            y_test,
            model_name
        )

        all_results.append(result)
        all_predictions[model_name] = y_pred

        # Vẽ Confusion Matrix
        plot_confusion_matrix(
            y_test,
            y_pred,
            model_name
        )

    # 7. Lưu bảng kết quả
    os.makedirs("results/metrics", exist_ok=True)
    os.makedirs("results/predictions", exist_ok=True)

    df = pd.DataFrame(all_results)

    df.to_csv(
        "results/metrics/test_results.csv",
        index=False
    )

    # 8. Lưu dự đoán của từng mô hình
    prediction_data = pd.DataFrame({
        "Actual": y_test
    })

    for model_name, predictions in all_predictions.items():

        column_name = model_name.replace(" ", "_").replace("+", "plus")

        prediction_data[column_name] = predictions

    prediction_data.to_csv(
        "results/predictions/test_predictions.csv",
        index=False
    )

    # 9. Vẽ biểu đồ so sánh
    plot_model_comparison(all_results)

    # 10. Hiển thị bảng tổng kết
    print("\n===== TONG KET KET QUA TEST =====")
    print(df.to_string(index=False))

    best_model = df.loc[df["Macro_F1"].idxmax()]

    print("\n===== MO HINH TOT NHAT TREN TEST =====")
    print("Model:", best_model["Model"])
    print(f"Accuracy: {best_model['Accuracy']:.4f}")
    print(f"Macro F1: {best_model['Macro_F1']:.4f}")

    print("\nDa luu ket qua tai:")
    print("- results/metrics/test_results.csv")
    print("- results/predictions/test_predictions.csv")
    print("- results/figures/")

    print("\nHOAN THANH GIAI DOAN 5!")


if __name__ == "__main__":
    main()
