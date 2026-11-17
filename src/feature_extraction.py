
import numpy as np
from skimage.feature import hog


def extract_hog_features(X):
    """
    Trích xuất đặc trưng HOG từ dữ liệu ảnh chữ số.

    X: Dữ liệu có dạng (n_samples, 64)
    Mỗi mẫu gồm 64 pixel của ảnh 8x8.

    Kết quả: Vector đặc trưng HOG.
    """

    # Chuyển dữ liệu về dạng ảnh 8x8
    images = X.reshape(-1, 8, 8)

    features = []

    for image in images:
        hog_feature = hog(
            image,
            orientations=8,
            pixels_per_cell=(4, 4),
            cells_per_block=(1, 1),
            block_norm="L2-Hys",
            feature_vector=True
        )

        features.append(hog_feature)

    return np.array(features)
