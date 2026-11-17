import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def show_digit_samples(images, labels):

    plt.figure(figsize=(10, 5))

    for i in range(10):
        plt.subplot(2, 5, i + 1)

        plt.imshow(images[i], cmap="gray")

        plt.title(f"Label: {labels[i]}")

        plt.axis("off")

    plt.tight_layout()
    plt.show()


def show_class_distribution(labels):

    df = pd.DataFrame({
        "Digit": labels
    })

    plt.figure(figsize=(9, 5))

    sns.countplot(
        data=df,
        x="Digit",
        hue="Digit",
        palette="viridis",
        legend=False
    )

    plt.title("Distribution of Handwritten Digits")

    plt.xlabel("Digit")
    plt.ylabel("Number of Images")

    plt.show()