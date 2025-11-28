import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


def show_images_from_dataset(dataset, num=8, ncols=4):
    images = []
    labels = []
    for img, lbl in dataset.unbatch().take(num):
        arr = img.numpy()
        if arr.dtype != np.uint8:
            arr = np.clip(arr * 255.0, 0, 255).astype('uint8')
        images.append(arr)
        try:
            labels.append(int(lbl.numpy()))
        except Exception:
            labels.append(None)

    n = len(images)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols*3, nrows*3))
    axes = axes.flatten()
    for i in range(n):
        axes[i].imshow(images[i])
        if labels[i] is not None:
            axes[i].set_title(str(labels[i]))
        axes[i].axis('off')
    for ax in axes[n:]:
        ax.axis('off')
    plt.tight_layout()
    plt.show()