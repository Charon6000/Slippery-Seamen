import matplotlib.pyplot as plt
import numpy as np


def show_images_from_dataset(dataset, num=8, ncols=4):
    images = []
    labels = []
    for img, lbl in dataset.unbatch().take(num):
        arr = img.numpy()
        # Convert to uint8 for display WITHOUT blindly multiplying by 255.
        # Strategy:
        # - if already uint8, use as-is
        # - else (float), perform robust per-image min-max scaling to 0-255
        #   (avoids pure-white/black when values are in unexpected ranges)
        if arr.dtype == np.uint8:
            disp = arr
        else:
            a = arr.astype('float64')
            minv = float(np.nanmin(a))
            maxv = float(np.nanmax(a))
            if not np.isfinite(minv) or not np.isfinite(maxv):
                disp = np.zeros_like(a, dtype='uint8')
            elif maxv - minv < 1e-6:
                disp = np.clip(a, 0, 255).astype('uint8')
            else:
                norm = (a - minv) / (maxv - minv)
                disp = np.clip((norm * 255.0), 0, 255).astype('uint8')

        # If grayscale single-channel, convert to RGB for consistent display
        if disp.ndim == 2:
            disp = np.stack([disp] * 3, axis=-1)
        elif disp.ndim == 3 and disp.shape[-1] == 1:
            disp = np.concatenate([disp] * 3, axis=-1)

        images.append(disp)
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