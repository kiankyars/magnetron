import urllib
import numpy as np
from PIL import Image, ImageFilter
from scipy.ndimage import center_of_mass
from tqdm import tqdm


def download_with_progress(url: str, filename: str) -> None:
    class TqdmBarUpdater(tqdm):
        def update_to(
            self, b: int = 1, bsize: int = 1, tsize: int | None = None
        ) -> None:
            if tsize is not None:
                self.total = tsize
            self.update(b * bsize - self.n)

    with TqdmBarUpdater(
        unit='B', unit_scale=True, miniters=1, desc=url.split('/')[-1]
    ) as t:
        urllib.request.urlretrieve(url, filename, reporthook=t.update_to)

def center_image(image_np: np.ndarray) -> np.ndarray:
    cy, cx = center_of_mass(image_np)
    shift_y = int(round(image_np.shape[0] / 2 - cy))
    shift_x = int(round(image_np.shape[1] / 2 - cx))
    return np.roll(np.roll(image_np, shift_y, axis=0), shift_x, axis=1)


def post_process_image(
    image: Image.Image, target_size: int = 28, padding: int = 4
) -> Image.Image:
    arr = np.array(image)
    threshold = np.percentile(arr, 90)
    coords = np.column_stack(np.nonzero(arr > threshold))
    if coords.size == 0:
        return Image.new('L', (target_size, target_size), 0)
    top, left = coords.min(axis=0)
    bottom, right = coords.max(axis=0)
    padding_px = 2
    top = max(0, top - padding_px)
    left = max(0, left - padding_px)
    bottom = min(arr.shape[0] - 1, bottom + padding_px)
    right = min(arr.shape[1] - 1, right + padding_px)
    cropped = image.crop((left, top, right + 1, bottom + 1))
    cropped = cropped.filter(ImageFilter.GaussianBlur(0.8))
    scale = (target_size - 2 * padding) / max(cropped.size)
    new_size = tuple([int(dim * scale) for dim in cropped.size])
    resized = cropped.resize(new_size, Image.LANCZOS)
    new_image = Image.new('L', (target_size, target_size), 0)
    x_offset = (target_size - resized.size[0]) // 2
    y_offset = (target_size - resized.size[1]) // 2
    new_image.paste(resized, (x_offset, y_offset))
    arr = np.array(new_image, dtype=np.float32) / 255.0
    arr = center_image(arr)
    new_image = Image.fromarray((arr * 255).astype(np.uint8))
    return new_image
