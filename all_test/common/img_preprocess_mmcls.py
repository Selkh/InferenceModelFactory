import cv2
import numpy as np
from PIL import Image
from cv2 import (IMREAD_COLOR, IMREAD_GRAYSCALE, IMREAD_IGNORE_ORIENTATION,
                 IMREAD_UNCHANGED)

imread_flags = {
    'color': IMREAD_COLOR,
    'grayscale': IMREAD_GRAYSCALE,
    'unchanged': IMREAD_UNCHANGED,
    'color_ignore_orientation': IMREAD_IGNORE_ORIENTATION | IMREAD_COLOR,
    'grayscale_ignore_orientation':
    IMREAD_IGNORE_ORIENTATION | IMREAD_GRAYSCALE
}

cv2_interp_codes = {
    'nearest': cv2.INTER_NEAREST,
    'bilinear': cv2.INTER_LINEAR,
    'bicubic': cv2.INTER_CUBIC,
    'area': cv2.INTER_AREA,
    'lanczos': cv2.INTER_LANCZOS4
}

pillow_interp_codes = {
    'nearest': Image.NEAREST,
    'bilinear': Image.BILINEAR,
    'bicubic': Image.BICUBIC,
    'box': Image.BOX,
    'lanczos': Image.LANCZOS,
    'hamming': Image.HAMMING
}


def is_str(x):
    """Whether the input is an string instance.
    Note: This method is deprecated since python 2 is no longer supported.
    """
    return isinstance(x, str)


def imfrombytes(
    content,
    flag,
    channel_order='bgr',
):
    img_np = np.frombuffer(content, np.uint8)
    flag = imread_flags[flag] if is_str(flag) else flag
    img = cv2.imdecode(img_np, flag)
    if flag == IMREAD_COLOR and channel_order == 'rgb':
        cv2.cvtColor(img, cv2.COLOR_BGR2RGB, img)
    return img


def imresize(img,
             size,
             return_scale=False,
             interpolation='bilinear',
             backend='cv2',
             out=None):

    h, w = img.shape[:2]
    if backend not in ['cv2', 'pillow']:
        raise ValueError(f'backend: {backend} is not supported for resize.'
                         f"Supported backends are 'cv2', 'pillow'")
    if backend == 'pillow':
        assert img.dtype == np.uint8, 'Pillow backend only support uint8 type'
        pil_image = Image.fromarray(img)
        pil_image = pil_image.resize(size, pillow_interp_codes[interpolation])
        resized_img = np.array(pil_image)
    else:
        resized_img = cv2.resize(img,
                                 size,
                                 dst=out,
                                 interpolation=cv2_interp_codes[interpolation])
    if not return_scale:
        return resized_img
    else:
        w_scale = size[0] / w
        h_scale = size[1] / h
        return resized_img, w_scale, h_scale


def Resize(img, size, interpolation, adaptive_side):
    if size[1] == -1:
        adaptive_resize = True

    ignore_resize = False
    if adaptive_resize:
        h, w = img.shape[:2]
        target_size = size[0]

        condition_ignore_resize = {
            'short': min(h, w) == target_size,
            'long': max(h, w) == target_size,
            'height': h == target_size,
            'width': w == target_size
        }

        if condition_ignore_resize[adaptive_side]:
            ignore_resize = True
        elif any([
                adaptive_side == 'short' and w < h,
                adaptive_side == 'long' and w > h,
                adaptive_side == 'width',
        ]):
            width = target_size
            height = int(target_size * h / w)
        else:
            height = target_size
            width = int(target_size * w / h)
    else:
        height, width = size
    if not ignore_resize:
        img = imresize(
            img,
            size=(width, height),
            interpolation=interpolation,
            return_scale=False,
            backend='pillow',
        )
    return img


def bbox_scaling(bboxes, scale, clip_shape=None):
    """Scaling bboxes w.r.t the box center.
    Args:
        bboxes (ndarray): Shape(..., 4).
        scale (float): Scaling factor.
        clip_shape (tuple[int], optional): If specified, bboxes that exceed the
            boundary will be clipped according to the given shape (h, w).
    Returns:
        ndarray: Scaled bboxes.
    """
    if float(scale) == 1.0:
        scaled_bboxes = bboxes.copy()
    else:
        w = bboxes[..., 2] - bboxes[..., 0] + 1
        h = bboxes[..., 3] - bboxes[..., 1] + 1
        dw = (w * (scale - 1)) * 0.5
        dh = (h * (scale - 1)) * 0.5
        scaled_bboxes = bboxes + np.stack((-dw, -dh, dw, dh), axis=-1)
    if clip_shape is not None:
        return bbox_clip(scaled_bboxes, clip_shape)
    else:
        return scaled_bboxes


def bbox_clip(bboxes, img_shape):
    """Clip bboxes to fit the image shape.
    Args:
        bboxes (ndarray): Shape (..., 4*k)
        img_shape (tuple[int]): (height, width) of the image.
    Returns:
        ndarray: Clipped bboxes.
    """
    assert bboxes.shape[-1] % 4 == 0
    cmin = np.empty(bboxes.shape[-1], dtype=bboxes.dtype)
    cmin[0::2] = img_shape[1] - 1
    cmin[1::2] = img_shape[0] - 1
    clipped_bboxes = np.maximum(np.minimum(bboxes, cmin), 0)
    return clipped_bboxes


def imcrop(img, bboxes, scale=1.0, pad_fill=None):
    """Crop image patches.
    3 steps: scale the bboxes -> clip bboxes -> crop and pad.
    Args:
        img (ndarray): Image to be cropped.
        bboxes (ndarray): Shape (k, 4) or (4, ), location of cropped bboxes.
        scale (float, optional): Scale ratio of bboxes, the default value
            1.0 means no padding.
        pad_fill (Number | list[Number]): Value to be filled for padding.
            Default: None, which means no padding.
    Returns:
        list[ndarray] | ndarray: The cropped image patches.
    """
    chn = 1 if img.ndim == 2 else img.shape[2]
    if pad_fill is not None:
        if isinstance(pad_fill, (int, float)):
            pad_fill = [pad_fill for _ in range(chn)]
        assert len(pad_fill) == chn

    _bboxes = bboxes[None, ...] if bboxes.ndim == 1 else bboxes
    scaled_bboxes = bbox_scaling(_bboxes, scale).astype(np.int32)
    clipped_bbox = bbox_clip(scaled_bboxes, img.shape)

    patches = []
    for i in range(clipped_bbox.shape[0]):
        x1, y1, x2, y2 = tuple(clipped_bbox[i, :])
        if pad_fill is None:
            patch = img[y1:y2 + 1, x1:x2 + 1, ...]
        else:
            _x1, _y1, _x2, _y2 = tuple(scaled_bboxes[i, :])
            if chn == 1:
                patch_shape = (_y2 - _y1 + 1, _x2 - _x1 + 1)
            else:
                patch_shape = (_y2 - _y1 + 1, _x2 - _x1 + 1, chn)
            patch = np.array(pad_fill, dtype=img.dtype) * np.ones(
                patch_shape, dtype=img.dtype)
            x_start = 0 if _x1 >= 0 else -_x1
            y_start = 0 if _y1 >= 0 else -_y1
            w = x2 - x1 + 1
            h = y2 - y1 + 1
            patch[y_start:y_start + h, x_start:x_start + w,
                  ...] = img[y1:y1 + h, x1:x1 + w, ...]
        patches.append(patch)

    if bboxes.ndim == 1:
        return patches[0]
    else:
        return patches


def CenterCrop(img, crop_size, efficientnet_style, crop_padding,
               interpolation):
    crop_height, crop_width = crop_size[0], crop_size[1]

    # img.shape has length 2 for grayscale, length 3 for color
    img_height, img_width = img.shape[:2]
    # https://github.com/tensorflow/tpu/blob/master/models/official/efficientnet/preprocessing.py#L118 # noqa
    if efficientnet_style:
        img_short = min(img_height, img_width)
        crop_height = crop_height / (crop_height + crop_padding) * img_short
        crop_width = crop_width / (crop_width + crop_padding) * img_short

    y1 = max(0, int(round((img_height - crop_height) / 2.)))
    x1 = max(0, int(round((img_width - crop_width) / 2.)))
    y2 = min(img_height, y1 + crop_height) - 1
    x2 = min(img_width, x1 + crop_width) - 1

    # crop the image
    img = imcrop(img, bboxes=np.array([x1, y1, x2, y2]))
    if efficientnet_style:
        img = imresize(img,
                       tuple(crop_size[::-1]),
                       interpolation=interpolation)
    return img


def imnormalize(img, mean, std, to_rgb=True):
    # cv2 inplace normalization does not accept uint8
    img = img.copy().astype(np.float32)
    assert img.dtype != np.uint8
    mean = np.float64(mean.reshape(1, -1))
    stdinv = 1 / np.float64(std.reshape(1, -1))
    if to_rgb:
        cv2.cvtColor(img, cv2.COLOR_BGR2RGB, img)  # inplace
    cv2.subtract(img, mean, img)  # inplace
    cv2.multiply(img, stdinv, img)  # inplace
    return img


def Normalize(results, mean, std, to_rgb):
    results = imnormalize(results, mean, std, to_rgb)
    return results