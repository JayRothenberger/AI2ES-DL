import math

import tensorflow as tf
import numpy as np
from keras.layers.preprocessing import image_preprocessing as image_ops


def fftfreqnd(h, w):
    """ Get bin values for discrete fourier transform of size (h, w, z)
    :param h: Required, first dimension size
    :param w: Optional, second dimension size
    :param z: Optional, third dimension size
    """
    fz = fx = 0
    fy = np.fft.fftfreq(h)

    fy = np.expand_dims(fy, -1)

    if w % 2 == 1:
        fx = np.fft.fftfreq(w)[: w // 2 + 2]
    else:
        fx = np.fft.fftfreq(w)[: w // 2 + 1]

    return tf.math.sqrt(fx * fx + fy * fy)


def get_spectrum(freqs, decay_power, ch, h, w=0, z=0):
    """ Samples a fourier image with given size and frequencies decayed by decay power
    :param freqs: Bin values for the discrete fourier transform
    :param decay_power: Decay power for frequency decay prop 1/f**d
    :param ch: Number of channels for the resulting mask
    :param h: Required, first dimension size
    :param w: Optional, second dimension size
    :param z: Optional, third dimension size
    """
    scale = np.ones(1) / (np.maximum(freqs, np.array([1. / max(w, h, z)])) ** decay_power)

    param_size = (ch, *freqs.shape, 2)
    param = tf.random.normal(tuple(param_size), dtype=tf.double)

    scale = tf.expand_dims(scale, -1)[None, :]

    return scale * tf.cast(param, tf.double)


def fout(x, y, alpha=.3, delta=3):
    shape = (x.shape[-3], x.shape[-2])

    freqs = fftfreqnd(*shape)
    spectrum = get_spectrum(freqs, delta, x.shape[-1], *shape)
    spectrum = tf.dtypes.complex(spectrum[:, 0], spectrum[:, 1])
    mask = np.real(np.fft.irfftn(spectrum, shape))
    mask = mask[:1, :shape[0], :shape[1]]
    flat_mask = tf.reshape(mask, (shape[0] * shape[1],))
    top_k, _ = tf.math.top_k(flat_mask, k=int(alpha * shape[0] * shape[1]))
    kth = tf.reduce_min(top_k)

    mask = tf.reshape(tf.where(mask <= kth, tf.ones_like(mask), tf.zeros_like(mask)), (1, shape[0], shape[1]))

    mask = np.moveaxis(mask, 0, -1)
    mask = tf.stack([mask for i in range(x.shape[0])])

    x = tf.multiply(mask, tf.cast(x, tf.double))

    x, y = tf.cast(x, tf.float32), tf.cast(y, tf.float32)

    return x, y


def foff_dset(train_dset, alpha=0.125, delta=3.0, **kwargs):
    return train_dset.map(lambda x, y: tf.py_function(fout, inp=[x, y, tf.cast(alpha, tf.double),
                                                                 tf.cast(delta, tf.double)],
                                                      Tout=(tf.float32, tf.float32)),
                          num_parallel_calls=tf.data.AUTOTUNE)


def arithmetic_blend(x, x_1, alpha=1.0):
    """
     sum a batch along the batch dimension weighted by a uniform random vector from the n simplex
      (convex hull of unit vectors)
    """
    rng = np.random.default_rng()
    weights = rng.dirichlet((alpha, alpha), 1)[0]

    x = np.stack([x, x_1])

    weights = np.array(weights, dtype=np.double).reshape(-1, 1)
    # sum along the 0th dimension weighted by weights
    x = tf.tensordot(weights, tf.cast(x, tf.double), (0, 0))[0]
    # return the convex combination
    return tf.cast(x, tf.float32)


def geometric_blend(x, x_1, alpha=1.0):
    """
     multiply two batches along the first dimension weighted by a uniform random vector from the n simplex
      (convex hull of unit vectors)
    then take the square root
    """
    x = x - tf.reduce_min(x)
    x_1 = x_1 - tf.reduce_min(x_1)

    x = x / tf.reduce_max(x)
    x_1 = x_1 / tf.reduce_max(x_1)

    rng = np.random.default_rng()
    weights = rng.dirichlet((alpha, alpha), 1)[0]
    w_0, w_1 = weights

    x = tf.pow(x, w_0)
    x_1 = tf.pow(x_1, w_1)

    x = tf.math.multiply(x, x_1)
    x = tf.pow(x, tf.cast(1.0 / tf.reduce_sum(weights), tf.float32))

    x = x - tf.reduce_mean(x)
    # return the convex combination
    return tf.cast(x, tf.float32)


def to_4d(image: tf.Tensor) -> tf.Tensor:
    """Converts an input Tensor to 4 dimensions.
  4D image => [N, H, W, C] or [N, C, H, W]
  3D image => [1, H, W, C] or [1, C, H, W]
  2D image => [1, H, W, 1]
  Args:
    image: The 2/3/4D input tensor.
  Returns:
    A 4D image tensor.
  Raises:
    `TypeError` if `image` is not a 2/3/4D tensor.
  """
    shape = tf.shape(image)
    original_rank = tf.rank(image)
    left_pad = tf.cast(tf.less_equal(original_rank, 3), dtype=tf.int32)
    right_pad = tf.cast(tf.equal(original_rank, 2), dtype=tf.int32)
    new_shape = tf.concat(
        [
            tf.ones(shape=left_pad, dtype=tf.int32),
            shape,
            tf.ones(shape=right_pad, dtype=tf.int32),
        ],
        axis=0,
    )
    return tf.reshape(image, new_shape)


def from_4d(image: tf.Tensor, ndims: tf.Tensor) -> tf.Tensor:
    """Converts a 4D image back to `ndims` rank."""
    shape = tf.shape(image)
    begin = tf.cast(tf.less_equal(ndims, 3), dtype=tf.int32)
    end = 4 - tf.cast(tf.equal(ndims, 2), dtype=tf.int32)
    new_shape = shape[begin:end]
    return tf.reshape(image, new_shape)


def wrap(image: tf.Tensor) -> tf.Tensor:
    """Returns 'image' with an extra channel set to all 1s."""
    shape = tf.shape(image)
    extended_channel = tf.expand_dims(tf.ones(shape[:-1], image.dtype), -1)
    extended = tf.concat([image, extended_channel], axis=-1)
    return extended


def unwrap(image: tf.Tensor, replace: float) -> tf.Tensor:
    """Unwraps an image produced by wrap.
  Where there is a 0 in the last channel for every spatial position,
  the rest of the three channels in that spatial dimension are grayed
  (set to 128).  Operations like translate and shear on a wrapped
  Tensor will leave 0s in empty locations.  Some transformations look
  at the intensity of values to do preprocessing, and we want these
  empty pixels to assume the 'average' value, rather than pure black.
  Args:
    image: A 3D Image Tensor with 4 channels.
    replace: A one or three value 1D tensor to fill empty pixels.
  Returns:
    image: A 3D image Tensor with 3 channels.
  """
    image_shape = tf.shape(image)
    # Flatten the spatial dimensions.
    flattened_image = tf.reshape(image, [-1, image_shape[-1]])

    # Find all pixels where the last channel is zero.
    alpha_channel = tf.expand_dims(flattened_image[..., 3], axis=-1)

    # replace = tf.concat([[replace], tf.ones([1], image.dtype)], 0)

    # Where they are zero, fill them in with 'replace'.
    flattened_image = tf.where(
        tf.equal(alpha_channel, 0),
        tf.ones_like(flattened_image, dtype=image.dtype) * replace,
        flattened_image)

    image = tf.reshape(flattened_image, image_shape)
    image = tf.slice(
        image,
        [0] * image.shape.rank,
        tf.concat([image_shape[:-1], [3]], -1))
    return image


def _convert_angles_to_transform(angles: tf.Tensor, image_width: tf.Tensor,
                                 image_height: tf.Tensor) -> tf.Tensor:
    """Converts an angle or angles to a projective transform.
  Args:
    angles: A scalar to rotate all images, or a vector to rotate a batch of
      images. This must be a scalar.
    image_width: The width of the image(s) to be transformed.
    image_height: The height of the image(s) to be transformed.
  Returns:
    A tensor of shape (num_images, 8).
  Raises:
    `TypeError` if `angles` is not rank 0 or 1.
  """
    angles = tf.convert_to_tensor(angles, dtype=tf.float32)
    if len(angles.get_shape()) == 0:  # pylint:disable=g-explicit-length-test
        angles = angles[None]
    elif len(angles.get_shape()) != 1:
        raise TypeError('Angles should have a rank 0 or 1.')
    x_offset = ((image_width - 1) -
                (tf.math.cos(angles) * (image_width - 1) - tf.math.sin(angles) *
                 (image_height - 1))) / 2.0
    y_offset = ((image_height - 1) -
                (tf.math.sin(angles) * (image_width - 1) + tf.math.cos(angles) *
                 (image_height - 1))) / 2.0
    num_angles = tf.shape(angles)[0]
    return tf.concat(
        values=[
            tf.math.cos(angles)[:, None],
            -tf.math.sin(angles)[:, None],
            x_offset[:, None],
            tf.math.sin(angles)[:, None],
            tf.math.cos(angles)[:, None],
            y_offset[:, None],
            tf.zeros((num_angles, 2), tf.dtypes.float32),
        ],
        axis=1,
    )


def _convert_translation_to_transform(translations: tf.Tensor) -> tf.Tensor:
    """Converts translations to a projective transform.
  The translation matrix looks like this:
    [[1 0 -dx]
     [0 1 -dy]
     [0 0 1]]
  Args:
    translations: The 2-element list representing [dx, dy], or a matrix of
      2-element lists representing [dx dy] to translate for each image. The
      shape must be static.
  Returns:
    The transformation matrix of shape (num_images, 8).
  Raises:
    `TypeError` if
      - the shape of `translations` is not known or
      - the shape of `translations` is not rank 1 or 2.
  """
    translations = tf.convert_to_tensor(translations, dtype=tf.float32)
    if translations.get_shape().ndims is None:
        raise TypeError('translations rank must be statically known')
    elif len(translations.get_shape()) == 1:
        translations = translations[None]
    elif len(translations.get_shape()) != 2:
        raise TypeError('translations should have rank 1 or 2.')
    num_translations = tf.shape(translations)[0]

    return tf.concat(
        values=[
            tf.ones((num_translations, 1), tf.dtypes.float32),
            tf.zeros((num_translations, 1), tf.dtypes.float32),
            -translations[:, 0, None],
            tf.zeros((num_translations, 1), tf.dtypes.float32),
            tf.ones((num_translations, 1), tf.dtypes.float32),
            -translations[:, 1, None],
            tf.zeros((num_translations, 2), tf.dtypes.float32),
        ],
        axis=1,
    )


def translate(image: tf.Tensor, translations) -> tf.Tensor:
    """Translates image(s) by provided vectors.
  Args:
    image: An image Tensor of type uint8.
    translations: A vector or matrix representing [dx dy].
  Returns:
    The translated version of the image.
  """
    transforms = _convert_translation_to_transform(translations)
    return transform(image, transforms=transforms)


def blend(image1: tf.Tensor, image2: tf.Tensor, factor: float) -> tf.Tensor:
    """Blend image1 and image2 using 'factor'.
  Factor can be above 0.0.  A value of 0.0 means only image1 is used.
  A value of 1.0 means only image2 is used.  A value between 0.0 and
  1.0 means we linearly interpolate the pixel values between the two
  images.  A value greater than 1.0 "extrapolates" the difference
  between the two pixel values, and we clip the results to values
  between 0 and 1.
  Args:
    image1: An image Tensor of type float32.
    image2: An image Tensor of type float32.
    factor: A floating point value above 0.0.
  Returns:
    A blended image Tensor of type float32.
  """
    if factor == 0.0:
        return tf.convert_to_tensor(image1)
    if factor == 1.0:
        return tf.convert_to_tensor(image2)

    image1 = tf.cast(image1, tf.float32)
    image2 = tf.cast(image2, tf.float32)

    difference = image2 - image1
    scaled = factor * difference

    # Do addition in float.
    temp = tf.cast(image1, tf.float32) + scaled

    # Interpolate
    if 0.0 < factor < 1.0:
        # Interpolation means we always stay within 0 and 255.
        return tf.cast(temp, tf.float32)

    # Extrapolate:
    #
    # We need to clip and then cast.
    return tf.cast(tf.clip_by_value(temp, 0.0, 1.0), tf.float32)


def transform(image: tf.Tensor, transforms) -> tf.Tensor:
    """Prepares input data for `image_ops.transform`."""
    original_ndims = tf.rank(image)
    transforms = tf.convert_to_tensor(transforms, dtype=tf.float32)
    if transforms.shape.rank == 1:
        transforms = transforms[None]
    image = to_4d(image)
    image = image_ops.transform(
        images=image, transforms=transforms, interpolation='nearest')
    return from_4d(image, original_ndims)


def identity(img, magnitude=None):
    return img


def translate_x(image: tf.Tensor, fraction: float, replace: float = 0.0) -> tf.Tensor:
    """Equivalent of PIL Translate in X dimension."""
    pixels = image[0].shape[0] * fraction
    image = translate(wrap(image), [-pixels, 0])
    return unwrap(image, replace)


def translate_y(image: tf.Tensor, fraction: float, replace: float = 0.0) -> tf.Tensor:
    """Equivalent of PIL Translate in Y dimension."""
    pixels = image[0].shape[0] * fraction
    image = translate(wrap(image), [0, -pixels])
    return unwrap(image, replace)


def shear_x(image: tf.Tensor, level: float, replace: float = 0.0) -> tf.Tensor:
    """Equivalent of PIL Shearing in X dimension."""
    # Shear parallel to x axis is a projective transform
    # with a matrix form of:
    # [1  level
    #  0  1].
    image = transform(
        image=wrap(image), transforms=[1., level, 0., 0., 1., 0., 0., 0.])
    return unwrap(image, replace)


def shear_y(image: tf.Tensor, level: float, replace: float = 0.0) -> tf.Tensor:
    """Equivalent of PIL Shearing in Y dimension."""
    # Shear parallel to y axis is a projective transform
    # with a matrix form of:
    # [1  0
    #  level  1].
    image = transform(
        image=wrap(image), transforms=[1., 0., 0., level, 1., 0., 0., 0.])
    return unwrap(image, replace)


def rotate(image: tf.Tensor, range: float) -> tf.Tensor:
    """Rotates the image by degrees either clockwise or counterclockwise.
  Args:
    image: An image Tensor of type float32.
    range: Float, a scalar angle in [0, 1] to rotate all images by. If
      degrees is positive the image will be rotated clockwise otherwise it will
      be rotated counterclockwise.
  Returns:
    The rotated version of image.
  """
    range = range if np.random.choice([True, False], 1) else 0 - range
    degrees = range * 360
    # Convert from degrees to radians.
    degrees_to_radians = math.pi / 180.0
    radians = tf.cast(degrees * degrees_to_radians, tf.float32)

    original_ndims = tf.rank(image)
    image = to_4d(image)

    image_height = tf.cast(tf.shape(image)[1], tf.float32)
    image_width = tf.cast(tf.shape(image)[2], tf.float32)
    transforms = _convert_angles_to_transform(
        angles=radians, image_width=image_width, image_height=image_height)
    # In practice, we should randomize the rotation degrees by flipping
    # it negatively half the time, but that's done on 'degrees' outside
    # of the function.
    image = transform(image, transforms=transforms)
    return from_4d(image, original_ndims)


def flip_ud(image: tf.Tensor, range: float) -> tf.Tensor:
    if np.random.uniform(0, 1, 1) < range:
        return tf.image.flip_up_down(image)
    else:
        return image


def flip_lr(image: tf.Tensor, range: float) -> tf.Tensor:
    if np.random.uniform(0, 1, 1) < range:
        return tf.image.flip_left_right(image)
    else:
        return image


def custom_rand_augment(x, y, M=.2, N=1, alpha=1.0):
    """
    performs random augmentation with magnitude M for N iterations

    :param y:
    :param x:
    :param N:
    :param M:
    :param alpha:
    :return: augmented image
    """
    blend_fns = [arithmetic_blend, geometric_blend]
    transforms = [identity, rotate, shear_x, shear_y, translate_x, translate_y, flip_lr]
    # needs to take a rank 3 numpy tensor, and return a tensor of the same rank
    for op in np.random.choice(transforms, N):
        x_1 = op(x, np.random.uniform(0, M))
        for fn in np.random.choice(blend_fns, 1):
            x = fn(x_1, x, alpha)
    return x, y


def custom_rand_augment_dset(ds, M=.2, N=1):
    return ds.map(
        lambda x, y: tf.py_function(custom_rand_augment, inp=[x, y, M, N], Tout=(tf.float32, tf.float32)),
        num_parallel_calls=tf.data.AUTOTUNE)


def add_gaussian_noise(x, y, std=0.01):
    return x + tf.random.normal(shape=x.shape, mean=0.0, stddev=std, dtype=tf.float32, seed=42), tf.cast(y, tf.float32)


def add_gaussian_noise_dset(ds, std=0.01):
    return ds.map(
        lambda x, y: tf.py_function(add_gaussian_noise, inp=[x, y, std], Tout=(tf.float32, tf.float32)),
        num_parallel_calls=tf.data.AUTOTUNE)
