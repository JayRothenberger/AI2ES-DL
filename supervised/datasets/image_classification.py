import tensorflow as tf
import tensorflow_datasets as tfds


def cats_dogs(batch_size=16, image_size=(128, 128, 3), cache=False, prefetch=4, **kwargs):
    setattr(tfds.image_classification.cats_vs_dogs, '_URL',
            "https://download.microsoft.com/download/3/E/1/3E1C3F21-ECDB-4869-8368-6DEBA77B919F"
            "/kagglecatsanddogs_5340.zip")
    ds = tfds.load('cats_vs_dogs', split='train[:80%]', as_supervised=True)
    test = tfds.load('cats_vs_dogs', split='train[80%:]', as_supervised=True)

    def preprocess_image(x, y, center=True):
        """
        Load in image from filename and resize to target shape.
        """
        image = tf.image.convert_image_dtype(x, tf.float32)
        image = tf.image.resize(image, (image_size[0], image_size[1]))
        if center:
            image = image - tf.reduce_mean(image)

        label = tf.one_hot(y, 2, dtype=tf.float32)

        return image, label

    ds = ds.map(
        lambda x, y: tf.py_function(preprocess_image, inp=[x, y, True], Tout=(tf.float32, tf.float32)),
        num_parallel_calls=tf.data.AUTOTUNE)

    test = test.map(
        lambda x, y: tf.py_function(preprocess_image, inp=[x, y, True], Tout=(tf.float32, tf.float32)),
        num_parallel_calls=tf.data.AUTOTUNE)

    return {'train': ds, 'val': test, 'test': test}


def deep_weeds(image_size=(128, 128, 3), center=True, **kwargs):
    ds = tfds.load('deep_weeds', split='train', as_supervised=True)
    test = tfds.load('deep_weeds', split='train', as_supervised=True)

    def preprocess_image(x, y):
        """
        Load in image from filename and resize to target shape.
        """
        image = tf.image.convert_image_dtype(x, tf.float32)
        image = tf.image.resize(image, (image_size[0], image_size[1]))
        if center:
            image = image - tf.reduce_mean(image)

        label = tf.one_hot(y, 9, dtype=tf.float32)

        return image, label

    ds = ds.map(
        lambda x, y: tf.py_function(preprocess_image, inp=[x, y], Tout=(tf.float32, tf.float32)),
        num_parallel_calls=tf.data.AUTOTUNE)

    test = test.map(
        lambda x, y: tf.py_function(preprocess_image, inp=[x, y], Tout=(tf.float32, tf.float32)),
        num_parallel_calls=tf.data.AUTOTUNE)

    return {'train': ds, 'val': test, 'test': test}


def citrus_leaves(batch_size=16, image_size=(128, 128, 3), cache=False, prefetch=4, **kwargs):
    ds = tfds.load('citrus_leaves', split='train[:80%]', as_supervised=True)
    plain = tfds.load('citrus_leaves', split='train[80%:]', as_supervised=True)

    def preprocess_image(x, y, center=True):
        """
        Load in image from filename and resize to target shape.
        """
        image = tf.image.convert_image_dtype(x, tf.float32)
        image = tf.image.resize(image, (image_size[0], image_size[1]))
        if center:
            image = image - tf.reduce_mean(image)

        label = tf.one_hot(y, 4, dtype=tf.float32)

        return image, label

    ds = ds.map(
        lambda x, y: tf.py_function(preprocess_image, inp=[x, y, True], Tout=(tf.float32, tf.float32)),
        num_parallel_calls=tf.data.AUTOTUNE)

    plain = plain.map(
        lambda x, y: tf.py_function(preprocess_image, inp=[x, y, True], Tout=(tf.float32, tf.float32)),
        num_parallel_calls=tf.data.AUTOTUNE)

    plain = plain.batch(batch_size).repeat()

    ds = ds.batch(batch_size)
    if cache:
        ds = ds.cache()
    ds = ds.shuffle(32)
    ds = ds.repeat()
    ds = ds.prefetch(prefetch)

    return ds, plain, plain