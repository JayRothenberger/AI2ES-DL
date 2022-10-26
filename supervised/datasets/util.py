import os

import pandas as pd
import tensorflow as tf


def df_to_image_dataset(df, shuffle=False, image_size=(256, 256), batch_size=16, prefetch=1, seed=42,
                        class_mode='sparse', center=False, cache=True, repeat=True, batch=True, **kwargs):
    """
        convert a dataframe with columns ['image_path', 'class'] to a tf dataset object

    :param df: dataframe with columns ['image_path', 'class']
    :param shuffle:
    :param image_size:
    :param batch_size:
    :param prefetch:
    :param seed:
    :param class_mode:
    :param center:
    :param cache:
    :param repeat:
    :param batch:
    :param kwargs:
    :return:
    """
    if df is None:
        return None

    def preprocess_image(item, target_shape):
        """
        Load in image from filename and resize to target shape.
        """

        filename, label = item[0], item[1]

        image_bytes = tf.io.read_file(filename)
        image = tf.io.decode_image(image_bytes)  # this line does not work kinda
        image = tf.image.convert_image_dtype(image, tf.float32)
        image = tf.image.resize(image, target_shape)
        if center:
            image = image - tf.reduce_mean(image)

        if class_mode == 'categorical':
            label = tf.one_hot(tf.strings.to_number(label, tf.dtypes.int32), 3, dtype=tf.float32)
        elif class_mode == 'sparse':
            label = tf.strings.to_number(label, tf.dtypes.int32)
        else:
            raise ValueError('improper class mode')

        return image, label

    try:
        df['class'] = df['class'].astype(int).astype(str)
    except KeyError as e:
        print('setting class manually to 0...')
        df['class'] = ['0' for i in range(len(df))]
    except ValueError as e:
        print('setting class manually to 0...')
        df['class'] = ['0' for i in range(len(df))]
    slices = df.to_numpy()
    out_type = tf.int32 if class_mode == 'sparse' else tf.float32

    ds = tf.data.Dataset.from_tensor_slices(
        slices
    )

    ds = ds.map(lambda x:
                tf.py_function(func=preprocess_image,
                               inp=[x, image_size],
                               Tout=(tf.float32, out_type)),
                num_parallel_calls=tf.data.AUTOTUNE)

    return ds


def get_file_name_tuples(dirname, classmap):
    """
    Takes a directory name and a dictionary that serves as a map between the class names and the class integger labels,
    and returns a list of tuples (filename, class)

    :param dirname: a directory name local to the running process
    :param classmap: a dictionary of str: int
    :return: a list of tuples (filename, class) parsed from dirname
    """
    return [(f'{dirname}/{f}', classmap[dirname.split('/')[-1]]) for f in os.listdir(dirname)]


def df_from_dirlist(dirlist):
    """
    Creates a pandas dataframe from the files in a list of directories

    :param dirlist: a list of directories where the last sub-directory is the class label for the images within
    :return: a pandas dataframe with columns=['filepath', 'class']
    """
    # remove trailing slash in path (it will be taken care of later)
    dirlist = [dirname if dirname[-1] != '/' else dirname[:-2] for dirname in dirlist]

    # determine the list of classes by directory names
    # E.g. "foo/bar/cat - (split) -> ['foo', 'bar', 'cat'][-1] == 'cat'
    classes = sorted(list(set([dirname.split('/')[-1] for dirname in dirlist])))

    class_map = {c: str(i) for (i, c) in enumerate(classes)}
    # find all of the file names
    names = sum([get_file_name_tuples(dirname, class_map) for dirname in dirlist], [])
    return pd.DataFrame(names, columns=['filepath', 'class']), class_map


def get_cv_rotation(dirlist, rotation=0, k=5, train_fraction=1):
    """
    Return train, val, test dataframes for a cross-validation rotation

    :param dirlist: list of directories containing the data
    :param rotation: rotation of cross validation
    :param k: number of folds for cross_validation
    :param train_fraction: fraction of training data to use
    :return: returns three data generators train, val, test
    """
    assert isinstance(k, int), "k should be integer"
    assert isinstance(rotation, int), "rotation should be integer"
    assert rotation < k, "rotation must always be less than k"
    assert rotation >= 0, "rotation must be strictly nonnegative"
    assert k >= 1, "k must be strictly positive"

    # want to split this into k shards and then
    df = df_from_dirlist(dirlist)
    # shuffle the dataframe
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    # shard the dataframe
    shards = [df[i * (len(df) // (k + 1)):(i + 1) * (len(df) // (k + 1))] if i < k else df[i * (len(df) // (k + 1)):]
              for i in range(k + 1)]
    # test is always the last shard
    test, shards = shards[-1], shards[:-1]
    val = shards[rotation]
    train = pd.concat([shards[i] for i in range(k) if i != rotation])

    return train, val, test
