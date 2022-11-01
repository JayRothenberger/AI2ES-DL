import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Flatten, Conv2D, MaxPooling2D, Dense, Input, Concatenate, Dropout, SpatialDropout2D, \
    MultiHeadAttention, Add, BatchNormalization, LayerNormalization, Conv1D, Reshape, Cropping2D, ZeroPadding3D, \
    GlobalMaxPooling2D, GlobalAveragePooling2D, Lambda, Average, SeparableConv2D, DepthwiseConv2D, UpSampling2D, \
    Conv2DTranspose, AveragePooling2D

from tensorflow.keras.applications import EfficientNetB0, ResNet50V2, VGG16, MobileNetV3Small
from time import time

HARDSWISH = lambda x: x * tf.nn.relu6(x + 3) / 6
MISH = lambda x: x * tf.nn.tanh(tf.nn.softplus(x))


def build_keras_application(application, image_size=(256, 256, 3), learning_rate=1e-4, loss='categorical_crossentropy',
                            n_classes=10, dropout=0, **kwargs):
    inputs = Input(image_size)

    try:
        model = application(input_tensor=inputs, include_top=False, weights='imagenet', pooling='avg',
                            include_preprocessing=False)
    except TypeError as t:
        # no preprocessing for ResNet
        model = application(input_tensor=inputs, include_top=False, weights='imagenet', pooling='avg')

    outputs = Dense(n_classes, activation='softmax')(Dropout(dropout)(Flatten()(model.output)))

    model = tf.keras.models.Model(inputs=[inputs], outputs=[outputs])

    opt = tf.keras.optimizers.Nadam(learning_rate=learning_rate,
                                    beta_1=0.9, beta_2=0.999,
                                    epsilon=None, decay=0.99)
    opt = tf.keras.mixed_precision.LossScaleOptimizer(opt)

    accuracy = 'sparse_categorical_accuracy' if loss == 'sparse_categorical_crossentropy' else 'categorical_accuracy'
    # compile the model
    model.compile(loss=loss,
                  optimizer=opt,
                  metrics=[accuracy])

    return model


def build_special_VGG(image_size, classes=(3, 4, 5), learning_rate=1e-3, loss='categorical_crossentropy'):
    inputs = Input(image_size)
    # load VGG16 with imagenet weights
    model = VGG16(input_tensor=inputs, include_top=False, weights='imagenet', pooling='avg',
                  include_preprocessing=False)

    outputs = []
    for n_classes in classes:
        outputs.append(Dense(n_classes, activation='softmax')(Flatten()(model.output)))
    # 'outputs' can take a list of output tensors
    model = tf.keras.models.Model(inputs=[inputs], outputs=outputs)
    # you can ignore these next two statements
    opt = tf.keras.optimizers.Nadam(learning_rate=learning_rate,
                                    beta_1=0.9, beta_2=0.999,
                                    epsilon=None, decay=0.99)
    opt = tf.keras.mixed_precision.LossScaleOptimizer(opt)
    # select the correct kind of accuracy
    accuracy = 'sparse_categorical_accuracy' if loss == 'sparse_categorical_crossentropy' else 'categorical_accuracy'
    # compile the model
    model.compile(loss=loss,
                  optimizer=opt,
                  metrics=[accuracy])
    # return the compiled model
    return model


def build_EfficientNetB0(**kwargs):
    return build_keras_application(EfficientNetB0, **kwargs)


def build_ResNet50V2(**kwargs):
    return build_keras_application(ResNet50V2, **kwargs)


def build_MobileNetV3Small(**kwargs):
    return build_keras_application(MobileNetV3Small, **kwargs)


def build_hallucinetv4_upcycle_plus_plus(conv_filters,
                                         conv_size,
                                         attention_heads,
                                         learning_rate,
                                         image_size,
                                         iterations=24,
                                         loss='categorical_crossentropy',
                                         pooling='max',
                                         l1=None, l2=None,
                                         activation=lambda x: x * tf.nn.relu6(x + 3) / 6,
                                         n_classes=10,
                                         downsample=3,
                                         dropout=0.1,
                                         depth=2,
                                         skips=2,
                                         **kwargs):
    if isinstance(conv_filters, str):
        conv_filters = [int(i) for i in conv_filters.strip('[]').split(', ')]
    if isinstance(conv_size, str):
        conv_size = [int(i) for i in conv_size.strip('[]').split(', ')]
    if isinstance(attention_heads, str):
        dense_layers = [int(i) for i in attention_heads.strip('[]').split(', ')]

    inputs = Input(image_size)

    conv_params = {
        'use_bias': False,
        'kernel_initializer': tf.keras.initializers.GlorotUniform(),
        'bias_initializer': 'zeros',
        'kernel_regularizer': tf.keras.regularizers.L1L2(l1=l1, l2=l2),
        'bias_regularizer': tf.keras.regularizers.L1L2(l1=l1, l2=l2),
        'padding': 'same'
    }

    x = inputs

    # real = tf.math.real(fft)
    # imag = tf.math.imag(fft)

    # mag = tf.cast(tf.math.sqrt(real ** 2 + imag ** 2), tf.float32)
    # phase = tf.cast(tf.math.atan(imag / real), tf.float32)

    # x = Concatenate()([x, mag, phase, TFPositionalEncoding2D(2)(x)])

    x = Lambda(lambda z: tf.pad(z, ((0, 0), (0, 0), (0, 0), (0, conv_filters[0] - z.shape[-1]))))(x)

    for block in range(len(conv_filters)):
        thrifty_exp = Conv2D(filters=conv_filters[block], kernel_size=1, activation=None, **conv_params)

        thrifty_convs = [
            Conv2D(conv_filters[block], kernel_size=3, activation=activation, **conv_params),
            Conv2D(conv_filters[block], kernel_size=3, activation=None, **conv_params)
        ]

        # rconv, iconv = Conv2D(conv_filters[block], kernel_size=3, activation=activation, **conv_params), \
        #                Conv2D(conv_filters[block], kernel_size=3, activation=activation, **conv_params)

        thrifty_inv3 = UpSampling2D(2)

        thrifty_squeeze = Conv2D(filters=conv_filters[block], kernel_size=1, activation=activation, **conv_params)

        def thrifty_imb(z, output_dim):
            exp, exc, inv, sqzn = thrifty_exp, thrifty_convs, thrifty_inv3, thrifty_squeeze
            inp = z
            z = MaxPooling2D(2, 2, 'same')(z)
            z = Concatenate()([inv(e(z)) for e in exc])
            z = Conv2D(filters=output_dim // 2, kernel_size=1, activation=None, **conv_params)(z)
            z = BatchNormalization()(z)
            z = activation(z)
            z0 = Concatenate()([e(inp) for e in exc] + [inp])
            z0 = Conv2D(filters=output_dim // 2, kernel_size=1, activation=None, **conv_params)(z0)
            z0 = BatchNormalization()(z0)
            z0 = activation(z0)
            z = Concatenate(name=f"chkpt_{time()}")([z0, z])
            return z

        """
        def thrifty_freq_imb(z):
            inp = z
            fft = tf.signal.fft2d(tf.cast(z, tf.complex64))
            # imaginary, real
            i, r = tf.math.imag(fft), tf.math.real(fft)
            # mag
            mag = tf.math.sqrt(r ** 2 + i ** 2)
            # phase
            phase = tf.math.atan(i / tf.math.maximum(r, tf.ones_like(r) * 2 ** (-16)))

            def to_transform(t):
                t = AveragePooling2D(8)(t)
                t = Concatenate()([t, TFPositionalEncoding2D(2)(t)])
                return Reshape((int(tf.reduce_prod(t.shape[1:3])), t.shape[-1]))(t)

            mag, phase, flat_inp = to_transform(mag), \
                                   to_transform(phase), \
                                   to_transform(inp)

            fft = MultiHeadAttention(n_classes,
                                     4,
                                     kernel_regularizer=tf.keras.regularizers.L1L2(l1=l1, l2=l2),
                                     dropout=dropout, output_shape=(conv_filters[block], ))(mag, flat_inp, phase)

            fft = Reshape((image_size[0] // 8, image_size[1] // 8, conv_filters[block]))(fft)
            fft = UpSampling2D(8)(fft)

            fft = Concatenate()([fft, inp])
            z = thrifty_imb(fft, conv_filters[block])
            z = Conv2D(filters=conv_filters[block], kernel_size=1, activation=activation, **conv_params)(z)
            z = BatchNormalization(name=f"chkpt_{time()}")(z)
            return z
        """
        prev_layers = [thrifty_imb(x, conv_filters[block])]
        for i in range(iterations):
            x = thrifty_imb(x, conv_filters[block])
            prev_layers.append(x)
            x = Add()(prev_layers[-skips:])
            x = BatchNormalization()(x)
        x = Add()(prev_layers)
    # this dense operation is over an input tensor of size (batch, width, height, channels)
    # semantic segmentation output with extra (irrelevant) channel
    x = Dense(n_classes + 1, activation='softmax', use_bias=False)(x)
    # reduce sum over width / height
    y = Lambda(
        lambda z: tf.reduce_sum(
            tf.stack([z[:, :, :, -1] / (n_classes * 2) for i in range(n_classes)], -1), axis=(1, 2)
        )
    )(x)

    x = Lambda(lambda z: tf.reduce_sum(z[:, :, :, :-1], axis=(1, 2)))(x)
    x = Add()([x, y, tf.ones_like(x) * (2 ** (-10))])
    # want to re-normalize without destroying the gradient
    outputs = Lambda(lambda z: tf.linalg.normalize(z, 1, axis=-1)[0])(x)
    # outputs shape is (batch, n_classes)

    accuracy = 'sparse_categorical_accuracy' if loss == 'sparse_categorical_crossentropy' else 'categorical_accuracy'
    # TODO: abstract away the optimizer and loss scale optimizer
    opt = tf.keras.optimizers.Nadam(learning_rate=learning_rate,
                                    beta_1=0.9, beta_2=0.999,
                                    epsilon=None, decay=0.99)

    opt = tf.keras.mixed_precision.LossScaleOptimizer(opt)

    model = tf.keras.Model(inputs=[inputs], outputs=[outputs],
                           name=f'thrifty_model_{"%02d" % time()}')

    model.compile(loss=loss,
                  optimizer=opt,
                  metrics=[accuracy])

    return model
