import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Flatten, Conv2D, MaxPooling2D, Dense, Input, Concatenate, Dropout, SpatialDropout2D, \
    MultiHeadAttention, Add, BatchNormalization, LayerNormalization, Conv1D, Reshape, Cropping2D, ZeroPadding3D, \
    GlobalMaxPooling2D, GlobalAveragePooling2D, Lambda, Average, SeparableConv2D, DepthwiseConv2D, UpSampling2D, \
    Conv2DTranspose, AveragePooling2D

from tensorflow.keras.applications import EfficientNetB0, ResNet50V2, VGG16, MobileNetV3Small


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
