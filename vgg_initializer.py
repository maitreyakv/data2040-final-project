"""
Function to initialize a VGG-like CNN or CNN-LSTM
"""

__author__ = "Maitreya Venkataswamy"

import tensorflow as tf


def _add_block(out, n_conv, n_filters, dropout):
    for i in range(n_conv):
        out = tf.keras.layers.Conv2D(
            filters=n_filters,
            kernel_size=(3,3),
            padding="same",
            activation="relu"
        )(out)
        out = tf.keras.layers.Dropout(dropout)(out)
        out = tf.keras.layers.BatchNormalization()(out)

    out = tf.keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2))(out)

    return out


def _add_block_3d(out, n_conv, n_filters, dropout):
    for i in range(n_conv):
        out = tf.keras.layers.Conv3D(
            filters=n_filters,
            kernel_size=(3,3,3),
            padding="same",
            activation="relu"
        )(out)
        out = tf.keras.layers.Dropout(dropout)(out)
        out = tf.keras.layers.BatchNormalization()(out)

    out = tf.keras.layers.MaxPool3D(pool_size=(1,2,2), strides=(1,2,2))(out)

    return out


def _add_block_lstm(out, n_conv, n_filters, dropout):
    for i in range(n_conv):
        out = tf.keras.layers.ConvLSTM2D(
            filters=n_filters,
            kernel_size=(3,3),
            padding="same",
            activation="tanh",
            return_sequences=True,
            stateful=True
        )(out)
        out = tf.keras.layers.Dropout(dropout)(out)
        out = tf.keras.layers.LayerNormalization()(out)

    out = tf.keras.layers.MaxPool3D(pool_size=(1,2,2), strides=(1,2,2))(out)

    return out


def initialize_vgg(vgg_size, batch_size, img_size, dropout=0.25):
    inp = tf.keras.Input(shape=img_size, batch_size=batch_size)

    out = inp

    out = _add_block(out, 2, 64, dropout=dropout)
    out = _add_block(out, 2, 128, dropout=dropout)

    out = _add_block(out, vgg_size, 256, dropout=dropout)
    out = _add_block(out, vgg_size, 512, dropout=dropout)
    out = _add_block(out, vgg_size, 512, dropout=dropout)

    out = tf.keras.layers.Flatten()(out)
    out = tf.keras.layers.Dense(2, activation="softmax")(out)

    return tf.keras.Model(inputs=inp, outputs=out)


def initialize_vgg_3d(vgg_size, batch_size, img_size, seq_size=1, filter_reduction_fac=1, dropout=0.25):
    inp = tf.keras.Input(shape=(seq_size,) + img_size, batch_size=batch_size)

    out = inp

    out = _add_block_3d(out, 2, 64 // filter_reduction_fac, dropout=dropout)
    out = _add_block_3d(out, 2, 128 // filter_reduction_fac, dropout=dropout)

    out = _add_block_3d(out, vgg_size, 256 // filter_reduction_fac, dropout=dropout)
    out = _add_block_3d(out, vgg_size, 512 // filter_reduction_fac, dropout=dropout)
    out = _add_block_3d(out, vgg_size, 512 // filter_reduction_fac, dropout=dropout)

    shape = out.shape
    out = tf.keras.layers.Reshape((seq_size, shape[2]*shape[3]*shape[4]))(out)
    out = tf.keras.layers.Dense(2, activation="softmax")(out)

    return tf.keras.Model(inputs=inp, outputs=out)


def initialize_vgg_lstm(vgg_size, batch_size, img_size, seq_size=1, filter_reduction_fac=1, dropout=0.25):
    inp = tf.keras.Input(shape=(seq_size,) + img_size, batch_size=batch_size)

    out = inp

    out = tf.keras.layers.Lambda(lambda x: x / 255.)(out)

    out = _add_block_lstm(out, 2, 64 // filter_reduction_fac, dropout)
    out = _add_block_lstm(out, 2, 128 // filter_reduction_fac, dropout)

    out = _add_block_lstm(out, vgg_size, 256 // filter_reduction_fac, dropout)
    out = _add_block_lstm(out, vgg_size, 512 // filter_reduction_fac, dropout)
    out = _add_block_lstm(out, vgg_size, 512 // filter_reduction_fac, dropout)

    shape = out.shape
    out = tf.keras.layers.Reshape((seq_size, shape[2]*shape[3]*shape[4]))(out)
    out = tf.keras.layers.Dense(2, activation="softmax")(out)

    return tf.keras.Model(inputs=inp, outputs=out)


if __name__ == "__main__":
    model = initialize_vgg(2, batch_size=None, img_size=(128, 128, 3))
    model.summary()

    model = initialize_vgg_3d(2, batch_size=None, img_size=(128, 128, 3), seq_size=128, filter_reduction_fac=3)
    model.summary()

    model = initialize_vgg_lstm(2, batch_size=64, img_size=(128, 128, 3), seq_size=2, filter_reduction_fac=8)
    model.summary()
