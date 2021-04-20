"""
Function to create a Tensorflow Dataset from the TFRecord file
"""

__author__ = "Maitreya Venkataswamy"

import tensorflow as tf
import json


def create_data_loader(tfrecord_file, metadata_file, valid_size=0.5, seq_size=1, batch_size=1,
                       n_channels=1):
    """Creates the training and validation data for the RatSI classification problem"""
    # Read the TFRecord file into a TFRecord dataset
    dataset = tf.data.TFRecordDataset(tfrecord_file)

    # Read in the dataset metadata
    with open(metadata_file, "r") as f:
        metadata = json.load(f)

    # Description of the TFRecord examples
    frame_feature_desc = {
        "frame": tf.io.FixedLenFeature([], tf.string),
        "label": tf.io.FixedLenFeature([], tf.int64)
    }

    # Function to parse the examples
    @tf.autograph.experimental.do_not_convert
    def _parse_frame_fn(example):
        example = tf.io.parse_single_example(example, frame_feature_desc)
        frame = tf.io.decode_jpeg(example["frame"], channels=n_channels)
        frame = tf.cast(frame, tf.float32)
        label = tf.cast(example["label"], tf.int32)
        return frame, label

    # Map the parsing function over the whole dataset
    dataset = dataset.map(_parse_frame_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    # If a sequence length was specified, contruct sequences using the batching function
    if seq_size > 1:
        dataset = dataset.batch(seq_size, drop_remainder=True)

    # Apply batching to the dataset
    if seq_size > 1:
        dataset = dataset.window(batch_size, 1, seq_size, drop_remainder=True)
        dataset = dataset.flat_map(
            lambda img, label: tf.data.Dataset.zip((
                img.batch(batch_size),
                label.batch(batch_size)
            ))
        )
    else:
        dataset = dataset.batch(batch_size)

    # Compute the number of batches
    n_batches = tf.math.ceil(metadata["n_records"] / (seq_size * batch_size))

    # Split the dataset into validation and training data
    dataset_train = dataset.take(int(valid_size * n_batches))
    dataset_valid = dataset.skip(int(valid_size * n_batches))
    dataset_valid = dataset_valid.take(int((1. - valid_size) * n_batches))

    # Prefetch data
    dataset_train = dataset_train.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    dataset_valid = dataset_valid.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    # Return the datasets
    return dataset_train, dataset_valid


def main():
    # Example usage
    tfrecord_file = "ratsi_data.tfrecord"
    metadata_file = "ratsi_data.metadata.json"
    dataset_train, dataset_valid = create_data_loader(
        tfrecord_file,
        metadata_file,
        valid_size=0.5,
        batch_size=32,
        seq_size=4,
    )

    for i, (img, label) in enumerate(dataset_train.as_numpy_iterator()):
        print(i, img.shape, label.shape)

    for i, (img, label) in enumerate(dataset_valid.as_numpy_iterator()):
        print(i, img.shape, label.shape)


if __name__ == "__main__":
    main()
