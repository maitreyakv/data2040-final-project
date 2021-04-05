"""
Script to process raw video data into TFRecord data files
"""

__author__ = "Maitreya Venkataswamy"

import skvideo.io
from tqdm import tqdm
import tensorflow as tf
import numpy as np
import pandas as pd
import json


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value):
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def frame_example(frame_bytes, label):
    feature = {
        "frame": _bytes_feature(frame_bytes),
        "label": _int64_feature(label)
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))


def process_and_write_frame(frame, label, tfrecord_writer, crop_frac, img_size):
    frame = tf.image.central_crop(frame, crop_frac)
    frame = tf.image.resize(frame, img_size)
    frame = tf.cast(frame, tf.uint8)
    tfrecord_writer.img_size = frame[..., :1].shape

    frame_bytes = tf.io.encode_jpeg(image=frame[..., :1], format="grayscale")
    example = frame_example(frame_bytes, label)
    tfrecord_writer.write(example.SerializeToString())
    tfrecord_writer.n_frames += 1


def process_video(video_file, label_file, tfrecord_writer, crop_frac, img_size, classes_map,
                  break_label="Uncertain"):
    labels = pd.read_csv(label_file, sep=";", index_col=0).action.map(classes_map)
    assert all(labels.notna()), "there is a label in the dataset not in classes_map"
    video_gen = skvideo.io.vreader(video_file)
    for frame, label in tqdm(zip(video_gen, labels), total=len(labels)):
        process_and_write_frame(frame, label, tfrecord_writer, crop_frac, img_size)

    frame = 255 * np.ones_like(frame)
    process_and_write_frame(frame, classes_map[break_label], tfrecord_writer, crop_frac, img_size)
    tfrecord_writer.img_size = frame[..., :1].shape


def main():
    tfrecord_file = "ratsi_data.tfrecord"

    with open("classes_map.json", "r") as f:
        classes_map = json.load(f)

    with tf.io.TFRecordWriter(tfrecord_file) as tfrecord_writer:
        tfrecord_writer.n_frames = 0
        for i in range(1, 10):
            video_file = "../../RatSI_v1.01/videos/Observation{:02d}.mp4".format(i)
            label_file = "../../RatSI_v1.01/annotations/Observation{:02d}.csv".format(i)
            process_video(video_file, label_file, tfrecord_writer, 0.8, (128, 128), classes_map)

        with open("ratsi_data.metadata.json", "w") as f:
            json.dump({
                "n_records": tfrecord_writer.n_frames,
                "img_size": tfrecord_writer.img_size
            }, f)


if __name__ == "__main__":
    main()
