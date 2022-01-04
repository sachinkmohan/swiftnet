import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import time
from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image
import random

gpu_options = tf.GPUOptions(visible_device_list='0')
config = tf.ConfigProto(log_device_placement=False, gpu_options=gpu_options)
# Change these values for the model used
# num_classes = 3  # Change this value to the number of classes of the model
image_height, image_width = (768, 768)  # Output display size as you want
num_classes = 12
epsilon = 1e-8
batch_size = 32
# Use images in test dir
dataset_dir = '/home/mohan/Documents/Thesis/cs_zero_bg/'

image_files = sorted(
    [os.path.join(dataset_dir, 'test/bonn', file) for file in
     os.listdir(dataset_dir + "test/bonn") if
     file.endswith('.png')])
annotation_files = sorted(
    [os.path.join(dataset_dir, "testannot/bonn_annot", file) for file in
     os.listdir(dataset_dir + "testannot/bonn_annot") if
     file.endswith('.png')])
# n=32
#
# image_files=image_files[0:n]
# annotation_files=annotation_files[0:n]




pred_time_list = []

or_ = np.zeros(num_classes, dtype=np.float32) + epsilon
and_ = np.zeros(num_classes, dtype=np.float32)
T_ = np.zeros(num_classes, dtype=np.float32) + epsilon
R_ = np.zeros(num_classes, dtype=np.float32) + epsilon


def decode(a, b):
    a = tf.io.read_file(a)
    a = tf.image.decode_png(a, channels=3)
    a = tf.cast(a, dtype=tf.float32)
    b = tf.io.read_file(b)
    b = tf.image.decode_png(b, channels=1)
    # random scale
    # scale = tf.random_uniform([1], minval=0.75, maxval=1.25, dtype=tf.float32)
    # hi = tf.floor(scale * 1024)
    # wi = tf.floor(scale * 2048)
    # hi=550
    # wi=688
    # s = tf.concat([hi, wi], 0)
    # s = tf.cast(s, dtype=tf.int32)
    # a = tf.image.resize_images(a, s, method=0, align_corners=True)
    # b = tf.image.resize_images(b, s, method=1, align_corners=True)
    b = tf.image.convert_image_dtype(b, dtype=tf.float32)
    # random crop and flip
    m = tf.concat([a, b], axis=-1)
    m = tf.image.random_crop(m, [image_height, image_width, 4])
    m = tf.image.random_flip_left_right(m)

    m = tf.split(m, num_or_size_splits=4, axis=-1)
    a = tf.concat([m[0], m[1], m[2]], axis=-1)
    img = tf.image.convert_image_dtype(a / 255, dtype=tf.uint8)
    a = a - [123.68, 116.779, 103.939]
    b = m[3]
    b = tf.image.convert_image_dtype(b, dtype=tf.uint8)
    a.set_shape(shape=(image_height, image_width, 3))
    b.set_shape(shape=(image_height, image_width, 1))
    img.set_shape(shape=(image_height, image_width, 3))
    return a, b,img

class_map = {0: [0, 0, 0],
             1: [108, 64, 20],
             2: [255, 229, 204],
             3: [0, 102, 0],
             4: [0, 128, 255],
             5: [64, 64, 64],
             6: [255, 128, 0],
             7: [153, 76, 0],
             8: [102, 102, 0],
             9: [255, 153, 204],
             10: [153, 204, 255],
             11: [101, 101, 11]}
# Set paths to the trained model

PATH_TO_CKPT = "./output_graph_swiftnet.pb"

# Set tensorflow graph
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')


def rgb_mask(mask):
    mask = mask.astype("uint8")
    print(mask.shape)
    h, w = mask.shape
    label = np.zeros((h, w, 3), dtype=np.uint8)
    # print(label.shape)
    for i, rgb in enumerate(class_map.values()):
        indices = np.where(mask == [i])
        label[:, :, 0][indices] = rgb[0]
        label[:, :, 1][indices] = rgb[1]
        label[:, :, 2][indices] = rgb[2]
    return label


def display(display_list):
    plt.figure(figsize=(15, 15))

    title = ['Input Image', 'True Mask', 'Predicted Mask']

    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i + 1)
        plt.title(title[i])
        if i == 0:
            mask = display_list[i]
        else:
            mask = rgb_mask(display_list[i])
        plt.imshow(mask)  # ,cmap='gray', vmin=0, vmax=255)
        plt.axis('off')
    plt.show()


def metrics(annotations, predictions):
    raw_gt_v = tf.reshape(tf.reshape(annotations, shape=[-1, 768, 768]), [-1, ])
    indices_v = tf.squeeze(tf.where(tf.greater_equal(raw_gt_v, 0)), 1)
    gt_v = tf.cast(tf.gather(raw_gt_v, indices_v), tf.int32)
    gt_one_v = tf.one_hot(gt_v, num_classes, axis=-1)
    raw_prediction_v = tf.argmax(tf.reshape(predictions, [-1, num_classes]), -1)
    prediction_v = tf.gather(raw_prediction_v, indices_v)
    prediction_ohe_v = tf.one_hot(prediction_v, num_classes, axis=-1)

    and_val = gt_one_v * prediction_ohe_v
    and_sum = tf.reduce_sum(and_val, [0])
    or_val = tf.to_int32((gt_one_v + prediction_ohe_v) > 0.)
    or_sum = tf.reduce_sum(or_val, axis=[0])
    T_sum = tf.reduce_sum(gt_one_v, axis=[0])
    R_sum = tf.reduce_sum(prediction_ohe_v, axis=[0])

    and_sum_val = and_sum.eval(session=tf.Session(config=config))
    or_sum_val = or_sum.eval(session=tf.Session(config=config))
    T_sum_val = T_sum.eval(session=tf.Session(config=config))
    R_sum_val = R_sum.eval(session=tf.Session(config=config))
    return and_sum_val, or_sum_val, T_sum_val, R_sum_val


# Inference pipeline
def run_inference(image, graph):
    with graph.as_default():

        with tf.Session(config=config) as sess:
            # Get handles to input and output tensors
            output_node_names = ["class/logits_to_softmax:0"]
            image_tensor = tf.get_default_graph().get_tensor_by_name('IteratorGetNext:0')

            # Run inference
            output = sess.run(output_node_names, feed_dict={image_tensor: image})
            # print(output)

            print('checked')

    return output



# Run the inference for each image
images = tf.convert_to_tensor(image_files)
annotations = tf.convert_to_tensor(annotation_files)

dataset = tf.data.Dataset.from_tensor_slices((images, annotations)).map(decode).shuffle(100).batch(batch_size)
iterator = dataset.make_one_shot_iterator()
data_set = iterator.get_next()
file = open("inference_10.txt", "w")
with tf.Session(config=config) as sess:
    try:
        while True:
            data = sess.run(data_set)
            img = data[0]
            mask = data[1]
            img_dis = data[2]
            print(img.shape)
            print(mask.shape)
            start = time.clock()
            output_seg = run_inference(img, detection_graph)
            end = time.clock()
            pred_time_list.append(end - start)
            print(f"time for image is {end - start}")
            output_seg = np.squeeze(output_seg)
            print(output_seg.shape)  # ).shape)
            pred = np.argmax(output_seg, -1)
            pred = np.uint8(pred)
            print(pred.shape)
            # for i in range(img.shape[0]):
            #     image=img_dis[i]
            #     label=np.squeeze(mask[i])
            #     prediction=pred[i]
            #     display([image,label,prediction])
            and_eval_batch, or_eval_batch, T_eval_batch, R_eval_batch = metrics(mask, output_seg)
            and_ = and_ + and_eval_batch
            or_ = or_ + or_eval_batch
            T_ = T_ + T_eval_batch
            R_ = R_ + R_eval_batch
            Recall_rate = and_ / T_
            Precision = and_ / R_
            IoU = and_ / or_
            mPrecision = np.mean(Precision)
            mRecall_rate = np.mean(Recall_rate)
            mIoU = np.mean(IoU)
            print(" batch Precision:")
            print(Precision)
            print(" batch Recall rate:")
            print(Recall_rate)
            print("batch IoU:")
            print(IoU)
            print("batch mPrecision:")
            print(mPrecision,2)
            print("batch mRecall_rate:")
            print(round(mRecall_rate,2))
            print("batch mIoU")
            print(round(mIoU,2))

    except tf.errors.OutOfRangeError:
        pass
        # , [550, 550,12]).numpy()




print(f"T_ {T_}")
print(f"R_ {R_}")
print(f"and_ {and_}")
print(f"or_ {or_}")
Recall_rate = and_ / T_
Precision = and_ / R_
IoU = and_ / or_
mPrecision = np.mean(Precision)
mRecall_rate = np.mean(Recall_rate)
mIoU = np.mean(IoU)
print("Precision:")
print(round(Precision,3))
print("Recall rate:")
print(round(Recall_rate,3))
print("IoU:")
print(round(IoU,3))
print("mPrecision:")
print(mPrecision)
print("mRecall_rate:")
print(mRecall_rate)
print("mIoU")
print(mIoU)
# # plt.axis('off')
# plt.show()
# PIL_image = Image.fromarray(np.uint8(pred)).convert('RGB')
print(pred_time_list)
file.write(f" pred_time_list{round(pred_time_list,3)}\n")
file.write(f" miou{round(mIoU,3)}\n")
file.write(f" mRecall_rate{round(mRecall_rate,3)}\n")
file.write(f" mPrecision{round(mPrecision,3)}\n")
file.write(f" Precision{round(Precision,3)}\n")
file.write(f" Recall_rate{round(Recall_rate,3)}\n")
file.write(f" IoU{round(IoU,3)}\n")
file.close()