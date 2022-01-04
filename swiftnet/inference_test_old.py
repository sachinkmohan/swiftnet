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

gpu_options = tf.GPUOptions(visible_device_list='0',per_process_gpu_memory_fraction=0.1)
config = tf.ConfigProto(log_device_placement=False, gpu_options=gpu_options)
# Change these values for the model used
#num_classes = 3  # Change this value to the number of classes of the model
IMAGE_SIZE = (768, 768)  # Output display size as you want
num_classes=19
epsilon = 1e-8
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

'''
class_map={0: [0, 0, 0],
1: [108, 64, 20],
2 :[255, 229, 204],
3 :[0, 102, 0],
4 :[0, 128, 255],
5 :[64, 64, 64],
6 :[255, 128, 0],
7 :[153, 76, 0],
8 :[102, 102, 0],
9 :[255, 153, 204],
10 :[153, 204, 255],
11 :[101, 101, 11]}
# Set paths to the trained model

'''

PATH_TO_CKPT = "./output_graph_swiftnet.pb"

# Set tensorflow graph
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')


# Convert input image to a numpy array
def load_image_to_numpy(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)


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
def rgb_mask(mask):
    mask = mask.astype("uint8")
    print(mask.shape)
    h,w=mask.shape
    label = np.zeros((h, w, 3), dtype=np.uint8)
    #print(label.shape)
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
    plt.subplot(1, len(display_list), i+1)
    plt.title(title[i])
    if i==0:
        mask=display_list[i]
    else:
        mask=rgb_mask(display_list[i])
    plt.imshow(mask)#,cmap='gray', vmin=0, vmax=255)
    plt.axis('off')
  plt.show()

def img_prep(a, b):
    image_ = Image.open(a)
    mask_ = Image.open(b)
    image_= image_.resize((768, 768), Image.ANTIALIAS)
    mask_=mask_.resize((768, 768), Image.ANTIALIAS)
    image_np = np.float32(np.asarray(image_))
    mask_np= np.uint8(np.asarray(mask_))
    # Conver the image to numpy array


    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
    image_np_expanded = np.expand_dims(image_np, axis=0)
    return image_np_expanded,mask_np
#
# def metrics(mask, pred):
#     annotations=mask
#     predictions= pred
#
#     raw_gt_v = np.reshape(np.reshape(annotations, (-1, 550, 550)), (-1, ))
#     #indices_v = np.squeeze(np.where(.greater_equal(raw_gt_v, 0)), 1)
#     #gt_v = tf.cast(tf.gather(raw_gt_v, indices_v), tf.int32)
#     gt_one_v=np.eye(num_classes)[raw_gt_v]
#     #gt_one_v = tf.one_hot(raw_gt_v, num_classes, axis=-1)
#     raw_prediction_v = np.argmax(np.reshape(predictions, [-1, num_classes]), -1)
#     #prediction_v = tf.gather(raw_prediction_v, indices_v)
#     prediction_ohe_v = np.eye(num_classes)[raw_prediction_v]
#     #prediction_ohe_v = tf.one_hot(raw_prediction_v, num_classes, axis=-1)
#
#     and_val = gt_one_v * prediction_ohe_v
#     and_sum = tf.reduce_sum(and_val, [0])
#     or_val = tf.to_int32((gt_one_v + prediction_ohe_v) > 0.)
#     or_sum = tf.reduce_sum(or_val, axis=[0])
#     T_sum = tf.reduce_sum(gt_one_v, axis=[0])
#     R_sum = tf.reduce_sum(prediction_ohe_v, axis=[0])
#     and_sum_val = and_sum.eval(session=tf.Session(config=config))
#     or_sum_val = or_sum.eval(session=tf.Session(config=config))
#     T_sum_val = T_sum.eval(session=tf.Session(config=config))
#     R_sum_val = R_sum.eval(session=tf.Session(config=config))
#     #print(f"and_sum {and_sum_val}")
 #   return  and_sum_val, or_sum_val, T_sum_val, R_sum_val

def metrics(mask, pred):
    annotations=tf.convert_to_tensor(mask,np.float32)
    predictions= tf.convert_to_tensor(pred, np.float32)
    #print(annotations)
    #print(predictions)
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
    return  and_sum_val, or_sum_val, T_sum_val, R_sum_val

pred_time_list=[]
num_class=19
or_ = np.zeros((num_class), dtype=np.float32) + epsilon
and_ = np.zeros((num_class), dtype=np.float32)
T_ = np.zeros((num_class), dtype=np.float32) + epsilon
R_ = np.zeros((num_class), dtype=np.float32) + epsilon

# Run the inference for each image
for i in range(len(image_files)):
    print("image", i)
    image,mask=img_prep(image_files[i],annotation_files[i])
    image_np = np.subtract(image, [123.68, 116.779, 103.939])
    print(f"image dimensions {image.shape}")
    # Perform the interence
    start = time.clock()
    output_seg = run_inference(image_np, detection_graph)
    end = time.clock()
    pred_time_list.append(end - start)
    print(f"time for image {i} is {end - start}")
    output_seg = np.squeeze(output_seg)  # , [550, 550,12]).numpy()
    # output_seg=np.array(output_seg)
    # plt.figure(figsize=IMAGE_SIZE)
    # plt.imshow(output_seg)
    # print(f"output_seg {len(output_seg)}")
    image = np.squeeze(image)
    print(output_seg.shape)  # ).shape)
    pred = np.argmax(output_seg, -1)
    pred = np.uint8(pred)


    #display([image/255.0,mask,pred])
    # plt.imshow(pred, interpolation='nearest')
    # plt.show(annotation_files[i])
    and_eval_batch, or_eval_batch, T_eval_batch, R_eval_batch = metrics(mask,output_seg)
    and_ = and_ + and_eval_batch
    or_ = or_ + or_eval_batch
    # print()
    #print(f"T_eval_batch *************************** {T_eval_batch} T_  {T_}")
    T_ = T_ + T_eval_batch
    R_ = R_ + R_eval_batch

    # print(f"shape of T_ and R_", {np.shape(T_)}, {np.shape(R_)})
    #
    # print(f"T_eval_batch ======================={T_eval_batch}, T_  {T_}")
    # print(f"and_eval_batch {and_eval_batch}")
    # print(f"or_eval_batch {or_eval_batch}")
    # print(f"R_eval_batch {R_eval_batch}")
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
print(Precision)
print("Recall rate:")
print(Recall_rate)
print("IoU:")
print(IoU)
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