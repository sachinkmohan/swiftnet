#https://newbedev.com/tensorflow-how-to-convert-meta-data-and-index-model-files-into-one-graph-pb-file
import tensorflow as tf

log='/home/mohan/git/swiftnet/swiftnet/log/swiftnet/'
#experiment='Experiment_4_2/'
meta_path =log+ 'model.ckpt-271.meta' # Your .meta file
output_node_names = ["class/logits_to_softmax"]    # Output nodes

with tf.Session() as sess:
    # Restore the graph
    saver = tf.train.import_meta_graph(meta_path)

    # Load weights
    saver.restore(sess,tf.train.latest_checkpoint(log))

    # Freeze the graph
    frozen_graph_def = tf.graph_util.convert_variables_to_constants(
        sess,
        sess.graph_def,
        output_node_names)

    # Save the frozen graph
    with open('output_graph_swiftnet.pb', 'wb') as f:
      f.write(frozen_graph_def.SerializeToString())