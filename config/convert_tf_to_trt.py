# Import TensorFlow and TensorRT
import tensorflow as tf
import tensorflow.contrib.tensorrt as trt
# Inference with TF-TRT frozen graph workflow:

import sys, os

graph_name = sys.argv[1]

graph = tf.Graph()
with graph.as_default():
    with tf.Session() as sess:
        # First deserialize your frozen graph:
        with tf.gfile.GFile(sys.argv[1], 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
        # Now you can create a TensorRT inference graph from your
        # frozen graph:
        trt_graph = trt.create_inference_graph(
            input_graph_def=graph_def,
            outputs=['network/output/Argmax', 'network/upscore_8s/upscore8/upscore8/BiasAdd'],
            max_batch_size=1,
            max_workspace_size_bytes=2500000000,
            precision_mode='FP16')
        # Import the TensorRT graph into a new graph and run:
        output_node = tf.import_graph_def(
            trt_graph,
            return_elements=['network/output/Argmax', 'network/upscore_8s/upscore8/upscore8/BiasAdd' ])
        sess.run(output_node)
