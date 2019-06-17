# Import TensorFlow and TensorRT
import tensorflow as tf
import tensorflow.contrib.tensorrt as trt
# Inference with TF-TRT frozen graph workflow:

import sys, os
from tqdm import tqdm
import numpy as np
graph_name = sys.argv[1]

graph = tf.Graph()
with graph.as_default():
    with tf.Session(graph=graph) as sess:
        # First deserialize your frozen graph:
        with tf.gfile.GFile(sys.argv[1], 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
        # Now you can create a TensorRT inference graph from your
        # frozen graph:
        #for n in graph_def.node:
        #    print(n.name)
        trt_graph = trt.create_inference_graph(
            input_graph_def=graph_def,
            outputs=[ 'network/upscore_8s/upscore8/upscore8/BiasAdd'],
            max_batch_size=1,
            max_workspace_size_bytes=2500000000,
            precision_mode='INT8')
        print('\n\nFinish tensorrt creation, now  import to tensorflow')
        # Import the TensorRT graph into a new graph and run:
        output_node = tf.import_graph_def(
            trt_graph,
            return_elements=[ 'network/upscore_8s/upscore8/upscore8/BiasAdd:0' ])
        print(output_node)
        img = np.ones((1, 480, 640, 3), dtype=np.uint8)
        #for n in graph_def.node:
        #    print(n.name)
        x = graph.get_tensor_by_name('import/network/input/Placeholder:0')
        for _ in tqdm(range(1000)):
            sess.run([output_node], feed_dict={x: img})
