import tensorflow as tf
import numpy as np
import time
import argparse
from tqdm import tqdm
import pdb
parser = argparse.ArgumentParser(description="Inference test")
parser.add_argument('--graph', type=str)
parser.add_argument('--iterations', default=1000, type=int)

# Parse the arguments
args = parser.parse_args()

if args.graph is not None:
    with tf.gfile.GFile(args.graph, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
else:
    raise ValueError("--graph should point to the input graph file.")

G = tf.Graph()
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

with tf.Session(graph=G,config=config) as sess:
    # The inference is done till the argmax layer on the logits, as the softmax layer is not important.
    #y, = tf.import_graph_def(graph_def, return_elements=['network/output/ArgMax:0'])
    y, = tf.import_graph_def(graph_def, return_elements=['network/output/ClassIndexPrediction:0'], name='')
    #y, = tf.import_graph_def(graph_def, return_elements=['SemanticPredictions:0'])
    #print('Operations in Graph:')
    #print([op.name for op in G.get_operations()])
    x = G.get_tensor_by_name('network/input/Placeholder:0')
    #b = G.get_tensor_by_name('import/network/input/Placeholder_2:0')
    #d = G.get_tensor_by_name("import/network/upscore_8s/upscore8/upscore8/BiasAdd:0")
    d = G.get_tensor_by_name("network/output/ClassDistribution:0")
    print(d.shape)
    #x = G.get_tensor_by_name('import/ImageTensor:0')
    #d = G.get_tensor_by_name('import/ResizeBilinear_2:0')
    #tf.global_variables_initializer().run()

    img = np.ones((1, 640, 480, 3), dtype=np.uint8)

    # Experiment should be repeated in order to get an accurate value for the inference time and FPS.
    for _ in tqdm(range(args.iterations)):
        start = time.time()
        #out = sess.run(y, feed_dict={x: img, b:False})
        out = sess.run([y,d], feed_dict={x: img})


