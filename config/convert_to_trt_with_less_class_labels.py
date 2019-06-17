import tensorflow as tf
import numpy as np
import time
import argparse
from tqdm import tqdm
import tensorflow.contrib.tensorrt as trt
import pdb
parser = argparse.ArgumentParser(description="Inference test")
parser.add_argument('--graph', type=str)
parser.add_argument('--new_num_class', type=int, default=14)
parser.add_argument('--precision', type=str, default='FP32')
parser.add_argument('--trt_optimized_graph', type=str)
# Parse the arguments
args = parser.parse_args()

new_num_class = int(args.new_num_class)

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
    y, = tf.import_graph_def(graph_def, return_elements=['network/output/ArgMax:0'], name='')
    #y, = tf.import_graph_def(graph_def, return_elements=['SemanticPredictions:0'])
    #print('Operations in Graph:')
    #print([op.name for op in G.get_operations()])
    x = G.get_tensor_by_name('network/input/Placeholder:0')
    #b = G.get_tensor_by_name('import/network/input/Placeholder_2:0')
    d = G.get_tensor_by_name("network/upscore_8s/upscore8/upscore8/BiasAdd:0")

    distribution_sliced = d[:, :, :, :new_num_class]
    class_prob_tensor = tf.nn.softmax(distribution_sliced, name='network/output/ClassDistribution')
    label_tensor = tf.argmax(class_prob_tensor, axis=-1, name='network/output/ClassIndexPrediction', output_type=tf.int32)
    
    print("tensorrt graph conversion...")
    trt_graph = trt.create_inference_graph(
        input_graph_def=tf.get_default_graph().as_graph_def(),
        outputs=[ 'network/output/ClassDistribution', 'network/output/ClassIndexPrediction' ],
        max_batch_size=1,
        max_workspace_size_bytes=2500000000,
        precision_mode=args.precision)
    print('\n\nFinish tensorrt creation, now  import to tensorflow')
    # Import the TensorRT graph into a new graph and run:
    #print(distribution_sliced.shape)
    #x = G.get_tensor_by_name('import/ImageTensor:0')
    #d = G.get_tensor_by_name('import/ResizeBilinear_2:0')
    #tf.global_variables_initializer().run()
g_new = tf.Graph()
with tf.Session(graph=g_new,config=config) as sess2:
    output_node = tf.import_graph_def(
    trt_graph,
    return_elements=[ 'network/output/ClassDistribution', 'network/output/ClassIndexPrediction' ], name='')


    x = g_new.get_tensor_by_name('network/input/Placeholder:0')
    class_prob_tensor = g_new.get_tensor_by_name('network/output/ClassDistribution:0')
    label_tensor = g_new.get_tensor_by_name('network/output/ClassIndexPrediction:0')
    print('class_prob_tensor.shape is ',class_prob_tensor.shape)

    img = np.ones((1, 640, 480, 3), dtype=np.uint8)
    
    # Experiment should be repeated in order to get an accurate value for the inference time and FPS.
    for _ in tqdm(range(100)):
        start = time.time()
        #out = sess.run(y, feed_dict={x: img, b:False})
        out = sess2.run([label_tensor,class_prob_tensor], feed_dict={x: img})

    tf.io.write_graph(g_new, './', args.trt_optimized_graph, as_text=False)
