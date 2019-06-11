~/thirdparty/tensorflow/bazel-bin/tensorflow/tools/graph_transforms/transform_graph \
--in_graph='mobilenet_const_nclt_2019_06_11.pb' \
--out_graph='optimized_mobilenet_const_nclt_2019_06_11.pb' \
--inputs='network/input/Placeholder' \
--outputs='network/output/ArgMax,network/upscore_8s/upscore8/upscore8/BiasAdd' \
--transforms='
fold_old_batch_norms
strip_unused_nodes
'
