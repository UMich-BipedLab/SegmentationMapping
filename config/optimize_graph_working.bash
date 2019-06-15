~/thirdparty/tensorflow/bazel-bin/tensorflow/tools/graph_transforms/transform_graph \
--in_graph='mobilenet_nclt/optimized_mobilenet_frozen_nclt_bn_480.pb' \
--out_graph='mobilenet_nclt/optimized_mobilenet_nclt_bn_2019_06_14.pb' \
--inputs='network/input/Placeholder' \
--outputs='network/output/ArgMax,network/upscore_8s/upscore8/upscore8/BiasAdd' \
--transforms='
fold_constants(ignore_errors=true)
fold_old_batch_norms
strip_unused_nodes
'
