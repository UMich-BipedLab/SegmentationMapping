~/thirdparty/tensorflow/bazel-bin/tensorflow/tools/graph_transforms/transform_graph \
--in_graph='mobilenet_nclt_2018_06_10.pb' \
--out_graph='optimized_mobilenet_nclt_2019_06_10.pb' \
--inputs='network/input/Placeholder' \
--outputs='network/output/ArgMax,network/upscore_8s/upscore8/upscore8/BiasAdd' \
--transforms='
  strip_unused_nodes
  fold_constants(ignore_errors=true)
  fold_batch_norms
  fold_old_batch_norms
  quantize_weights'
