from collections import namedtuple
NeuralNetConfigs = namedtuple("NeuralNetConfigs", "path \
                                                   num_classes \
                                                   image_input_tensor \
                                                   is_train_input_tensor \
                                                   input_width \
                                                   input_height \
                                                   label_output_tensor \
                                                   distribution_output_tensor \
                                                   ")
