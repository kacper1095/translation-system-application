import numpy as np
import math
import tensorflow as tf
import keras.backend as K


def sliding_window(image, step_size, window_size):
    nb_rows = (image.shape[0] - window_size[0])/step_size[0] + 1
    nb_cols = (image.shape[1] - window_size[1])/step_size[1] + 1
    result = np.ndarray((int(math.ceil(nb_cols * nb_rows)) + 1, window_size[0], window_size[1], image.shape[-1]))
    index = 0
    for y in range(0, image.shape[0], step_size[0]):
        for x in range(0, image.shape[1], step_size[1]):
            result[index] = image[y:y + window_size[0], x: x + window_size[1]]
            index += 1
    return result


def reconstruct_sliding_window(batches, original_step_size, original_shape):
    result = np.ndarray((original_shape[0], original_shape[1], 1))
    index = 0
    try:
        for y in range(0, original_shape[0], original_step_size[0]):
            for x in range(0, original_shape[1], original_step_size[1]):
                result[y: y + original_step_size[0], x: x + original_step_size[1], :] = batches[index]
                index += 1
    except IndexError as e:
        pass
    return result


class TensorflowWrapper(object):

    def __init__(self, path_to_ckpt, input_tensor_name, output_tensor_names):
        self.ckpt_path = path_to_ckpt
        self.detection_graph = tf.Graph()

        self.load_model()
        self.image_tensor = self.detection_graph.get_tensor_by_name(input_tensor_name)
        self.output_tensors = []
        for output_tensor_name in output_tensor_names.split(','):
            self.output_tensors.append(self.detection_graph.get_tensor_by_name(output_tensor_name))
        self.session = tf.Session(graph=self.detection_graph)

    def load_model(self):
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(self.ckpt_path, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

    def get_feed_dict(self, input_images):
        return {self.image_tensor: input_images}

    def predict(self, image):
        if len(image.shape) == 3:
            image = np.expand_dims(image, axis=0)
        if K.image_dim_ordering() == 'th':
            image = image.transpose((0, 2, 3, 1))
        output = self.session.run(self.output_tensors, self.get_feed_dict(image))
        return output

    def __del__(self):
        self.session.close()


