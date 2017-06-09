import numpy as np
import math


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
