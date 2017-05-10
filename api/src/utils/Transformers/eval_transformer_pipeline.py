from src.common import SHOW_DEBUG_IMAGES, SHOW_DEBUG_PRINT

import matplotlib.pyplot as plt


def score_transformer_pipeline(video, transformer_list):
    video_transform_steps = [video]
    scores = []
    for transformer in transformer_list:
        step_output = transformer.transform(video_transform_steps[-1])
        scores.append(transformer.score(video_transform_steps[-1]))
        if SHOW_DEBUG_IMAGES:
            print(transformer.get_name())
            plt.imshow(step_output)
            plt.axis('off')
            plt.show()
        video_transform_steps.append(step_output)
    return scores[-1]


def eval_transformer_pipeline(video, transformer_list):
    video_transform_steps = [video]
    for transformer in transformer_list:
        step_output = transformer.transform(video_transform_steps[-1])
        if SHOW_DEBUG_IMAGES:
            print(transformer.get_name())
            plt.imshow(step_output)
            plt.axis('off')
            plt.show()
        video_transform_steps.append(step_output)
    return video_transform_steps[-1]


def eval_transformer_pipeline_store_all(video, transformer_list):
    video_transform_steps = [video]
    for transformer in transformer_list:
        step_output = transformer.transform(video_transform_steps[-1])
        if SHOW_DEBUG_IMAGES:
            print(transformer.get_name())
            plt.imshow(step_output)
            plt.axis('off')
            plt.show()
        video_transform_steps.append(step_output)
    return video_transform_steps
