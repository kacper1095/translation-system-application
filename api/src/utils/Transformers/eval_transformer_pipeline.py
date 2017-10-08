from src.utils.Logger import Logger


def score_transformer_pipeline(video, transformer_list):
    video_transform_steps = [video]
    scores = []
    for transformer in transformer_list:
        step_output = transformer.transform(video_transform_steps[-1])
        scores.append(transformer.score(video_transform_steps[-1]))
        video_transform_steps.append(step_output)
    return scores[-1]


def eval_transformer_pipeline(video, transformer_list):
    video_transform_steps = [video]
    for transformer in transformer_list:
        step_output = transformer.transform(video_transform_steps[-1])
        video_transform_steps.append(step_output)
    return video_transform_steps[-1]


def eval_transformer_pipeline_store_all(video, transformer_list):
    video_transform_steps = [video]
    outputs = {'init': video}
    for transformer in transformer_list:
        with Logger(transformer.__class__.__name__):
            step_output = transformer.transform(video_transform_steps[-1])
        outputs[transformer.out_key] = step_output
        video_transform_steps.append(step_output)
    return outputs
