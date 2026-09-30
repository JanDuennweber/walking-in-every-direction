import numpy as np
from numpy.typing import NDArray
import torch
from sklearn.cluster import MeanShift

def apply_mean_shift_to_model_output(model_output, bandwidth=2.0) -> NDArray:
    if type(model_output) is torch.Tensor:
        model_output = model_output.detach().numpy()
    # model_output: bs x channels x height x width
    bs, channels, height, width = model_output.shape
    clustered_output = np.zeros((bs, height, width), dtype=int)
    for i in range(bs):
        embeddings = model_output[i].reshape(channels, -1).T
        #print("embeddings: ", embeddings.shape)
        #print(embeddings)
        ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
        ms.fit(embeddings)
        labels = ms.labels_

        clustered_output[i] = labels.reshape(height, width)

    return clustered_output

def apply_mean_shift_to_model_output_without_background_pixels(model_output,input_seg_masks, bandwidth=2.0) -> NDArray:
    if type(model_output) is torch.Tensor:
        model_output = model_output.detach().numpy()
    if type(input_seg_masks) is torch.Tensor:
        input_seg_masks = input_seg_masks.detach().numpy()
    # model_output: bs x channels x height x width
    bs, channels, height, width = model_output.shape
    clustered_output = np.zeros((bs, height, width), dtype=int)
    for i in range(bs):
        # check if input_seg_masks has any 1s -> if not then skip
        bool_seg_mask = input_seg_masks[i] > 0
        if not np.any(bool_seg_mask):
            clustered_output[i] = np.zeros((height,width),dtype=int)
            continue

        model_output_with_backgroun_pixel_zero = model_output[i] * input_seg_masks[i]
        embeddings = model_output_with_backgroun_pixel_zero.reshape(channels, -1).T
        mask = np.any(embeddings != 0, axis=1)
        embeddings_without_bg = embeddings[mask]

        ms = MeanShift(bandwidth=bandwidth)
        ms.fit(embeddings_without_bg)
        # labels length: embeddings_without_bg.length
        labels = ms.labels_

        result_projected = np.zeros((height,width), dtype=int)
        curr_idx_in_labels = 0
        for h in range(height):
            for w in range(width):
                if input_seg_masks[i,0,h,w] == 1:
                    result_projected[h,w] = labels[curr_idx_in_labels]
                    curr_idx_in_labels += 1


        clustered_output[i] = result_projected 

    return clustered_output

