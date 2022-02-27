import numpy as np
def convert_to_multiple(tensor):
    if len(tensor.shape) == 4:
        return [convert_to_multiple(i) for i in tensor]
    else:
        images = (tensor[:, :, i:i+3] for i in range(0, tensor.shape[-1], 3))
        images = np.vstack(images)
        return images

def multiple_to_single(tensor):
    if len(tensor.shape) == 4:
        return [multiple_to_single(i) for i in tensor]
    else:
        section_size = int(tensor.shape[0] / 9)
        images = [tensor[i:i+section_size, :, :] for i in range(0, tensor.shape[0], section_size)]
        images = np.concatenate(images, axis=-1)
        return images