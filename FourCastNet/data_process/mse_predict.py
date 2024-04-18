import numpy as np
import os
import glob

def resize_time(x, ones=False):
    if ones:
        result = np.ones((25,)+x.shape[1:])
    else:
        result = np.zeros((25,)+x.shape[1:])
    result[:24] = x
    return result

def compute_mse(location): 
    mse_files = glob.glob(location + '/mse/mse*.npy')

    total = []
    for file in mse_files:
        total.append(np.load(file))
    result = np.mean(np.stack(total, axis=0), axis=0)
    expanded_result = resize_time(result, ones=True)
    np.save(location + '/mse/final_mse.npy', expanded_result)


def inference(loc1, loc2, dest):
    a = np.load(loc1 + '/mse/final_mse.npy')
    b = np.load(loc2 + '/mse/final_mse.npy')
    b = b[::-1] # reverse time steps for backwards
    k = a / (a + b)
    if np.isnan(k).any():
        raise Exception("bad k")
    if len(k.shape) == 2:
        k = np.expand_dims(k, (2, 3))

    forward_files = glob.glob(loc1 + '/prediction*.npy')
    forward_files.sort()
    backward_files = glob.glob(loc2 + '/prediction*.npy')
    backward_files.sort()
    for forward, backward in zip(forward_files, backward_files):
        print(forward, backward)
        x_a = resize_time(np.load(forward))
        x_b = resize_time(np.load(backward))[::-1]
        prediction = x_a + k * (x_b - x_a)
        file_name = forward.split('/')[-1]
        np.save(os.path.join(dest, file_name), prediction)


loc = '/discover/nobackup/awang17/SFNO/'
loc1 = loc + 'afno_backbone_finetune/2'
loc2 = loc + 'backward/afno_backbone_finetune/1'
dest = loc + 'interpolate'

compute_mse(loc1)
print('done loc1')
compute_mse(loc2)
print('done loc2')
inference(loc1, loc2, dest)

