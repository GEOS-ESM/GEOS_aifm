import numpy as np
import glob
import os


loc = '/discover/nobackup/awang17/SFNO/'
loc1 = loc + 'afno_backbone_finetune/2'
loc2 = loc + 'backward/afno_backbone_finetune/1'
dest = loc + 'interpolate'


files1 = glob.glob(loc1 + '/ground_truth*')
files2 = glob.glob(loc2 + '/ground_truth*')
files1.sort()
files2.sort()

for file1, file2 in zip(files1, files2):
    print(file1, file2)
    fname = file1.split('/')[-1]
    gt1 = np.load(file1)
    gt2 = np.load(file2)
    '''
    idx = -1
    for i in range(0, gt1.shape[0]):
        if (gt1[3] == gt2[i]).all():
            idx = i
            """
        if (gt1[i] != gt2[-i]).any():
            print(gt1.shape)
            print(gt2.shape)
            print(gt1[i,1,:10,:10])
            print(gt2[-i,1,:10,:10])
            raise Exception
            """
    if idx == -1:
        raise Exception
    print(idx)
    '''
    gt = np.concatenate((gt1, gt2[0:1]), axis=0)
    print(gt.shape)
    np.save(os.path.join(dest, fname), gt)

