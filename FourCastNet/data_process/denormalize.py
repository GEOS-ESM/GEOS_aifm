import xarray as xr
from datetime import datetime
from calendar import monthrange
import multiprocessing as mp
import numpy as np
import os
import glob

variables = ['H250', 'H500', 'H850', 'PS', 'Q500', 'Q850', 'SLP', 'T2M', 'T500', 'T850', 'TQV', 'U10M', 'U250', 'U500', 'U850', 'V10M', 'V250', 'V500', 'V850']

means = []
variances = []
suffixes = [12] + [i+1 for i in range(11)]
for month in suffixes:
    old_variables = [
        'PS', 'SLP', 'U10M', 'V10M', 'T2M',
        'H850',
        'T500', 'U500', 'V500', 'H500', 'Q500',
        'U250', 'V250', 'H250',
        'TQV'
    ]
    old = xr.open_dataset(f'/discover/nobackup/awang17/stats/old/means{month:02d}.nc4')
    m = xr.open_dataset(f'/discover/nobackup/awang17/stats/means{month:02d}.nc4')
    m.close()
    means.append(xr.merge([m, old[old_variables]]))
    v = xr.open_dataset(f'/discover/nobackup/awang17/stats/vars{month:02d}.nc4')
    v.close()
    old_vars = old[['Var_'+name for name in old_variables]]
    old_vars = old_vars.rename({'Var_'+name: name for name in old_variables})
    variances.append(xr.merge([v, old_vars]))

def convert(stat):
    result = np.zeros((1, 19, 360, 576))
    for i, var in enumerate(variables):
        result[0,i,:,:] = stat[var].data[:-1,:]
    return result

means = [convert(m) for m in means]
variances = [convert(v) for v in variances]

means.append(means[0])
variances.append(variances[0])
means.append(means[1])
variances.append(variances[1])
print(len(variances))
print(len(means))

def zero_means(data, day, month, year):
    day -= 15
    if day < 0:
        total = monthrange(year, (month + 10) % 12 + 1)[1]
        day += total
        weight = day / total
        m = means[month-1] * (1-weight) + means[month] * weight
        std = np.sqrt(variances[month-1] * (1-weight) + variances[month] * weight)
    elif day == 0:
        m = means[month]
        std = np.sqrt(variances[month])
    else:
        total = monthrange(year, month)[1]
        weight = day / total
        m = means[month] * (1-weight) + means[month+1] * weight
        std = np.sqrt(variances[month] * (1-weight) + variances[month+1] * weight)
    
    return (data * std)

def denormalize(data, day, month, year):
    day -= 15
    if day < 0:
        total = monthrange(year, (month + 10) % 12 + 1)[1]
        day += total
        weight = day / total
        m = means[month-1] * (1-weight) + means[month] * weight
        std = np.sqrt(variances[month-1] * (1-weight) + variances[month] * weight)
    elif day == 0:
        m = means[month]
        std = np.sqrt(variances[month])
    else:
        total = monthrange(year, month)[1]
        weight = day / total
        m = means[month] * (1-weight) + means[month+1] * weight
        std = np.sqrt(variances[month] * (1-weight) + variances[month+1] * weight)
    
    return (data * std) + m


def denorm_main():
    location = '/discover/nobackup/awang17/SFNO/interpolate'
    files = glob.glob(location + '/prediction*.npy')
    files.extend(glob.glob(location + '/ground_truth*.npy'))
    files.sort()
    for file in files:
        print(file)
        tokens = [t for t in file.split('/')]
        dest = '/'.join(tokens[:-1]) + '/denorm_' + tokens[-1]
        file_name = tokens[-1][:-4]
        day = int(file_name[-2:])
        month = int(file_name[-4:-2])
        values = np.load(file)
        result = denormalize(values, day, month, 2020)
        np.save(dest, result)


def anomaly_corr():
    location = '/discover/nobackup/awang17/SFNO/interpolate'
    pred_files = glob.glob(location + '/prediction*.npy')
    pred_files.sort()
    tar_files = glob.glob(location + '/ground_truth*.npy')
    tar_files.sort()

    year_results = [[] for _ in range(4)]
    names = ['/anomaly_full.npy', '/anomaly_north.npy', '/anomaly_equator.npy', '/anomaly_south.npy']
    slices = [slice(None), slice(220, 360), slice(140, 220), slice(0, 140)]
    slices = [(slice(None), s, slice(None)) for s in slices]
    
    for pred_file, tar_file in zip(pred_files, tar_files):
        print(pred_file, tar_file)
        tokens = [t for t in pred_file.split('/')]
        dest = '/'.join(tokens[:-1]) + '/anomaly_' + tokens[-1]
        file_name = tokens[-1][:-4]
        day = int(file_name[-2:])
        month = int(file_name[-4:-2])
        pred = zero_means(np.load(pred_file), day, month, 2020) # t, 19, 360, 576
        tar = zero_means(np.load(tar_file), day, month, 2020)
        for i, s in enumerate(slices):
            result = np.zeros((pred.shape[0], 19))
            for t in range(pred.shape[0]):
                p_slice = pred[(t,) + s]
                t_slice = tar[(t,) + s]
                p_var = p_slice - np.mean(p_slice, axis=(-2,-1), keepdims=True)
                t_var = t_slice - np.mean(t_slice, axis=(-2,-1), keepdims=True)
                result[t] = np.mean(p_var * t_var, axis=(-2,-1)) / np.sqrt( np.mean(p_var * p_var, axis=(-2,-1)) * np.mean(t_var * t_var, axis=(-2,-1)))
            #np.save(dest, result)
            year_results[i].append(result)
    for i, name in enumerate(names):
        result = np.mean(np.stack(year_results[i], axis=0), axis=0)
        np.save(location + name, result)

denorm_main()
anomaly_corr()


