#BSD 3-Clause License
#
#Copyright (c) 2022, FourCastNet authors
#All rights reserved.
#
#Redistribution and use in source and binary forms, with or without
#modification, are permitted provided that the following conditions are met:
#
#1. Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
#2. Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
#3. Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
#THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
#AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
#IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
#DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
#FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
#DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
#SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
#CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
#OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
#OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
#The code was authored by the following people:
#
#Jaideep Pathak - NVIDIA Corporation
#Shashank Subramanian - NERSC, Lawrence Berkeley National Laboratory
#Peter Harrington - NERSC, Lawrence Berkeley National Laboratory
#Sanjeev Raja - NERSC, Lawrence Berkeley National Laboratory 
#Ashesh Chattopadhyay - Rice University 
#Morteza Mardani - NVIDIA Corporation 
#Thorsten Kurth - NVIDIA Corporation 
#David Hall - NVIDIA Corporation 
#Zongyi Li - California Institute of Technology, NVIDIA Corporation 
#Kamyar Azizzadenesheli - Purdue University 
#Pedram Hassanzadeh - Rice University 
#Karthik Kashinath - NVIDIA Corporation 
#Animashree Anandkumar - California Institute of Technology, NVIDIA Corporation

import logging
import glob
import torch
import random
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torch import Tensor
#import h5py
import netCDF4
import math
#import cv2
from utils.img_utils import reshape_fields, reshape_precip
from collections import OrderedDict
import gc

class SimpleLRU():
  def __init__(self, size):
    self.size = size
    self.values = OrderedDict()

  def __contains__(self, item):
    return item in self.values

  def __getitem__(self, item):
    self.values.move_to_end(item)
    return self.values[item]

  def __setitem__(self, item, value):
    if item in self.values:
      self.values.move_to_end(item)
    self.values[item] = value
    
    if len(self.values) > self.size:
      val = self.values.popitem(last=False)
      del val
      gc.collect()



def get_data_loader(params, files_pattern, distributed, train):

  dataset = GetDataset(params, files_pattern, train)
  sampler = DistributedSampler(dataset, shuffle=train) if distributed else None
  
  dataloader = DataLoader(dataset,
                          batch_size=int(params.batch_size),
                          num_workers=params.num_data_workers,
                          shuffle=False, #(sampler is None),
                          sampler=sampler if train else None,
                          drop_last=True,
                          pin_memory=torch.cuda.is_available())

  if train:
    return dataloader, dataset, sampler
  else:
    return dataloader, dataset

class GetDataset(Dataset):
  def __init__(self, params, location, train):
    self.params = params
    self.location = location
    self.train = train
    self.dt = params.dt
    self.n_history = params.n_history
    self.in_channels = np.array(params.in_channels)
    self.out_channels = np.array(params.out_channels)
    self.n_in_channels = len(self.in_channels)
    self.n_out_channels = len(self.out_channels)
    self.crop_size_x = params.crop_size_x
    self.crop_size_y = params.crop_size_y
    self.roll = params.roll
    self._get_files_stats()
    self.two_step_training = params.two_step_training
    self.orography = params.orography
    self.precip = True if "precip" in params else False
    self.add_noise = params.add_noise if train else False # TODO want to remove noise?

    if self.precip: #TODO change paths if using precip in the future?
        path = params.precip+'/train' if train else params.precip+'/test'
        self.precip_paths = glob.glob(path + "/*.h5")
        self.precip_paths.sort()

    try:
        self.normalize = params.normalize
    except:
        self.normalize = True #by default turn on normalization if not specified in config

    if self.orography:
      self.orography_path = params.orography_path

  def _get_files_stats(self):
    print(self.location + "/*.nc4");
    self.files_paths = glob.glob(self.location + "/*.nc4")
    self.files_paths.sort()
    print(self.files_paths[0], self.files_paths[-1])
    """
    self.n_years = len(self.files_paths)
    self.n_samples_per_year = []
    for i in range(len(self.files_paths)):
      with netCDF4.Dataset(self.files_paths[0], 'r') as _f:
          logging.info("Getting file stats from {}".format(self.files_paths[0]))
          self.n_samples_per_year.append(_f['fields'].shape[0])
          #original image shape (before padding)
          self.img_shape_x = _f['fields'].shape[2] - 1#just get rid of one of the pixels; necessary? yes, hack for getting correct dimensions of output from patches
          self.img_shape_y = _f['fields'].shape[3]
    """
    self.n_samples_total = len(self.files_paths) * 24
    self.img_shape_x = 360
    self.img_shape_y = 576
    self.files = {}#Simple_LRU(500)
    #self.precip_files = [None for _ in range(len(self.files_paths))]
    logging.info("Found data at path {}. Number of examples: {}. Image Shape: {} x {} x {}".format(self.location, self.n_samples_total, self.img_shape_x, self.img_shape_y, self.n_in_channels))
    logging.info("Delta t: {} hours".format(6*self.dt))
    logging.info("Including {} hours of past history in training at a frequency of {} hours".format(6*self.dt*self.n_history, 6*self.dt))


  def _open_file(self, year_idx):
    _file = netCDF4.Dataset(self.files_paths[year_idx], 'r')
    self.files[year_idx] = _file['fields']  
    if self.orography:
      _orog_file = h5py.File(self.orography_path, 'r')
      self.orography_field = _orog_file['orog']
    if self.precip:
      self.precip_files[year_idx] = h5py.File(self.precip_paths[year_idx], 'r')['tp']
    
  def get_data_from_file(self, year_idx):
    if year_idx in self.files:
      return self.files[year_idx]
    _file = netCDF4.Dataset(self.files_paths[year_idx], 'r')
    variables = ['H250', 'H500', 'H850', 'PS', 'Q500', 'Q850', 'SLP', 'T2M', 'T500', 'T850', 'TQV', 'U10M', 'U250', 'U500', 'U850', 'V10M', 'V250', 'V500', 'V850']
    result = np.stack([_file[v] for v in variables], axis=1)
    _file.close()
    if len(self.files) > 40:
      self.files = {}
    self.files[year_idx] = result
    return result
  
  def __len__(self):
    return self.n_samples_total

  def get_indices(self, global_idx):
    file_idx = global_idx // 24
    hour_idx = global_idx % 24
    return file_idx, hour_idx

  def __getitem__(self, global_idx):
    # year_idx = int(global_idx/self.n_samples_per_year) #which year we are on
    # local_idx = int(global_idx%self.n_samples_per_year) #which sample in that year we are on - determines indices for centering
    if self.two_step_training:
        global_idx = min(global_idx, self.n_samples_total-3) if not params['backward'] else max(2, global_idx)
    else:
        global_idx = min(global_idx, self.n_samples_total-2) if not params['backward'] else max(1, global_idx)

    file_idx, hour_idx = self.get_indices(global_idx)

    y_roll = np.random.randint(0, 1440) if self.train else 0#roll image in y direction TODO what is this for?

    #open image file
    data = self.get_data_from_file(file_idx)
    if (not params['backward']) and hour_idx == 23 or (hour_idx == 22 and self.two_step_training):
        data = np.concatenate((data, self.get_data_from_file(file_idx+1)), axis=0)
    elif params['backward'] and hour_idx == 0 or (hour_idx == 1 and self.two_step_training):
        hour_idx += 24
        data = np.concatenate((self.get_data_from_file(file_idx-1), data), axis=0)

    #if self.files[file_idx] is None:
    #    self._open_file(year_idx)

    step = self.dt
    """
    if not self.precip:
      #if we are not at least self.dt*n_history timesteps into the prediction
      if local_idx < self.dt*self.n_history:
          local_idx += self.dt*self.n_history

      #if we are on the last image in a year predict identity, else predict next timestep
      step = 0 if local_idx >= self.n_samples_per_year[year_idx]-self.dt else self.dt
    else:
      inp_local_idx = local_idx
      tar_local_idx = local_idx
      #if we are on the last image in a year predict identity, else predict next timestep
      step = 0 if tar_local_idx >= self.n_samples_per_year[year_idx]-self.dt else self.dt
      # first year has 2 missing samples in precip (they are first two time points)
      # if year_idx == 0:
      #   lim = 1458 #TODO what is this for?
      #   local_idx = local_idx%lim 
      #   inp_local_idx = local_idx + 2
      #   tar_local_idx = local_idx
      #   step = 0 if tar_local_idx >= lim-self.dt else self.dt

    #if two_step_training flag is true then ensure that local_idx is not the last or last but one sample in a year
    if self.two_step_training:
        if local_idx >= self.n_samples_per_year[year_idx] - 2*self.dt:
            #set local_idx to last possible sample in a year that allows taking two steps forward
            local_idx = self.n_samples_per_year[year_idx] - 3*self.dt
    """
    if self.train and self.roll:
      y_roll = random.randint(0, self.img_shape_y)
    else:
      y_roll = 0

    if self.orography:
        orog = self.orography_field[0:720] 
    else:
        orog = None

    if self.train and (self.crop_size_x or self.crop_size_y):
      rnd_x = random.randint(0, self.img_shape_x-self.crop_size_x)
      rnd_y = random.randint(0, self.img_shape_y-self.crop_size_y)    
    else: 
      rnd_x = 0
      rnd_y = 0
      
    if self.precip:
      return reshape_fields(self.files[year_idx][inp_local_idx, self.in_channels], 'inp', self.crop_size_x, self.crop_size_y, rnd_x, rnd_y,self.params, y_roll, self.train), \
                reshape_precip(self.precip_files[year_idx][tar_local_idx+step], 'tar', self.crop_size_x, self.crop_size_y, rnd_x, rnd_y, self.params, y_roll, self.train)
    elif params['backward']:
        if self.two_step_training:
            return reshape_fields(data[(hour_idx-self.dt*self.n_history):(hour_idx+1):self.dt,:,:,:], 'inp', self.crop_size_x, self.crop_size_y, rnd_x, rnd_y,self.params, y_roll, self.train, self.normalize, orog, self.add_noise), \
                    (reshape_fields(data[hour_idx - step:hour_idx - step - 2:-1, :,:,:], 'tar', self.crop_size_x, self.crop_size_y, rnd_x, rnd_y, self.params, y_roll, self.train, self.normalize, orog) if hour_idx > 2 else reshape_fields(data[hour_idx - step::-1, :,:,:], 'tar', self.crop_size_x, self.crop_size_y, rnd_x, rnd_y, self.params, y_roll, self.train, self.normalize, orog))
        else:
            return reshape_fields(data[(hour_idx-self.dt*self.n_history):(hour_idx+1):self.dt, :,:,:], 'inp', self.crop_size_x, self.crop_size_y, rnd_x, rnd_y,self.params, y_roll, self.train, self.normalize, orog, self.add_noise), \
                    reshape_fields(data[hour_idx - step, :,:,:], 'tar', self.crop_size_x, self.crop_size_y, rnd_x, rnd_y, self.params, y_roll, self.train, self.normalize, orog)
    else:
        if self.two_step_training:
            return reshape_fields(data[(hour_idx-self.dt*self.n_history):(hour_idx+1):self.dt, :,:,:], 'inp', self.crop_size_x, self.crop_size_y, rnd_x, rnd_y,self.params, y_roll, self.train, self.normalize, orog, self.add_noise), \
                    reshape_fields(data[hour_idx + step:hour_idx + step + 2, :,:,:], 'tar', self.crop_size_x, self.crop_size_y, rnd_x, rnd_y, self.params, y_roll, self.train, self.normalize, orog)
        else:
            return reshape_fields(data[(hour_idx-self.dt*self.n_history):(hour_idx+1):self.dt, :,:,:], 'inp', self.crop_size_x, self.crop_size_y, rnd_x, rnd_y,self.params, y_roll, self.train, self.normalize, orog, self.add_noise), \
                    reshape_fields(data[hour_idx + step, :,:,:], 'tar', self.crop_size_x, self.crop_size_y, rnd_x, rnd_y, self.params, y_roll, self.train, self.normalize, orog)





    


  
    

