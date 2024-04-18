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

import numpy as np
from netCDF4 import Dataset


channels = [0, 1, 3, 5, 6, 7]  # t2m, ps, tqv, z500 t500, z300
path = "/discover/nobackup/adasilva/FCN/MERRA-2/{year}.nc4"
global_means = np.zeros((1, len(channels), 361, 576))
global_stds = np.zeros((1, len(channels), 361, 576))
train_years = [y for y in range(2000, 2020)]
for year in train_years:
	file = Dataset(path.format(year=year), 'r')
	print("Year", year, ", file shape", file["fields"].shape)
	for i, channel in enumerate(channels):
		slice = np.array(file["fields"][:,channel:channel+1,:,:])
		mask = slice > 1e30
		if len(np.unique(mask)) > 1:
			print(np.where(mask))
		masked_slice = np.ma.masked_array(slice, mask=mask)
		global_means[:,i:i+1,:,:] += np.mean(masked_slice, keepdims=True, axis=0)
		global_stds[:,i:i+1,:,:]  += np.var(masked_slice, keepdims=True, axis=0)
		print(global_means[0,:,0,0])
		print(global_stds[0,:,0,0])

global_means /= len(train_years)
global_stds = np.sqrt(global_stds / len(train_years))

np.save('/discover/nobackup/awang17/stats/global_means.npy', global_means)
np.save('/discover/nobackup/awang17/stats/global_stds.npy', global_stds)

print("means: ", global_means)
print("stds: ", global_stds)







