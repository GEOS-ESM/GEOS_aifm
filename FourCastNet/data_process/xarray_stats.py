import rioxarray
import xarray as xr
from calendar import monthrange
import multiprocessing as mp
import numpy as np
import os
from datetime import datetime

def compute_stats(month):
    files = [f'/discover/nobackup/awang17/temp_stats/interp_vars{year}{month:02d}.nc4' for year in range(1991,2021)]
    """
    variables = [
        'PS',
        'SLP',
        'U10M',
        'V10M',
        'T2M',
        'T850', 'U850', 'V850', 'H850', 'Q850',
        'T500', 'U500', 'V500', 'H500', 'Q500',
        'U250', 'V250', 'H250',
        'TQV'
    ]
    variables.extend(['Var_' + var for var in variables])
    print(variables)
    """
    variables = ['T850', 'U850', 'V850', 'Q850']
    def add_time_dim(xda):
        xda = xda.expand_dims(time=[datetime.now()])
        return xda
    dset = xr.open_mfdataset(files, preprocess=add_time_dim)
    #dset = dset[variables]
    #print(dset)
    m = dset.mean(dim='time')
    m.to_netcdf(f'/discover/nobackup/awang17/stats/vars{month:02d}.nc4')
    dset.close()

def main_stats():
    months = [month+1 for month in range(12)]
    with mp.Pool() as pool:
        pool.map(compute_stats, months)

def interpolate_and_compute(date):
    month, year = date
    print("computing", month, year)
    #days = list(range(1, monthrange(year, month)[1]+1))
    days = [monthrange(year, month)[1]]
    variables = ['T850', 'U850', 'V850', 'Q850']
    files = [f'/discover/nobackup/awang17/temp/interpolated{year}{month:02d}{day:02d}.nc4' for day in days]
    for day in days:
        file = f'/css/merra2/MERRA2_all/Y{year}/M{month:02d}/MERRA2.tavg1_2d_slv_Nx.{year}{month:02d}{day:02d}.nc4'
        ds = xr.open_dataset(file)
        dset = ds[variables]
        dset.rio.set_spatial_dims("lon", "lat", inplace=True)
        dset.rio.write_crs("epsg:4326", inplace=True)
        dset = dset.rio.interpolate_na()
        fname = f'/discover/nobackup/awang17/temp/interpolated{year}{month:02d}{day:02d}.nc4'
        files.append(fname)
        print("done", fname)
        dset.to_netcdf(fname)
        ds.close()

    """
    dset = xr.open_mfdataset(files)
    print("opened")
    dset.mean(dim='time').to_netcdf(f'/discover/nobackup/awang17/temp_stats/interp_means{year}{month:02d}.nc4')
    dset.var(dim='time').to_netcdf(f'/discover/nobackup/awang17/temp_stats/interp_vars{year}{month:02d}.nc4')
    dset.close()
    """
    

def interpolate_serial(month, year):
    #days = list(range(1, monthrange(year, month)[1]+1))
    days = [monthrange(year, month)[1]]
    variables = [
        'PS',
        'SLP',
        'U10M',
        'V10M',
        'T2M',
        'T850', 'U850', 'V850', 'H850', 'Q850',
        'T500', 'U500', 'V500', 'H500', 'Q500',
        'U250', 'V250', 'H250',
        'TQV'
    ]
    for day in days:
        file = f'/css/merra2/MERRA2_all/Y{year}/M{month:02d}/MERRA2.tavg1_2d_slv_Nx.{year}{month:02d}{day:02d}.nc4'
        dset = xr.open_dataset(file)
        dset = dset[variables]
        to_interpolate = []
        for var in variables:
            if np.argwhere(np.isnan(dset[var].data)).size > 0:
                to_interpolate.append(var)
        print(to_interpolate)
        """subds = dset[to_interpolate]
        subds.rio.set_spatial_dims("lon", "lat", inplace=True)
        subds.rio.write_crs("epsg:4326", inplace=True)

        dset[to_interpolate] = subds.rio.interpolate_na()
        print("Done: ", file)
        dset.to_netcdf(f'/discover/nobackup/awang17/data/MERRA2.tavg1_2d_slv_Nx.{year}{month:02d}{day:02d}.nc4')
        """

def interpolate(month, year):
    #days = list(range(1, monthrange(year, month)[1]))
    days = [monthrange(year, month)[1]]
    files = [f'/css/merra2/MERRA2_all/Y{year}/M{month:02d}/MERRA2.tavg1_2d_slv_Nx.{year}{month:02d}{day:02d}.nc4' for day in days]
    variables = [
        'PS',
        'SLP',
        'U10M',
        'V10M',
        'T2M',
        'T850', 'U850', 'V850', 'H850', 'Q850',
        'T500', 'U500', 'V500', 'H500', 'Q500',
        'U250', 'V250', 'H250',
        'TQV'
    ]
    dset = xr.open_mfdataset(files)
    dset = dset[variables]
    dset.rio.set_spatial_dims("lon", "lat", inplace=True)
    dset.rio.write_crs("epsg:4326", inplace=True)
    filled = dset.rio.interpolate_na()
    for var in variables:
        if np.argwhere(np.isnan(filled[var].data)).size > 0:
            print(var)
    filled.to_netcdf(f'/discover/nobackup/awang17/data/MERRA2.tavg1_2d_slv_Nx.{year}{month:02d}.nc4')
    filled.mean(dim='time').to_netcdf(f'/discover/nobackup/awang17/data/means{year}{month:02d}.nc4')
    filled.var(dim='time').to_netcdf(f'/discover/nobackup/awang17/data/vars{year}{month:02d}.nc4')

def interpolate_year(year):
    for month in range(12):
        interpolate_serial(month + 1, year)

means = []
variances = []
suffixes = [12] + [i+1 for i in range(11)]
for month in suffixes:
    variables = [
        'PS', 'SLP', 'U10M', 'V10M', 'T2M',
        'H850',
        'T500', 'U500', 'V500', 'H500', 'Q500',
        'U250', 'V250', 'H250',
        'TQV'
    ]
    old = xr.open_dataset(f'/discover/nobackup/awang17/stats/old/means{month:02d}.nc4')
    m = xr.open_dataset(f'/discover/nobackup/awang17/stats/means{month:02d}.nc4')
    m.close()
    means.append(xr.merge([m, old[variables]]))
    v = xr.open_dataset(f'/discover/nobackup/awang17/stats/vars{month:02d}.nc4')
    v.close()
    old_vars = old[['Var_'+name for name in variables]]
    old_vars = old_vars.rename({'Var_'+name: name for name in variables})
    variances.append(xr.merge([v, old_vars]))
means.append(means[0])
variances.append(variances[0])
means.append(means[1])
variances.append(variances[1])
print(len(variances))
print(len(means))

def normalize(data, day, month, year):
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
    
    return (data - m) / std

def preprocess(date):
    print(date)
    month, year = date
    #days = list(range(1, monthrange(year, month)[1]+1))
    days = [monthrange(year, month)[1]]
    for day in days:
        dset = xr.open_dataset(f'/css/merra2/MERRA2_all/Y{year}/M{month:02d}/MERRA2.tavg1_2d_slv_Nx.{year}{month:02d}{day:02d}.nc4')
        dset.close()
        dset_interp = xr.open_dataset(f'/discover/nobackup/awang17/temp/interpolated{year}{month:02d}{day:02d}.nc4')
        dset_interp.close()
        dset_interp = dset_interp.drop(labels='spatial_ref')
        dset.update(dset_interp)
        result = normalize(dset, day, month, year)
        result.to_netcdf(f'/discover/nobackup/awang17/data/normalized_tavg1.{year}{month:02d}{day:02d}.nc4')

def check_nan(date):
    month, year = date
    days = list(range(1, monthrange(year, month)[1]+1))
    variables = [
        'PS',
        'SLP',
        'U10M',
        'V10M',
        'T2M',
        'T850', 'U850', 'V850', 'H850', 'Q850',
        'T500', 'U500', 'V500', 'H500', 'Q500',
        'U250', 'V250', 'H250',
        'TQV'
    ]
    for day in days:
        dset = xr.open_dataset(f'/discover/nobackup/awang17/data/normalized_tavg1.{year}{month:02d}{day:02d}.nc4')
        dset.close()
        for var in variables:
            if np.argwhere(np.isnan(dset[var].data)).size > 0:
                print(year, month, day)



def main():
    years = list(range(1991, 2021))
    print("Years:", years)
    dates = []
    for month in range(1, 13):
        print(month)
        dates = [(month, year) for year in years]
        with mp.Pool() as pool:
            pool.map(preprocess, dates)
    #with mp.Pool() as pool:
    #    pool.map(interpolate_year, years)
    """
    for year in years:
        print(year)
        interpolate_year(year)

    """
#main_stats()
#interpolate_and_compute(1, 2020)
main()
