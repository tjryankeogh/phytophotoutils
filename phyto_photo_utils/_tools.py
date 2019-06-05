#!/usr/bin/env python

from numpy import nan, repeat, concatenate, array, unique, reshape
from pandas import Grouper, DataFrame, read_csv, to_datetime
from tqdm import tqdm
from datetime import timedelta

def remove_outlier_from_time_average(df, time=4, multiplier=3):
    """
    
    Remove outliers when averaging transients before performing the fitting routines, used to improve the signal to noise ratio in low biomass systems.

    The function sets a time window to average over, using upper and lower limits for outlier detection.
    The upper and lower limits are determined by mean ± std * [1].
    The multiplier [1] can be adjusted by the user.

    Parameters
    ----------
    df : pandas.DataFrame
        A dataframe of the raw data, can either be imported from pandas.read_csv or the output from phyto_photo_utils.load
    time : int, default=4
     	The time window to average over, e.g. 4 = 4 minute averages
    multiplier : int, default=3	
        The multiplier to apply to the standard deviation for determining the upper and lower limits.

    Returns
    -------
    df : pandas.DataFrame 
        A dataframe of the time averaged data with outliers excluded.

    Example
    -------
    >>> ppu.remove_outlier_from_time_average(df, time=2, multiplier=3)

    """

    # Convert time window to string
    dt = str(time)+'T'
    # Group data by time window and flashlet number
    grp = df.groupby([Grouper(key='datetime', freq=dt), 'flashlet_number'])
    # Calculate means, standard deviations and counts of the groups
    mean = grp.mean()
    std = grp.std()
    c = grp.count()
    # Calculate upper and lower limits of each group, and repeat each value by its count
    ulim = repeat((mean.fyield.values + std.fyield.values * multiplier), c.fyield.values)
    llim = repeat((mean.fyield.values - std.fyield.values * multiplier), c.fyield.values)

    # Get indexes of data used to create each group
    idx = []
    for i, items in enumerate(grp.indices.items()):
        idx.append(items[-1])

    idx = concatenate(idx, axis=0)

    # Create pandas DataFrame of upper and lower using original indexes of data
    mask = DataFrame([ulim, llim, idx]).T
    mask.columns = ['ulim','llim','index']
    mask = mask.set_index('index').sort_index()

    # Create boolean array using mask DataFrame
    m = (df.fyield.values > mask.ulim) | (df.fyield.values < mask.llim)
    
    # Where condition is True, set values of fluorescence yield to NaN
    df.loc[m.values,'fyield'] = nan

    # Group data that is now corrected
    df = df.groupby([Grouper(key='datetime', freq=dt), 'flashlet_number']).mean().reset_index()
    
    # Return number of measurements that is used to create each average
    df['nseq'] = c.fyield.values
    
    return df

def correct_fire_instrument_bias(df, sat=False, pos=1, sat_len=100):
    
    """
    
    Corrects for instrumentation bias by calculating difference between flashlet 0 & flashlet[pos].
    This bias is then added to the relaxation phase.

    Parameters
    ----------
    df : pandas.DataFrame 
        A dataframe of the raw data, can either be imported from pandas.read_csv or the output from phyto_photo_utils.load
    sat : bool, default=False
     	If True, correct using bias of saturation phase. If False correct using bias of relaxation phase.
    pos : int, default=1
     	The flashlet number after the start of the phase, either saturation or relaxation, to calculate difference between.
    sat_len : int, default=100
     	The length of saturation measurements.

    Returns
    -------
    df : pandas.DataFrame 
        A dataframe of FIRe data corrected for the instrument bias.

    Example
    -------
    >>> ppu.correct_fire_bias_correction(df, sat=False, pos=1, sat_len=100)

    """

    fyield = array(df.fyield)
    seq = array(df.seq)
    
    ycorr = []
    
    # Loop through data using unique measurement (seq) number
    for s in tqdm(unique(seq)):

        i = seq == s
        y = fyield[i]

        # If sat is True will use bias of saturation phase
        if sat:
            d = y[pos] - y[0]
            y[sat_len:] += d
        
        # If sat is False will use bias of relaxation phase
        else:
            d = y[sat_len - pos] - y[sat_len]
            y[sat_len:] += d
        
        ycorr.append(y)
        
    fyield_corr = array(ycorr)
    fyield_corr = reshape(fyield_corr, (fyield_corr.shape[0] * fyield_corr.shape[1]))
    
    # Replace fluorescence yield in DataFrame with bias corrected data 
    df['fyield'] = fyield_corr
    
    return df

def calculate_blank_FastOcean(file_, seq_len=100):
     
    """
    Calculates the blank by averaging the fluorescence yield for the saturation phase.

    Parameters
    ----------
    file_ : str
        The path directory to the raw blank file in csv format.
    seq_len : int, default=100
        The length of the measurement sequence.
    
    Returns
    -------
    res : pandas.DataFrame
        The blank results.

    Example
    -------
    >>> ppu.calculate_blank_FastOcean(file_, seq_len=100)
    """

    df = read_csv(file_, skiprows=26, nrows=2, header=None)
    df = df.iloc[:,2:].T
    df.columns = ['date', 'time']
    df['datetime'] = to_datetime(df.date.values+' '+df.time.values)
    df = df.drop(columns=['date','time'])

    res = read_csv(file_, skiprows=43, nrows=seq_len, header=None)
    res = res.iloc[:,2:]
    res = res.agg(['mean','std']).T
    res.columns = ['blank_mean', 'blank_stdev']
    res = DataFrame(res)
    res['datetime'] = df.datetime

    return res

def calculate_blank_FIRe(file_):

    """
    Calculates the blank by averaging the fluorescence yield for the saturation phase.

    Parameters
    ----------
    file_ : str
        The path directory to the raw blank file.

    Returns
    -------
    res : pandas.DataFrame
        The blank results: blank, datetime

    Example
    -------
    >>> ppu.calculate_blank_FIRe(file_)
    """

    df = read_csv(file_)
    # Get date and time data
    dt=str(df.iloc[0,:].values).strip()[2:-2].split('  ')
    date = dt[0].strip()
    time = dt[-1].strip()
    time = str(timedelta(seconds=int(time)))
    datetime = date+' '+time
    format = '%m/%d/%Y %H:%M:%S'
    dt = to_datetime(datetime, format=format)

    # Get saturation phase length
    sat_len = int(str(df.iloc[5,:].values).split()[-1][:-2])

    # Read in the actual data and calculate the mean
    df = read_csv(file_, index_col=0, skiprows=20, header=None, delim_whitespace=True)
    df.columns = ['time', 'ex', 'fyield']
    blank = df.fyield[:sat_len].mean(axis=0)
    stdev = df.fyield[:sat_len].std(axis=0)
    data = array([blank, stdev, dt]).T
    res = DataFrame(data)
    res = res.T
    res.columns = ['blank_mean', 'blank_stdev','datetime']

    return res

