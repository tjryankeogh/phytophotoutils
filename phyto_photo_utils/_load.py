#!/usr/bin/env python

from pandas import read_csv, to_datetime, DataFrame, concat, Series
from numpy import array, repeat, arange, squeeze, reshape, r_, nansum
from os import chdir
from csv import reader

def load_FIRe_files(file_, append=False, save_files=False, res_path=None, 
                   seq_len=160, flen=1e-6, irrad=None):
    """

    Process the raw data file (.000 format) and convert to a csv with standard formatting.

    Parameters
    ----------
    file_ : str
        The path directory to the .000 data file from benchtop SAtlantic FIRe.
    append : bool, default=False
        If True, multiple files will be concatenated together.
    save_files : bool, default=False
        If True, files will be saved as .csv.
    res_path : str, default=None           
        The path directory where to save files, only required if save_files = True.
    seq_len : int , default=160               
        The number of flashlets in the protocol.
    flen : float, default=1e-6
        The flashlet length in seconds.
    irrad : int, default=None                 
        The LED output in μE m\ :sup:`2` s\ :sup:`-1`.

    Returns
    -------
    df : pandas.DataFrame, shape=[n,6]
        A dataframe of the raw fluorescence data with columns as below:
    flashlet_number : np.array, dtype=int, shape=[n,]
        A sequential number from 1 to ``seq_len``
    fyield : np.array, dtype=float, shape=[n,]
        The raw fluorescence yield data.
    datetime : np.array, dtype=datetime64, shape=[n,]
        The date and time of measurement.
    seq : np.array, dtype=int, shape=[n,]
        The sample measurement number.
    seq_time : np.array, dtype=float, shape=[n,]
        The measurement sequence time in μs.
    pfd : np.array, dype=float, shape=[n,]
        The photon flux density in μmol photons m\ :sup:`2` s\ :sup:`-1`.

    Example
    -------
    >>> fname = './data/raw/instrument/fire/FIRe_example.000'
    >>> output = './data/raw/ppu/fire/'
    >>> df = ppu.load_FIRe_files(fname, append=False, save_files=True, res_path=output, seq_len=160, flen=1e-6, irrad=47248)
       
    """

    format = '%m/%d/%Y%H:%M:%S.%f'
    
    name = file_.split('/')[-1].split('.')[0]
    data = read_csv(file_, header=None)

    cur_pos = 0
    temp_data = []
    fl_num = []
    final_data = []
    seq_time = []
    index = []

    is_prev_alpha = False
    while True:
        row = data.values[cur_pos][0]
        row_str = row.strip()
        is_alpha = row_str[0].isalpha()

        if not is_prev_alpha and is_alpha:
            row_str = row_str[:-10]
            row_str = row_str.replace(" ","")
            dt_str = row_str[10:]
            dt = to_datetime(dt_str, format=format)

        if len(temp_data) > 0 and (not is_prev_alpha) and is_alpha:
            temp_data = array(temp_data)
            
            fl_num.append(temp_data[:seq_len, 0])
            seq_time.append(temp_data[:seq_len, 1])
            final_data.append(temp_data[:seq_len, 2])
            index.append(dt)
                         
            temp_data = []

        if not is_alpha:
            temp_data.append(list(map(float, row_str.split())))

        cur_pos += 1
        is_prev_alpha = is_alpha

        if cur_pos == len(data.values)-1:
            break
    
    flashlet = array(fl_num)
    flashlet = Series(squeeze(reshape(flashlet, (1, (flashlet.shape[0]*flashlet.shape[1])))))
    time = array(seq_time)
    time = Series(squeeze(reshape(time, (1, (time.shape[0]*time.shape[1])))))
    data = array(final_data)
    datetime = repeat(index, data.shape[1])
    data = Series(squeeze(reshape(data, (1, (data.shape[0]*data.shape[1])))))
    
    df = DataFrame(data=[flashlet, data, datetime, time])
    df = df.T
    df.columns = ['flashlet_number', 'fyield', 'datetime', 'seq_time']
    df['seq'] = repeat(arange(0,(int(df.shape[0] / seq_len)),1), seq_len)
    sigscale = 1e-20
    df['pfd'] = (irrad * 1e-6 * 6.02e23) * flen * sigscale
    df['fyield'] = df.fyield.astype('float')
    df['seq_time'] = df.seq_time.astype('int')
    
    df = df[['flashlet_number','fyield','datetime','seq','seq_time','pfd']]

    list_ = []

    # If True, multiple files will be concatenated together
    if append:
        list_.append(df)
        df = concat(list_, axis=0)
        df['seq'] = repeat(arange(0, (df.shape[0] / seq_len), 1), seq_len)
    
    # If True, files will be saved as .csv in path directory specified by res_path
    if save_files:
        if res_path is None:
            print('WARNING: Files not saved. No output directory specified.')
        else:
            df.to_csv(res_path+str(name)+'.csv')
        
    return df


def load_FASTTrackaI_files(file_, append=False, save_files=False, res_path=None, seq_len=120, irrad=None):
    """

    Process the raw data file and convert to a csv with standard formatting.

    Parameters
    ----------
    file_ : str
        The path directory to the .000 data file from benchtop SAtlantic FIRe.
    append : bool, default=False
        If True, multiple files will be concatenated together.
    save_files : bool, default=False
        If True, files will be saved as .csv.
    res_path : dir, default=None             
        The path directory where to save files, only required if save_files = True.
    seq_len : int, default=120         
        The number of flashlets in the protocol.
    irrad : int, default=None            
        The light/dark chamber photons per count from the calibration file.

    Returns
    -------
    df : pandas.DataFrame, shape=[n,8]
        A dataframe of the raw fluorescence data with columns as below:
    flashlet_number : np.array, dtype=int, shape=[n,]
        A sequential number from 1 to ``seq_len``
    fyield : np.array, dtype=float, shape=[n,]
        The raw fluorescence yield data.
    datetime : np.array, dtype=datetime64, shape=[n,]
        The date and time of measurement.
    seq : np.array, dtype=int, shape=[n,]
        The sample measurement number.
    seq_time : np.array, dtype=float, shape=[n,]
        The measurement sequence time in μs.
    pfd : np.array, dype=float, shape=[n,]
        The photon flux density in μmol photons m\ :sup:`2` s\ :sup:`-1`.
    channel : np.array, dtype=str, shape=[n,]
        The chamber used for measurements, A = light chamber, B = dark chamber.
    gain : np.array, dtype=int, shape=[n,]
        The gain settings of the instrument.
    
    Example
    -------
    >>> fname = './data/raw/instrument/fasttrackai/FASTTrackaI_example.csv'
    >>> output = './data/raw/ppu/fasttrackai/'
    >>> df = ppu.load_FASTTrackaI_files(fname, append=False, save_files=True, res_path=output, seq_len=120, irrad=545.62e10)
       
    """    
    df = read_csv(file_, header=0)
    name = file_.split('/')[-1].split('.')[0]
    
    # Selecting only rows that contain metadata
    idx = df.shape[0]
    rows = arange(0, idx+1, seq_len+1)

    DESIRED_ROWS = rows
    with open(file_) as input_file:
        creader = reader(input_file)

        desired_rows = [row for row_number, row in enumerate(creader)
                        if row_number in DESIRED_ROWS]

    md = DataFrame(desired_rows)
    # Calculation of date + time for each measurement
    dt = md.iloc[:, 10].str.replace(':','/') + ' ' + md.iloc[:, 11]
    dt = to_datetime(dt.values)
    gain = md.iloc[:, 12].values
    md = md.iloc[:, 1:-18]
    md.columns = ['seq','channel','sflen','sat_len','sper','rflen','rel_len','rper']
    md['gain'] = gain
    md['datetime'] = dt
    
    # Calculation of saturation and relaxation flashlet length
    md['sflen'] = 1.1 + (md.sflen.astype('float64') - 4) * 0.06
    md['rflen'] = 1.1 + (md.rflen.astype('float64') - 4) * 0.06
    md['sper'] = 2.8 + md.sper.astype('float64') * 0.08
    md['rper'] = 2.8 + md.rper.astype('float64') * 0.08
    md['sat_len'] = md.sat_len.astype('float64')
    md['sflen'] = md.sflen.astype('float64')
    md['rel_len'] = md.rel_len.astype('float64')
    md['rflen'] = md.rflen.astype('float64')

    # Calculation of sequence time
    stime = []
    for i in range(md.shape[0]):
        s = (arange(0, md.sat_len[i], 1) * md.sper[i] + md.sflen[i]) * 1e-6
        r = (arange(0, md.rel_len[i], 1) * md.rper[i] + md.rflen[i]) * 1e-6
        r = (r+(md.sper[i]*1e-6))+s[-1]
        y = Series(r_[s,r])
        stime.append(y)
    stime = concat(stime, axis=0)
    
    # Selecting rows that contain the raw data
    df = read_csv(file_, header=None, skiprows=rows)
    df.columns= ['flashlet_number','ex', 'em']
    df['fyield'] = df.em / df.ex
    df['datetime'] = repeat(md.datetime.values, seq_len)
    df['seq'] = repeat(md.seq.values, seq_len)
    df['seq_time'] = stime.values * 1e6
    df['channel'] = repeat(md.channel.values, seq_len)

    # Calculation of photon flux density
    sigscale = 1e-20
    df['pfd'] = (df.ex * irrad) * sigscale
    df['gain'] = repeat(md.gain.values, seq_len) # won't be need if gain correction implemented here
    df = df.drop(['ex','em','flashlet_number'], axis=1)
    idx = int(df.seq.max()) + 1
    df['flashlet_number'] = array(list(arange(0,seq_len,1)) * idx)
    
    ##TO DO##
    #add if statements to handle the gain settings   
    #if irf:
    #    if df.channel == 'A':
    #        if df.gain == 1:
    #            df.fyield /= irf.

    df = df[['flashlet_number','fyield','datetime','seq','seq_time','pfd','channel','gain']]
    
    list_ = []
    # If True, multiple files will be concatenated together
    if append:
        list_.append(df)
        df = concat(list_, axis=0)
        df['seq'] = repeat(arange(0, (df.shape[0] / seq_len), 1), seq_len)
    
    # If True, files will be saved as .csv in path directory specified by res_path
    if save_files:
        if res_path is None:
            print('WARNING: Files not saved. No output directory specified.')
        else:
            df.to_csv(res_path+str(name)+'.csv')
        
    return df

def load_FastOcean_files(file_, append=False, save_files=False, led_separate=False, res_path=None, 
                   seq_len=140, flen=1e-6):
    """

    Process the raw data file and convert to a csv with standard formatting.

    Parameters
    ----------
    file_ : dir
        The path directory to the .000 data file from benchtop SAtlantic FIRe.
    append : bool, default=False
        If True, multiple files will be concatenated together.
    save_files : bool, default=False
        If True, files will be saved as .csv.
    led_separate : bool, default=False
        If True, the protocols will be separated dependent upon the LED sequence.
    res_path : dir               
        The path directory where to save files, only required if save_files = True.
    seq_len : int, default=140                 
        The number of flashlets in the protocol.
    flen : float, default=2e-6
        The flashlet length in seconds.

    Returns
    -------

    df : pandas.DataFrame, shape=[n,7]
        A dataframe of the raw fluorescence data with columns as below:
    flashlet_number : np.array, dtype=int, shape=[n,]
        A sequential number from 1 to ``seq_len``
    fyield : np.array, dtype=float, shape=[n,]
        The raw fluorescence yield data.
    datetime : np.array, dtype=datetime64, shape=[n,]
        The date and time of measurement.
    seq : np.array, dtype=int, shape=[n,]
        The sample measurement number.
    seq_time : np.array, dtype=float, shape=[n,]
        The measurement sequence time in μs.
    pfd : np.array, dype=float, shape=[n,]
        The photon flux density in μmol photons m\ :sup:`2` s\ :sup:`-1`.
    led_sequence : np.array, dtype=int, shape=[n,]
        The LED combination using during the measurement, see example below.

    Example
    -------
    >>> fname = './data/raw/instrument/fire/FastOcean_example.000'
    >>> output = './data/raw/ppu/fastocean/'
    >>> df = ppu.load_FastOcean_files(fname, append=False, save_files=True, led_separate=False, res_path=output, seq_len=140, flen=2e-6)
    >>> led_sequence == 1, LED 450 nm
    >>> led_sequence == 2, LED 450 nm + LED 530 nm
    >>> led_sequence == 3, LED 450 nm + LED 624 nm
    >>> led_sequence == 4, LED 530 nm + LED 624 nm
    >>> led_sequence == 5, LED 450 nm + LED 530 nm + LED 624 nm
    >>> led_sequence == 6, LED 530 nm
    >>> led_sequence == 7, LED 624 nm
    """    
    name = file_.split('/')[-1].split('.')[0]
    
    md = read_csv(file_, skiprows=16, nrows=12, header=None)
    md = md.T.iloc[2:,:]
    md.columns = ['seq', 'led450', 'led530', 'led624', 'PMT', 'a', 'pitch', 'reps', 'int', 'b', 'date', 'time']
    md = md.drop(['a','b'],axis=1).reset_index(drop=True)
    md['seq'] = md.seq.str.replace('A','').astype('int')
    pfd = md.iloc[:,1:4].astype(float).sum(axis=1)
    nled = md.iloc[:,1:4].count(axis=1)
    
    #determining LED sequence for filtering results
    led = md.iloc[:,1:4].astype(float)
    led = led.isnull()
    led_idx = []
    for i in range(led.shape[0]):
        if (led.iloc[i,1] == True) & (led.iloc[i,2] == True): # LED 1 only
            led_idx.append(1)
        elif (led.iloc[i,0] == False) & (led.iloc[i,1] == False): # LED 1 & 2
            led_idx.append(2)
        elif (led.iloc[i,0] == False) & (led.iloc[i,2] == False): # LED 1 & 3
            led_idx.append(3)
        elif (led.iloc[i,1] == False) & (led.iloc[i,2] == False): # LED 2 & 3
            led_idx.append(4)
        elif (led.iloc[i,0] == False) & (led.iloc[i,1] == False) & (led.iloc[i,2] == False): # ALL LEDs
            led_idx.append(5)
        elif (led.iloc[i,1] == False): # LED 2 only
            led_idx.append(6)
        elif (led.iloc[i,2] == False): # LED 3 only
            led_idx.append(7)
    
    df = read_csv(file_, skiprows=43, nrows=seq_len, header=None)
    seq_time = df.iloc[:,0]
    flashlet_number = df.iloc[:,1]
    df = df.iloc[:,2:]
    
    dfm = []
    for i in range(df.shape[1]):
        data = df.iloc[:,i]
        dfm.append(data)
    
    sigscale=1e-20  

    df = DataFrame(concat(dfm, axis=0)).reset_index(drop=True)
    df.columns = ['fyield']
    df['pfd'] = repeat((pfd.values*1e22), seq_len) * flen * sigscale
    df['seq'] = repeat(md.seq.values, seq_len)
    df['seq_time'] = array(list(seq_time) * len(md.seq))
    df['flashlet_number'] = array(list(flashlet_number) * len(md.seq))
    df['datetime'] = to_datetime(repeat((md.date.values+' '+md.time.values), seq_len), dayfirst=True)
    df['led_sequence'] = repeat(array(led_idx), seq_len)
    
    df = df[['flashlet_number','fyield','datetime','seq','seq_time','pfd','led_sequence']]

    list_ = []
    if append:
        list_.append(df)
        df = concat(list_, axis=0)
        df['seq'] = repeat(arange(0, (df.shape[0] / seq_len), 1), seq_len)
    
    if save_files:
        if res_path is None:
            print('WARNING: Files not saved. No output directory specified.')
        else:
            df.to_csv(res_path+str(name)+'.csv')
            
            if led_separate:
                i = df.led_sequence == 1
                dfi = df[i]
                if len(dfi) > 0:
                    dfi.to_csv(res_path+str(name)+'_L1.csv')
                i = df.led_sequence == 2
                dfi = df[i]
                if len(dfi) > 0:
                    dfi.to_csv(res_path+str(name)+'_L12.csv')
                i = df.led_sequence == 3
                dfi = df[i]
                if len(dfi) > 0:
                    dfi.to_csv(res_path+str(name)+'_L13.csv')
                i = df.led_sequence == 4
                dfi = df[i]
                if len(dfi) > 0:
                    dfi.to_csv(res_path+str(name)+'_L23.csv')
                i = df.led_sequence == 5
                dfi = df[i]
                if len(dfi) > 0:
                    dfi.to_csv(res_path+str(name)+'_L123.csv')
                i = df.led_sequence == 6
                dfi = df[i]
                if len(dfi) > 0:
                    dfi.to_csv(res_path+str(name)+'_L2.csv')
                i = df.led_sequence == 7
                dfi = df[i]
                if len(dfi) > 0:
                    dfi.to_csv(res_path+str(name)+'_L3.csv')
    
    return df