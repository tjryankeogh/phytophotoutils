#!/usr/bin/env python

from pandas import read_csv, to_datetime, DataFrame, concat, Series
from numpy import array, repeat, arange, squeeze, reshape, r_, nansum
from os import chdir
from csv import reader
from datetime import timedelta

def load_FIRe_files(file_, append=False, save_files=False, res_path=None, seq_len=160, 
                    flen=1e-6, irrad=None, continuous=False, light_step=False, single_turnover=True):
    """

    Process the raw data file (.000 format) and convert to a csv with standard formatting.

    Parameters
    ----------
    file_ : str
        The path directory to the data file from benchtop SAtlantic FIRe.
    append : bool, default=False
        If True, multiple files will be concatenated together.
    save_files : bool, default=False
        If True, files will be saved as .csv.
    res_path : str, default=None           
        The path directory where to save files, only required if save_files = True.
    seq_len : int , default=160               
        The number of flashlets in the protocol. Only required if continuous = True.
    flen : float, default=1e-6
        The flashlet length in seconds. 
    irrad : int, default=None                 
        The LED output in μE m\ :sup:`2` s\ :sup:`-1`. Only required if continuous = True.
    continuous : bool, default=False
        If True, will load files from the continuous format. If False, will load the discrete file format.
    light_step: bool, default=False
        If True, will load files from a discrete format FLC file. If False, will load the discrete file format with no light steps.
    single_turnover: bool, default=True
        If True, will load the saturation and relaxation from the single turnover measurement. If False, will load the multiple turnover measurement.

    Returns
    -------
    df : pandas.DataFrame, shape=[n,6]
        A dataframe of the raw fluorescence data with columns as below:
    flashlet_number : np.array, dtype=int, shape=[n,]
        A sequential number from 1 to ``seq_len``
    flevel : np.array, dtype=float, shape=[n,]
        The raw fluorescence data.
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

    

    if continuous:
        
        name = file_.split('/')[-1].split('.')[0]
        format = '%m/%d/%Y%H:%M:%S.%f'
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
        df['flevel'] = df.fyield.astype('float')
        df['seq_time'] = df.seq_time.astype('int')

    else:
        
        name = file_.split('/')[-1].split('.')[0]
        name2 = file_.split('/')[-1].split('.')[1]
        name = name+'_'+name2

        df = read_csv(file_, nrows = 16)
        dt = str(df.iloc[0,:].values).strip()[2:-2].split('  ')
        date = dt[0].strip()
        time = dt[-1].strip()
        time = str(timedelta(seconds=int(time)))
        datetime = date+' '+time
        gain = int(str(df.iloc[2,:].values).strip()[2:-2].split(':')[1])

        sat_len = int(str(df.iloc[5,:].values).split()[-1][:-2])
        rel_len = int(str(df.iloc[6,:].values).split()[-1][:-2])
        msat_len = int(str(df.iloc[9,:].values).split()[-1][:-2])
        mrel_len = int(str(df.iloc[10,:].values).split()[-1][:-2])
        if light_step:
            df = read_csv(file_, index_col=0, skiprows=24, header=None, delim_whitespace=True)
        else:
            df = read_csv(file_, index_col=0, skiprows=21, header=None, delim_whitespace=True)
        df.columns = ['seq_time', 'ex', 'flevel']
        df['pfd'] = (df.ex * 1e-6).drop(columns = 'ex')
        df['datetime'] = to_datetime(datetime)
        if single_turnover:
            df = df.iloc[:sat_len+rel_len,:]
            df['seq'] = int(df.shape[0] / (sat_len + rel_len))
            df['flashlet_number'] = arange(1, (sat_len + rel_len)+1, 1)
        else: 
            df = df.iloc[sat_len+rel_len:,:].reset_index(drop=True)
            df['seq'] = int(df.shape[0] / (msat_len + mrel_len))
            df['flashlet_number'] = arange(1, (msat_len + mrel_len)+1, 1)
    
    df = df[['flashlet_number','flevel','datetime','seq','seq_time','pfd']]

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
    flevel : np.array, dtype=float, shape=[n,]
        The raw fluorescence data.
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
    df['flevel'] = df.em / df.ex
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

    df = df[['flashlet_number','flevel','datetime','seq','seq_time','pfd','channel','gain']]
    
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
                   seq_len=125, seq_reps=None, flen=1e-6, delimiter=',', FastAct=True, Single_Acq=False):
    """

    Process the raw data file and convert to a csv with standard formatting.

    Parameters
    ----------
    file_ : dir
        The path directory to the .csv data file from the FastOcean with either the FastAct1 or FastAct2 laboratory system.
    append : bool, default=False
        If True, multiple files will be concatenated together. Not applicable if Single_Acq = True.
    save_files : bool, default=False
        If True, files will be saved as .csv.
    led_separate : bool, default=False
        If True, the protocols will be separated dependent upon the LED sequence.
    res_path : dir               
        The path directory where to save files, only required if save_files = True.
    seq_len : int, default=125                 
        The number of flashlets in the protocol.
    seq_reps : int, default=None
        The number of sequences per acquisition or in FastAct2 multiple acquisition files the number of light steps.
    flen : float, default=1e-6
        The flashlet length in seconds.
    delimiter : str, default=','
        Specify the delimiter to be used by Pandas.read_csv for loading the raw files.
    FastAct : bool, default=True
        If True, will load data from FastAct1 laboratory system format. If False, will load data from FastAct2 laboratory system format.
    Single_Acq : bool, default=False
        If True, will load a single acquisition data from either the FastAct or FastAct2 laboratory system, dependent upon whether FastAct is True of False.

    Returns
    -------

    df : pandas.DataFrame, shape=[n,7]
        A dataframe of the raw fluorescence data with columns as below:
    flashlet_number : np.array, dtype=int, shape=[n,]
        A sequential number from 1 to ``seq_len``
    flevel : np.array, dtype=float, shape=[n,]
        The raw fluorescence data.
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
    >>> fname = './data/raw/instrument/fire/FastOcean_example.csv'
    >>> output = './data/raw/ppu/fastocean/'
    >>> df = ppu.load_FastOcean_files(fname, append=False, save_files=True, led_separate=False, res_path=output, seq_len=125, flen=1e-6)
    >>> led_sequence == 1, LED 450 nm
    >>> led_sequence == 2, LED 450 nm + LED 530 nm
    >>> led_sequence == 3, LED 450 nm + LED 624 nm
    >>> led_sequence == 4, LED 530 nm + LED 624 nm + LED 624 nm
    >>> led_sequence == 5, LED 530 nm + LED 624 nm 
    >>> led_sequence == 6, LED 530 nm
    >>> led_sequence == 7, LED 624 nm
    """    
    name = file_.split('/')[-1].split('.')[0]
    sigscale=1e-20 

    if FastAct:
        if Single_Acq:
            md = read_csv(file_, delimiter=delimiter, skiprows=11, nrows=6, usecols=[0,1], header=None)
            datetime = to_datetime(str(md.iloc[4,1])+' '+str(md.iloc[5,1]))
            led = md.iloc[0:3,1].astype(float)
            led = led.isnull()
            led_idx = []
            if (led.iloc[0] == False) & (led.iloc[1] == True) & (led.iloc[2] == True):
                led_idx.append(1)
            elif (led.iloc[0] == False) & (led.iloc[1] == False) & (led.iloc[2] == True):
                led_idx.append(2)
            elif (led.iloc[0] == False) & (led.iloc[2] == False) & (led.iloc[1] == True):
                led_idx.append(3)
            elif (led.iloc[0] == False) & (led.iloc[1] == False) & (led.iloc[2] == False):
                led_idx.append(4)
            elif (led.iloc[1] == False) & (led.iloc[2] == False) & (led.iloc[0] == True):
                led_idx.append(5)
            elif (led.iloc[1] == False) & (led.iloc[0] == True) & (led.iloc[2] == True):
                led_idx.append(6)
            elif (led.iloc[2] == False) & (led.iloc[0] == True) & (led.iloc[1] == True):
                led_idx.append(7)
            led1 = array(float(md.iloc[0,1]))
            led2 = array(float(md.iloc[1,1]))
            led3 = array(float(md.iloc[2,1]))
            led = nansum([led1, led2, led3])
            df = read_csv(file_, delimiter=delimiter, skiprows = 21, nrows=seq_len, header=None).drop(columns=[2,3])
            flashlet_number = df.iloc[:,1]
            seq_time = df.iloc[:,0]
            seq_reps = df.iloc[:,2:].shape[1]
            df = Series(reshape(df.iloc[:,2:(seq_reps+2)].values.T, [-1]))
            datetime = repeat(Series(datetime), seq_len * seq_reps)
            pfd = repeat(Series((led * 1e22) * flen * sigscale), seq_len * seq_reps)
            seq_time = concat([Series(seq_time)] * seq_reps)
            flashlet_number = repeat(Series(flashlet_number), seq_reps)
            led_sequence = repeat(array(led_idx), seq_len * seq_reps)
            seq = concat([Series(arange(1, seq_reps+1))] * seq_len).sort_index()
            df = DataFrame([flashlet_number.values, df.values, datetime.values, seq.values, seq_time.values, pfd.values, led_sequence]).T
            df.columns = ['flashlet_number','flevel','datetime','seq','seq_time','pfd','led_sequence']
        else:
            md = read_csv(file_, skiprows=16, nrows=12, header=None, delimiter=delimiter)
            md = md.T.iloc[2:,:]
            md.columns = ['seq', 'led450', 'led530', 'led624', 'PMT', 'a', 'pitch', 'reps', 'int', 'b', 'date', 'time']
            md = md.drop(['a','b'],axis=1).reset_index(drop=True)
            md['seq'] = md.seq.str.replace('A','').astype('int')
            pfd = md.iloc[:,1:4].astype(float).sum(axis=1)
            nled = md.iloc[:,1:4].count(axis=1)
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
                    led_idx.append(5)
                elif (led.iloc[i,0] == False) & (led.iloc[i,1] == False) & (led.iloc[i,2] == False): # ALL LEDs
                    led_idx.append(4)
                elif (led.iloc[i,1] == False): # LED 2 only
                    led_idx.append(6)
                elif (led.iloc[i,2] == False): # LED 3 only
                    led_idx.append(7)
            
            df = read_csv(file_, skiprows=43, nrows=seq_len, header=None, delimiter=delimiter)
            seq_time = df.iloc[:,0]
            flashlet_number = df.iloc[:,1]
            df = df.iloc[:,2:]
            dfm = []
            for i in range(df.shape[1]):
                data = df.iloc[:,i]
                dfm.append(data)

            df = DataFrame(concat(dfm, axis=0)).reset_index(drop=True)
            df.columns = ['flevel']
            df['pfd'] = repeat((pfd.values*1e22), seq_len) * flen * sigscale
            df['seq'] = repeat(md.seq.values, seq_len)
            df['seq_time'] = array(list(seq_time) * len(md.seq))
            df['flashlet_number'] = array(list(flashlet_number) * len(md.seq))
            df['datetime'] = to_datetime(repeat((md.date.values+' '+md.time.values), seq_len), dayfirst=True)
            df['led_sequence'] = repeat(array(led_idx), seq_len)   
            df = df[['flashlet_number','flevel','datetime','seq','seq_time','pfd','led_sequence']]
    
    else:
        if Single_Acq:
            df = read_csv(file_, skiprows=2, nrows=18, usecols=[0,1], header=None)
            date = df.iloc[0,1]
            time = df.iloc[1,1]
            datetime = repeat(to_datetime(date+' '+time, dayfirst=True), seq_len * 4 * seq_reps)
            fln = concat([Series(arange(1,seq_len+1, 1))]*4*seq_reps).reset_index(drop=True)
            led_seq = Series(repeat(arange(1,5,1), seq_len*seq_reps))
            led1 = Series(repeat(((float(df.iloc[13,1]) * 1e22) * flen * sigscale), seq_len))
            led2 = Series(repeat(((float(df.iloc[14,1]) * 1e22) * flen * sigscale), seq_len))
            led3 = Series(repeat(((float(df.iloc[15,1]) * 1e22) * flen * sigscale), seq_len))
            df = read_csv(file_, skiprows=41, nrows=seq_len, header=None)
            seq_time = concat([df.iloc[:,1]]*4*seq_reps).reset_index(drop=True)
            df1 = Series(reshape(df.iloc[:,3:(seq_reps+3)].values.T, [-1]))
            df2 = Series(reshape(df.iloc[:,(seq_reps+6):((seq_reps+3)*2)].values.T, [-1]))
            df3 = Series(reshape(df.iloc[:,((seq_reps+4)*2)+1:((seq_reps+3)*3)].values.T, [-1]))
            df4 = Series(reshape(df.iloc[:,((seq_reps+4)*3):(seq_reps*5)-9].values.T, [-1]))
            dfc = concat([df1, df2, df3, df4]).reset_index(drop=True)
            pfd = concat([led1, led1+led2, led1+led3, led1+led2+led3]*seq_reps).reset_index(drop=True)
            df = DataFrame([fln, dfc, datetime, seq_time, pfd, led_seq]).T
            df.columns = ['flashlet_number','flevel','datetime','seq_time','pfd','led_sequence']
        
        else:
            df = read_csv(file_, skiprows=2, nrows=16, usecols=[0,1], header=None)
            date = df.iloc[0,1]
            time = df.iloc[1,1]
            datetime = repeat(to_datetime(date+' '+time, dayfirst=True), seq_len * 4 * seq_reps)
            fln = concat([Series(arange(1,seq_len+1, 1))] * 4 * seq_reps).values
            led_seq = Series(repeat(arange(1,5,1), seq_len * seq_reps))
            led1 = Series(repeat(((float(df.iloc[13,1]) * 1e22) * flen * sigscale), seq_len))
            led2 = Series(repeat(((float(df.iloc[14,1]) * 1e22) * flen * sigscale), seq_len))
            led3 = Series(repeat(((float(df.iloc[15,1]) * 1e22) * flen * sigscale), seq_len))
            df = read_csv(file_, skiprows=46, nrows=seq_len, header=None)
            seq_time = concat([df.iloc[:,0]] * (4 * seq_reps))
            df1 = Series(reshape(df.iloc[:,2:2+seq_reps].values.T, [-1]))
            df = read_csv(file_, skiprows=212, nrows=seq_len, header=None)
            df2 = Series(reshape(df.iloc[:,2:2+seq_reps].values.T, [-1]))
            df = read_csv(file_, skiprows=378, nrows=seq_len, header=None)
            df3 = Series(reshape(df.iloc[:,2:2+seq_reps].values.T, [-1]))
            df = read_csv(file_, skiprows=544, nrows=seq_len, header=None)
            df4 = Series(reshape(df.iloc[:,2:2+seq_reps].values.T, [-1]))
            dfc = concat([df1, df2, df3, df4]).reset_index(drop=True)
            pfd = repeat(concat([led1, led1+led2, led1+led3, led1+led2+led3]).reset_index(drop=True), seq_reps).values
            df = DataFrame([fln, dfc, datetime, seq_time, pfd, led_seq]).T
            df.columns = ['flashlet_number','flevel','datetime','seq_time','pfd','led_sequence']

    list_ = []
    if append:
            if Single_Acq:
                print('Unable to concatenate files from a single acquisition')
                pass
            else:
                list_.append(df)
                df = concat(list_, axis=0)
                df['seq'] = repeat(arange(0, (df.shape[0] / seq_len), 1), seq_len)
    
    if save_files:
        if res_path is None:
            print('WARNING: Files not saved. No output directory specified.')
        else:
            if Single_Acq:
                df['seq'] = Series(repeat(arange(1,(df.shape[0] / seq_len)+1, 1), seq_len)).values.astype(int)
                df = df[['flashlet_number','flevel','datetime','seq','seq_time','pfd','led_sequence']]
            else:
                df['seq'] = Series(repeat(arange(1,(df.shape[0] / seq_len)+1, 1), seq_len)).values.astype(int)
                df = df[['flashlet_number','flevel','datetime','seq','seq_time','pfd','led_sequence']]
            df.to_csv(res_path+str(name)+'.csv')
            
            if led_separate:
                i = df.led_sequence == 1
                dfi = df[i].reset_index(drop=True)
                if len(dfi) > 0:
                    if Single_Acq:
                        dfi.loc[:,'seq'] = Series(repeat(arange(1, seq_reps+1, 1), seq_len)).values.astype(int)
                        dfi = dfi[['flashlet_number','flevel','datetime','seq','seq_time','pfd','led_sequence']]
                    else:
                        dfi.loc[:,'seq'] = Series(repeat(arange(1, seq_reps+1, 1), seq_len)).values.astype(int)
                        dfi = dfi[['flashlet_number','flevel','datetime','seq','seq_time','pfd','led_sequence']]
                    dfi.to_csv(res_path+str(name)+'_L1.csv')

                i = df.led_sequence == 2
                dfi = df[i].reset_index(drop=True)
                if len(dfi) > 0:
                    if Single_Acq:
                        dfi.loc[:,'seq'] = Series(repeat(arange(1, seq_reps+1, 1), seq_len)).values.astype(int)
                        dfi = dfi[['flashlet_number','flevel','datetime','seq','seq_time','pfd','led_sequence']]
                    else:
                        dfi.loc[:,'seq'] = Series(repeat(arange(1, seq_reps+1, 1), seq_len)).values.astype(int)
                        dfi = dfi[['flashlet_number','flevel','datetime','seq','seq_time','pfd','led_sequence']]
                    dfi.to_csv(res_path+str(name)+'_L12.csv')

                i = df.led_sequence == 3
                dfi = df[i].reset_index(drop=True)
                if len(dfi) > 0:
                    if Single_Acq:
                        dfi.loc[:,'seq'] = Series(repeat(arange(1, seq_reps+1, 1), seq_len)).values.astype(int)
                        dfi = dfi[['flashlet_number','flevel','datetime','seq','seq_time','pfd','led_sequence']]
                    else:
                        dfi.loc[:,'seq'] = Series(repeat(arange(1, seq_reps+1, 1), seq_len)).values.astype(int)
                        dfi = dfi[['flashlet_number','flevel','datetime','seq','seq_time','pfd','led_sequence']]
                    dfi.to_csv(res_path+str(name)+'_L13.csv')

                i = df.led_sequence == 4
                dfi = df[i].reset_index(drop=True)
                if len(dfi) > 0:
                    if Single_Acq:
                        dfi.loc[:,'seq'] = Series(repeat(arange(1, seq_reps+1, 1), seq_len)).values.astype(int)
                        dfi = dfi[['flashlet_number','flevel','datetime','seq','seq_time','pfd','led_sequence']]
                    else:
                        dfi.loc[:,'seq'] = Series(repeat(arange(1, seq_reps+1, 1), seq_len)).values.astype(int)
                        dfi = dfi[['flashlet_number','flevel','datetime','seq','seq_time','pfd','led_sequence']]
                    dfi.to_csv(res_path+str(name)+'_L123.csv')

                i = df.led_sequence == 5
                dfi = df[i].reset_index(drop=True)
                if len(dfi) > 0:
                    if Single_Acq:
                        dfi.loc[:,'seq'] = Series(repeat(arange(1, seq_reps+1, 1), seq_len)).values.astype(int)
                        dfi = dfi[['flashlet_number','flevel','datetime','seq','seq_time','pfd','led_sequence']]
                    else:
                        dfi.loc[:,'seq'] = Series(repeat(arange(1, seq_reps+1, 1), seq_len)).values.astype(int)
                        dfi = dfi[['flashlet_number','flevel','datetime','seq','seq_time','pfd','led_sequence']]
                    dfi.to_csv(res_path+str(name)+'_L23.csv')

                i = df.led_sequence == 6
                dfi = df[i].reset_index(drop=True)
                if len(dfi) > 0:
                    if Single_Acq:
                        dfi.loc[:,'seq'] = Series(repeat(arange(1, seq_reps+1, 1), seq_len)).values.astype(int)
                        dfi = dfi[['flashlet_number','flevel','datetime','seq','seq_time','pfd','led_sequence']]
                    else:
                        dfi.loc[:,'seq'] = Series(repeat(arange(1, seq_reps+1, 1), seq_len)).values.astype(int)
                        dfi = dfi[['flashlet_number','flevel','datetime','seq','seq_time','pfd','led_sequence']]
                    dfi.to_csv(res_path+str(name)+'_L2.csv')

                i = df.led_sequence == 7
                dfi = df[i].reset_index(drop=True)
                if len(dfi) > 0:
                    if Single_Acq:
                        dfi.loc[:,'seq'] = Series(repeat(arange(1, seq_reps+1, 1), seq_len)).values.astype(int)
                        dfi = dfi[['flashlet_number','flevel','datetime','seq','seq_time','pfd','led_sequence']]
                    else:
                        dfi.loc[:,'seq'] = Series(repeat(arange(1, seq_reps+1, 1), seq_len)).values.astype(int)
                        dfi = dfi[['flashlet_number','flevel','datetime','seq','seq_time','pfd','led_sequence']]
                    dfi.to_csv(res_path+str(name)+'_L3.csv')
    
    return df

def load_LIFT_FRR_files(file_, append=False, save_files=False, res_path=None, seq_len=228):
    """

    Process the raw data file from the Soliense LIFT FRR and convert to a csv with standard formatting.

    Parameters
    ----------
    file_ : dir
        The path directory to the .000 data file from benchtop SAtlantic FIRe.
    append : bool, default=False
        If True, multiple files will be concatenated together.
    save_files : bool, default=False
        If True, files will be saved as .csv.
    res_path : dir               
        The path directory where to save files, only required if save_files = True.
    seq_len : int, default=228                 
        The number of flashlets in the protocol.

    Returns
    -------

    df : pandas.DataFrame, shape=[n,7]
        A dataframe of the raw fluorescence data with columns as below:
    flashlet_number : np.array, dtype=int, shape=[n,]
        A sequential number from 1 to ``seq_len``
    flevel : np.array, dtype=float, shape=[n,]
        The raw fluorescence data.
    datetime : np.array, dtype=datetime64, shape=[n,]
        The date and time of measurement.
    seq : np.array, dtype=int, shape=[n,]
        The sample measurement number.
    seq_time : np.array, dtype=float, shape=[n,]
        The measurement sequence time in μs.
    pfd : np.array, dype=float, shape=[n,]
        The photon flux density in μmol photons m\ :sup:`2` s\ :sup:`-1`.

    """
    name = file_.split('/')[-1].split('.')[0]
    df = read_csv(file_, header=0)
    md = df[df['----'].str.contains("DateTime:")]
    date = md.iloc[:,1]
    date = date.str.strip().values
    time = md.iloc[:,2]
    time = time.str.strip().values
    datetime = date+' '+time
    datetime = to_datetime(datetime, yearfirst=True)
    to_drop = ['DateTime:', 'PIF:','PARs:','Light:','Lamps:','Gain:','S/N Ratio:','Position:','Cycles:','DataPt','==========','----']
    df = df.iloc[:,:4]
    df = df[~df['----'].isin(to_drop)]
    to_drop = ['99']
    df = df[~df['Unnamed: 1'].isin(to_drop)]
    columns = ['flashlet_number','seq_time','ex','em']
    df.columns = columns
    df['flevel'] = df.em.astype(float) / df.ex.astype(float)
    seq_len = 228
    df['datetime'] = repeat(datetime.values, seq_len)
    df['pfd'] = df.ex.astype(float) * 1e-6
    df['seq'] = repeat(arange(1, (df.shape[0]/seq_len)+1, 1), seq_len).astype(int)
    df = df.drop(['ex','em'], axis=1)
    df = df[['flashlet_number','flevel','datetime','seq','seq_time','pfd']]
    df = df.reset_index(drop=True)

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

    return df
