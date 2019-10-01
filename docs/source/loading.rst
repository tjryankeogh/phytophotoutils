Loading
=======

load_FIRe_files
---------------

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

load_FASTTrackaI_files
----------------------

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

load_FastOcean_files
--------------------

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
