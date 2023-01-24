import your
from your import Your, Writer
import numpy as np
from your.formats.filwriter import make_sigproc_object
import time

datafile = 'HitCrabX_2021339160523-160533_8us_f.txt' #input data
newfile = 'crab_HIT_8G_8us_f_10s.fil' #output data

ch_num = 64

start = time.time()
sigproc_object = make_sigproc_object(
    rawdatafile  = str(newfile),
    source_name = "Crab", #target object
    nchans  = int(ch_num),
    foff = -8.0, #MHz channel frequency width. 
    fch1 = 8704, # MHz, Max observed frequency
    tsamp = 8e-6, # seconds, sampling time
    tstart = 59553.6704629629, #MJD
    src_raj = 053431.973, # HHMMSS.SS
    src_dej = +220052.06, # DDMMSS.SS
    machine_id=0,
    nbeams=0,
    ibeam=0,
    nbits=32, #bit size of 1 data point
    nifs=1, #no need to change
    barycentric=0,
    pulsarcentric=0,
    telescope_id=6, #no need to change?
    data_type=0,
    az_start=-1,
    za_start=-1,
)

sigproc_object.write_header(newfile)

print('Read file: ' + datafile) 
data = np.loadtxt(datafile, usecols=[0], dtype='float32') #"dtype" depends on datafile bit, corresponds to "nbit". usecol is the index of power part of datafile
#print(data[0])
print(time.time() - start, 'sec,  Datafile loaded')

sampling_num = int(len(data)/ch_num)
#data_split = np.split(data, sampling_num)
data = data.reshape(sampling_num, ch_num) #(depends on datafile format)
sigproc_object.append_spectra(data, newfile)
your_filterbank = Your(newfile)

#data_read = your_filterbank.get_data(nstart=0, nsamp=sampling_num)
#print(np.array_equal(data_read, data))

elapsed_time = time.time() - start
print ("elapsed_time:{0}".format(elapsed_time) + "[sec]")
