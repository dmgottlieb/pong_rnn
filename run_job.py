#from pong_recurrent_convolution_net import * 
import h5py
import data_generation
import sys
import subprocess
import numpy as np 
from pong_lookback_net import * 


#q,p,y = data_generation.generate_data_lookback(seq_length=4,num_seq=(2**18))

#f = h5py.File('data/lookback_large_data.hdf5','w')
#f['q']=q
#f['p']=p
#f['y']=y
#f.close()

subprocess.call('aws s3 sync data/ s3://pong-rnn-model-data',shell=True)

model = LookbackNet()


f = h5py.File('data/lookback_large_data.hdf5','r')

q = np.array(f['q'])
p = np.array(f['p'])
y = np.array(f['y'])

f.close()



alpha = 1e-1
lr_decay=0.99
num_epochs = 10

history = model.train_multibatch(q,p,y,alpha,lr_decay,num_epochs=num_epochs)

model.save_weights('weights/lookback-large.HDF5')
subprocess.call('aws s3 sync weights/ s3://model-checkpoints', shell=True)

f = h5py.File('history/lookback-large-hist.HDF5','w') 
f['hist'] = np.array(history,dtype=np.float32)
f.close()

subprocess.call('aws s3 sync history/ s3://model-history', shell=True)
