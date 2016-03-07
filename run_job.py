from pong_recurrent_convolution_net import * 
import h5py
import data_generation
import sys
import subprocess

model = PongRCNN(batch_size=64)

q,p,y = data_generation.generate_data_RNN(num_seq=(2**17))



alpha = 1e-1
lr_decay=0.99
num_epochs = 1

history = model.train_multibatch(q,p,y,alpha,lr_decay,num_epochs=10)

model.save_weights('weights/rcn-large.HDF5')
subprocess.call('aws sync weights/ s3://model-checkpoints', shell=True)

f = h5py.File('history/rcn-large-hist.HDF5') 
f['hist'] = history
f.close()

subprocess.call('aws sync history/ s3://model-history', shell=True)
