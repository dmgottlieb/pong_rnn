# do_job.py
# Dave Gottlieb 2016
#
# Describes training job to run on remote machine

from pong_rnn import PongRNN
from pandas import HDFStore, DataFrame
import numpy as np
from os import path

# gather data
q_list = []
p_list = []
y_list = []

data_path = path('~/pong/data/data.h5')

with HDFStore('/home/ubuntu/data/data.h5') as store: 
    for k in store.keys(): 
        d = np.array(store[k])[0:-1,:]
        
        assert d.shape == (100,1026)
        
        q = d[0:-1,0:1024].reshape((d.shape[0]-1, 32, 32))
        p = d[1:,1024:1026]
        y = d[1:,0:1024].reshape((d.shape[0]-1, 32, 32))
        
        assert ((q.shape[0] == p.shape[0]) and (q.shape[0] == y.shape[0]))
        assert q.shape == (99,32,32)
        assert p.shape == (99,2)
        assert y.shape == (99,32,32)
        
        q_list.append(q)
        p_list.append(p)
        y_list.append(y)

q_train = np.stack(q_list[0:-50])
p_train = np.stack((p_list[0:-50]))
y_train = np.stack(np.array(y_list[0:-50]))

y_train = y_train.reshape((-1,1024))
q_train = q_train.reshape((-1,1,32,32))
p_train = p_train.reshape((-1,2))


# train
model = PongRNN()
model.train(q_train, p_train, y_train, nb_epoch=1,checkpoints=True)


