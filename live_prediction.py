# live_prediction.py
# Dave Gottlieb 2016 (dmg1@stanford.edu) 
#
# Code to 

import numpy as np

def live_predict(model,seed,num_frames=30,window=4): 

	q = seed
	p = np.repeat(np.array([[[-1,-1]]]),q.shape[0],axis=0).astype(np.float32)

	for i in range(num_frames): 
		t = min(q.shape[0],window)
		q_pred = model.predict(q[0:t],p[0:t])
		q_next = q_pred[-1]
		q = np.concatenate((q,q_next.reshape((1,1,32,32))),axis=0) 

	return q
