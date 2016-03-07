# pong_recurrent_convolution_net.py
# Dave Gottlieb (dmg1@stanford.edu) 2016
#
# Uses the approach of Ballas et al. 2016 to attempt to learn 
# intertemporal dependencies of an evolving game state from 
# control inputs and screen outputs. 
#
# My current best-performing approach is limited to local dependencies
# (fixed window of frames), which is sufficient to reproduce Pong
# but would probably fail for even simple other games. 

from theano import *
import theano.tensor as T
import numpy as np
from my_layers import * 
import Model

class PongRCNN(Model.Model): 

    def __init__(self, seq_length=8,batch_size=128,height=32,width=32): 
        
        self.Q = T.tensor4('Q',dtype=config.floatX) # (seq,batch,H,W)
        self.P = T.tensor3('P',dtype=config.floatX) # (seq,batch,D=2)
        self.Y = T.tensor4('Y',dtype=config.floatX) # (seq,batch,H,W)

        Tt, N = self.Q.shape[0], self.Q.shape[1]
        self.batch_size = batch_size
        self.seq_length=seq_length

        self.alpha = T.scalar('alpha',dtype=config.floatX) # learning rate

        # reshape Q to (T,N,1,H,W) 
        self.Q_view = self.Q.reshape((Tt,N,1,height,width))
        
        self.LSTM_RCN1 = LSTM_RCNLayer(input_var=self.Q_view,sequence=seq_length,
        			n_input_channels=1, height=3,width=3,n_filters=8,
        			layerid='LSTM_RCN1')

        self.LSTM_RCN2 = LSTM_RCNLayer(input_var=self.LSTM_RCN1.output,
        			sequence=seq_length, layerid='LSTM_RCN2',
        			n_input_channels=8, height=3,width=3,n_filters=16)

        self.LSTM_RCN3 = LSTM_RCNLayer(input_var=self.LSTM_RCN2.output,
			sequence=seq_length, layerid='LSTM_RCN3',
			n_input_channels=16, height=3,width=3,n_filters=16)

        self.POOL = T.signal.pool.pool_2d(self.LSTM_RCN3.output,(2,2))

        Q_unroll = self.POOL.reshape((Tt,N,height*width*16/4),ndim=3)

        PandQ = T.concatenate([Q_unroll, 
                    self.P],
                    axis=2)

        self.FC1 = TemporalReluFC(input_var=PandQ,num_units=512,layerid='FC1',in_dim=(height*width*4/4 + 2))

        self.FC2 = TemporalFC(input_var=self.FC1.output,
                    num_units=height*width,
                    layerid='FC2',
                    in_dim=512)



        Y_pred = T.nnet.sigmoid(self.FC2.output.reshape(self.Q.shape))

        #self.loss = T.nnet.binary_crossentropy(Y_pred,self.Y).mean(dtype=config.floatX)
        #self.loss = (T.abs_(Y_pred - self.Y)).mean(dtype=config.floatX)
        #self.loss = ((Y_pred - self.Y) ** 2).mean(dtype=config.floatX) 
        self.loss = -(self.Y * T.log(Y_pred)*14 + (1-self.Y)* T.log(1-Y_pred)).mean(dtype=config.floatX)
        #self.compute_loss = function([self.Q,self.P,self.Y],outputs=self.loss)

        self.params = {}
        self.params.update(self.LSTM_RCN1.params)
        self.params.update(self.LSTM_RCN2.params)
        self.params.update(self.FC1.params)
        self.params.update(self.FC2.params)

        self.train_args = [self.Q,self.P,self.Y,self.alpha]
        self.predict_args = [self.Q,self.P]

        self.output = Y_pred

        super(PongRCNN,self).__init__()

        #self._validate = function([self.Q,self.P,self.Y],outputs=self.loss)
    

    def train(self, q, p, y,alpha=1e-2): 
        return self._train(q,p,y,alpha)

    def train_multibatch(self,q,p,y,alpha=1e-2,lr_decay=1.0,num_epochs=1):
        """
        Split data into minibatches (batch size is fixed at init). 
        If batch size doesn't divide data, remainder are not used. 

        Keep the best set of weights, defined as: lowest minibatch training loss.
        """
        best_params = self.get_weights()
        best_loss = np.inf

        history = None

        epochs = 0

        i = 0 
        N = self.batch_size
        sum_loss = 0
        frames_per_epoch = int(q.shape[1] / N)

        while (epochs < num_epochs): 
            if (i + N > q.shape[1]): 
                epochs += 1
                avg_loss = sum_loss / (1.0*i / N)
                print "Loss after %i epochs: %f" % (epochs, avg_loss)
                sum_loss = 0
                i = 0
                alpha = alpha * lr_decay
            q_batch = q[:,i:i+N]
            p_batch = p[:,i:i+N]
            y_batch = y[:,i:i+N]

            loss = self.train(q_batch,p_batch,y_batch,alpha)


            # put loss in history
            if history is None: 
                history = np.array([
                        [i*self.seq_length+epochs*frames_per_epoch,
                        loss]
                        ])
            else:
                row = np.array([
                        [i*self.seq_length+epochs*frames_per_epoch,
                        loss]
                        ])
                history = np.concatenate((history,row))


            sum_loss += loss
            if loss < best_loss:
                best_loss = loss
                best_params = self.get_weights()
            i = i + N



        self.restore_weights(best_params)

        return history


    def validate(self,q,p,y): 
    	pass
        # return self._validate(q,p,y)

    def predict(self, q,p): 
        return self._predict(q,p)

    def sample(self, seed, num=25): 
        pass

    def sym_grads(self,q,p,y): 
    	pass
        # return self._grad(q,p,y)

    def num_grads(self, q,p,y,eps=1e-6): 
    	pass
        # ngs = []
        # for p in self.params: 
        #     old_v = p.get_value()
        #     p.set_value(old_v - eps)
        #     fm = self.compute_loss(q,p,y)
        #     p.set_value(old_v + eps)
        #     fp = self.compute_loss(q,p,y)
        #     ng = (fp - fm) / 2 * eps
        #     p.set_value(old_v)
        #     ngs.append(ng)

        # return ng



