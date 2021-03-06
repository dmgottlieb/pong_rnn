# theano_pong_rnn.py
# Dave Gottlieb (dmg1@stanford.edu) 2016
#
# Pong RNN model reimplemented more correctly + flexibly directly in Theano

from theano import *
import theano.tensor as T
import numpy as np
from my_layers import *
import Model

class PongRNNModel(Model.Model): 
    

    
    def __init__(self, Tt, N, H, W): 

        self.batch_size=N
        self.seq_length=Tt
        
        self.Q = T.tensor4('Q',dtype=config.floatX) # (T,N,H,W), will reshape to (T,N,1,H,W) for convolution
        self.P = T.tensor3('P',dtype=config.floatX) # (T,N,D=2)
        self.Y = T.tensor4('Y',dtype=config.floatX) # (T,N,H,W)

        self.alpha = T.scalar('alpha',dtype=config.floatX) # learning rate

        self.Q_view = self.Q.reshape((Tt,N,1,H,W))
        
        self.CONV1 = TemporalConvReluLayer(input_var=self.Q_view,layerid='CONV1')
        self.CONV2 = TemporalConvReluLayer(input_var=self.CONV1.output,
            n_input_channels=8, n_filters=16,
            layerid='CONV2')

        self.POOL = T.signal.pool.pool_2d(self.CONV2.output,(2,2))
        PandQ = T.concatenate([self.POOL.reshape((Tt,N,4*H*W)), 
                    self.P],
                    axis=2)

        self.LSTM = LSTMLayer(input_var=PandQ,num_units=512,
                    layerid='LSTM', sequence=Tt,
                    in_dim=(32*32*16/4+2))

        #self.LSTM2 = LSTMLayer(input_var=self.LSTM.output,num_units=512,layerid='LSTM2',in_dim=(512))
        #self.LSTM3 = LSTMLayer(input_var=self.LSTM2.output,num_units=512,layerid='LSTM3',in_dim=(512))

        self.FC = TemporalFC(input_var=self.LSTM.output,
                    num_units=H*W,
                    layerid='FC',
                    in_dim=512)



        #Y_pred = T.nnet.softmax(self.FC.output.reshape((Tt*N,H*W))).reshape((Tt,N,H,W))*14.0
        Y_pred = T.nnet.sigmoid(self.FC.output.reshape(self.Q.shape))

        self.output = Y_pred

        #self.loss = T.nnet.binary_crossentropy(Y_pred,self.Y).mean(dtype=config.floatX)
        #self.loss = (T.abs_(Y_pred - self.Y)).mean(dtype=config.floatX)
        #self.loss = ((Y_pred - self.Y) ** 2).mean(dtype=config.floatX) 
        self.loss = -(self.Y * T.log(Y_pred)*14 + (1-self.Y)* T.log(1-Y_pred)).mean(dtype=config.floatX)
        #self.compute_loss = function([self.Q,self.P,self.Y],outputs=self.loss)

        self.params = {}
        self.params.update(self.CONV1.params)
        self.params.update(self.CONV2.params)
        self.params.update(self.LSTM.params)
        self.params.update(self.FC.params)

        self.train_args = [self.Q,self.P,self.Y,self.alpha]
        self.predict_args = [self.Q,self.P]

        # super constructor creates gradients, _train, and _predict
        super(PongRNNModel,self).__init__()



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



    def predict(self, q,p): 
        return self._predict(q,p)

    def sample(self, seed, num=25): 
        pass

    def sym_grads(self,q,p,y): 
        return self._grad(q,p,y)

    def num_grads(self, q,p,y,eps=1e-6): 
        ngs = []
        for p in self.params: 
            old_v = p.get_value()
            p.set_value(old_v - eps)
            fm = self.compute_loss(q,p,y)
            p.set_value(old_v + eps)
            fp = self.compute_loss(q,p,y)
            ng = (fp - fm) / 2 * eps
            p.set_value(old_v)
            ngs.append(ng)

        return ng

    



