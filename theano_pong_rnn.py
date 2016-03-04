# theano_pong_rnn.py
# Dave Gottlieb (dmg1@stanford.edu) 2016
#
# Pong RNN model reimplemented more correctly + flexibly directly in Theano

from theano import *
import theano.tensor as T
import numpy as np
from my_layers import *

class PongRNNModel(object): 
    

    
    def __init__(self, Tt, N, H, W): 
        
        self.Q = T.tensor4('Q',dtype=config.floatX) # (T,N,H,W), will reshape to (T,N,1,H,W) for convolution
        self.P = T.tensor3('P',dtype=config.floatX) # (T,N,D=2)
        self.Y = T.tensor4('Y',dtype=config.floatX) # (T,N,H,W)

        self.alpha = T.scalar('alpha',dtype=config.floatX) # learning rate

        self.Q = self.Q.reshape((Tt,N,1,H,W))
        
        self.CONV1 = TemporalConvReluLayer(input_var=self.Q,layerid='CONV1')
        self.CONV2 = TemporalConvReluLayer(input_var=self.CONV1.output,
            n_input_channels=8, n_filters=16,
            layerid='CONV2')

        self.POOL = T.signal.pool.pool_2d(self.CONV2.output,(2,2))
        PandQ = T.concatenate([self.POOL.reshape((Tt,N,4*H*W)), 
                    self.P],
                    axis=2)

        self.LSTM = LSTMLayer(input_var=PandQ,num_units=512,
                    layerid='LSTM',
                    in_dim=(32*32*16/4+2))

        self.LSTM2 = LSTMLayer(input_var=self.LSTM.output,num_units=512,layerid='LSTM2',in_dim=(512))
        self.LSTM3 = LSTMLayer(input_var=self.LSTM2.output,num_units=512,layerid='LSTM3',in_dim=(512))

        self.FC = TemporalFC(input_var=self.LSTM3.output,
                    num_units=H*W,
                    layerid='FC',
                    in_dim=512)



        #Y_pred = T.nnet.softmax(self.FC.output.reshape((Tt*N,H*W))).reshape((Tt,N,H,W))*14.0
        Y_pred = T.nnet.sigmoid(self.FC.output.reshape((Tt,N,H,W)))

        #self.loss = T.nnet.binary_crossentropy(Y_pred,self.Y).mean(dtype=config.floatX)
        #self.loss = (T.abs_(Y_pred - self.Y)).mean(dtype=config.floatX)
        #self.loss = ((Y_pred - self.Y) ** 2).mean(dtype=config.floatX) 
        self.loss = -(self.Y * T.log(Y_pred)*14 + (1-self.Y)* T.log(1-Y_pred)).mean(dtype=config.floatX)
        #self.compute_loss = function([self.Q,self.P,self.Y],outputs=self.loss)

        self.params = (self.CONV1.params + 
                    self.CONV2.params + 
                    self.LSTM.params + 
                    self.FC.params)

        self.grads = T.grad(cost=self.loss,wrt=self.params)

        self._grad = function([self.Q,self.P,self.Y],outputs=self.grads)

        self.updates=[]
        i = shared(np.cast['float32'](0))
        #i = shared(np.zeros(1).astype(np.float32))
        i_t = i + 1.0
        self.updates.append((i,i_t))
        alpha_t = T.sqrt(1 - 0.999 ** i_t) / (1 - 0.9 ** i_t) * self.alpha

        for p, g in zip(self.params,self.grads):
            m = shared(p.get_value()*0.)
            v = shared(p.get_value()*0.)
            m_t = 0.9*m + 0.1*g
            v_t = 0.999*v + 0.001*g**2
            p_t = p - m_t*alpha_t / T.sqrt(v_t + 1e-7)
            self.updates.append((m,m_t))
            self.updates.append((v,v_t))
            self.updates.append((p,p_t))



        #self.updates = [(self.params[i], self.params[i] - self.alpha * self.grads[i])
        #        for i in range(len(self.params))]

        self._train = function([self.Q,self.P,self.Y,self.alpha],
                    outputs=self.loss,
                    updates=self.updates)
    
        self._predict = function([self.Q,self.P],outputs=Y_pred)

        #self.print_y_softmax = function([self.Q,self.P],Y_pred)

    def train(self, q, p, y,alpha=1e-2): 
        return self._train(q,p,y,alpha)

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

    



