# my_layers.py
# Dave Gottlieb (dmg1@stanford.edu)
#
# Convenience classes and functions for Theano layers in a CNN / RNN. 

from theano import *
import theano.tensor as T
import numpy as np

class Layer(object): 

    def he_init(self, n_in=None, *args): 
        if n_in is None: 
            n_in = args[1] 
        scale = np.sqrt(2.0/n_in)
        return (np.random.randn(*args) * scale).astype(np.float32)
    
    def glorot_init(self, n_in=None, *args):
        if n_in is None: n_in = args[1]
        #print args
        scale = np.sqrt(1.0/n_in)
        return (np.random.randn(*args) * scale).astype(np.float32)
        
        
class ConvReluLayer(Layer): 
    """
    Minibatch convolution layer
    """
    
    def __init__(self, input_var, height=3,width=3,n_filters=8,
            n_input_channels=1, layerid=None):
        H,W,F,C = height, width, n_filters, n_input_channels
        X = input_var
        
        self.W = shared(self.he_init(H*W*F,F,C,H,W), name='W'+layerid)
        self.b = shared(np.zeros(F,dtype=np.float32), name='b'+layerid)
        self.params = {self.W.name: self.W,
                        self.b.name: self.b}

        CONV = T.nnet.conv2d(X,self.W,border_mode='half')
        BIAS = CONV + self.b.dimshuffle('x',0,'x','x')
        RELU = T.nnet.relu(BIAS)

        self.output = RELU

class TemporalConvReluLayer(Layer): 
    """
    Minibatch convolution layer that expects a 5D tensor, 
    (T,N,C,H,W), unrolls it to (T*N,C,H,W), convolves, and 
    then re-rolls. 

    For use in recurrent nets.
    """
    def __init__(self, input_var, height=3,width=3,n_filters=8,
            n_input_channels=1, layerid=None):
        H,W,F,C = height, width, n_filters, n_input_channels
        Tt, N = input_var.shape[0], input_var.shape[1]
        X = input_var.reshape((Tt * N, 
            input_var.shape[2], input_var.shape[3], input_var.shape[4]))
        
        # I think fan-in is H*W*F, but some previous results used
        # H*W*C -- if this stops working, change back to that. 
        self.W = shared(self.he_init(H*W*F,F,C,H,W), name='W'+layerid)
        self.b = shared(np.zeros(F,dtype=np.float32), name='b'+layerid)
        self.params = {self.W.name: self.W,
                        self.b.name: self.b}

        CONV = T.nnet.conv2d(X,self.W,border_mode='half')
        BIAS = CONV + self.b.dimshuffle('x',0,'x','x')
        RELU = T.nnet.relu(BIAS)

        reroll = RELU.reshape((Tt, N, F, 
            input_var.shape[3],input_var.shape[4]))

        self.output = reroll 

class LSTMLayer(Layer): 
    """
    Minibatch LSTM layer. 

    Input is (T,N,D). 
    """

    def step(self, X_t,h_tm1,c_tm1): 
        Wx,Wh,b,H = self.Wx,self.Wh,self.b,self.H
        a = T.dot(X_t,Wx) + T.dot(h_tm1,Wh) + b
        ai = a[:,0:H]
        af = a[:,H:2*H]
        ao = a[:,2*H:3*H]
        ag = a[:,3*H:4*H]
        i = T.nnet.sigmoid(ai)
        f = T.nnet.sigmoid(af)
        o = T.nnet.sigmoid(ao)
        g = T.tanh(ag) 

        c_t = (f * c_tm1 + i * g ).astype(config.floatX)
        h_t = o * T.tanh(c_t).astype(config.floatX)

        return h_t, c_t


    def __init__(self, input_var, num_units, layerid, sequence,
                        in_dim): 

        X = input_var
        Tt,N = X.shape[0], X.shape[1]
        H = num_units
        D = in_dim
        self.H = H

        self.Wx = shared(self.glorot_init(4*H,D,4*H),name='Wx'+layerid)
        self.Wh = shared(self.glorot_init(4*H,H,4*H),name='Wh'+layerid)
        self.b = shared(np.zeros(4*H,dtype=np.float32),name='b'+layerid)
        #self.h0 = shared(np.zeros(H,dtype=np.float32),name='h0'+layerid)
        #self.c0 = shared(np.zeros(H,dtype=np.float32),name='c0'+layerid)
        self.params = {
            self.Wx.name: self.Wx,
            self.Wh.name: self.Wh,
            self.b.name: self.b
            }#[self.Wx, self.Wh,self.b]#,self.h0,self.c0]

        [h,c], _ = scan(self.step,
                sequences=[X],
                outputs_info=[T.zeros((X.shape[1],H),dtype=config.floatX),T.zeros((X.shape[1],H),dtype=config.floatX)]
                )

        # Output: 
        # =======
        #
        # T x N x H matrix of hidden state activations
        #
        # These can be transformed with learned weights afterward or w/e.
        self.output = h

        # Sample output
        #seed = T.ivector('seed')
        
        # [h_pred,c_pred], _ = scan(self.step,)


class TemporalReluFC(Layer): 

    def __init__(self, input_var, num_units=512, layerid=None,
                in_dim=512): 

        X = input_var
        Tt,N,D = X.shape[0], X.shape[1], in_dim
        H = num_units

        self.W = shared(self.he_init(H,D,H),name='W'+layerid)
        self.b = shared(np.zeros(H,dtype=np.float32),name='b'+layerid)
        self.params = {self.W.name: self.W,
                        self.b.name: self.b}
                        
        X_unroll = X.reshape((Tt*N,D))

        preout = T.dot(X_unroll,self.W) + self.b
        preact = preout.reshape((Tt,N,H))

        self.output = T.nnet.relu(preact)

class TemporalFC(Layer): 

    def __init__(self, input_var, num_units=512, layerid=None,
                in_dim=512): 

        X = input_var
        Tt,N = X.shape[0], X.shape[1]
        D = in_dim
        H = num_units

        self.W = shared(self.glorot_init(H,D,H),name='W'+layerid)
        self.b = shared(np.zeros(H,dtype=np.float32),name='b'+layerid)
        self.params = {self.W.name: self.W,
                        self.b.name: self.b}

        X_unroll = X.reshape((Tt*N,D))

        preout = T.dot(X_unroll,self.W) + self.b

        self.output = preout.reshape((Tt,N,H))


class FC(Layer): 
    def __init__(self, input_var, num_units=512, layerid=None,
            in_dim=512): 

        X = input_var
        N = X.shape[0]
        D = in_dim
        H = num_units

        self.W = shared(self.glorot_init(H,D,H),name='W'+layerid)
        self.b = shared(np.zeros(H,dtype=np.float32),name='b'+layerid)
        self.params = {self.W.name: self.W,
                        self.b.name: self.b}



        self.output = T.dot(X,self.W) + self.b

class FCRelu(Layer): 
    def __init__(self, input_var, num_units=512, layerid=None,
            in_dim=512): 

        X = input_var
        N = X.shape[0]
        D = in_dim
        H = num_units

        self.W = shared(self.he_init(H,D,H),name='W'+layerid)
        self.b = shared(np.zeros(H,dtype=np.float32),name='b'+layerid)
        self.params = {self.W.name: self.W,
                        self.b.name: self.b}


        self.output = T.nnet.relu(T.dot(X,self.W) + self.b)

class LSTM_RCNLayer(Layer): 
    """
    Implements a minibatch recurrent convolutional layer, 
    similar to the GRU-RCN described in Ballas et al. (2016).
    http://arxiv.org/pdf/1511.06432.pdf
    """

    def step(self, X_t,h_tm1,c_tm1): 
        Wx,Wh,b,F = self.Wx,self.Wh,self.b,self.n_filters
        # preactivation with dims (N, 4*F, H, W)
        a = (T.nnet.conv2d(X_t,self.Wx,border_mode='half') +
                T.nnet.conv2d(h_tm1,Wh,border_mode='half') + 
                b.dimshuffle('x',0,'x','x'))
        ai = a[:,0:F]
        af = a[:,F:2*F]
        ao = a[:,2*F:3*F]
        ag = a[:,3*F:4*F]
        i = T.nnet.sigmoid(ai)
        f = T.nnet.sigmoid(af)
        o = T.nnet.sigmoid(ao)
        g = T.tanh(ag) 

        c_t = (f * c_tm1 + i * g ).astype(config.floatX)
        h_t = o * T.tanh(c_t).astype(config.floatX)

        return h_t, c_t

    def __init__(self, input_var, 
                layerid,sequence,n_input_channels=1,
                height=3,width=3,n_filters=8):
        
        X = input_var
        imH, imW = X.shape[-2],X.shape[-1]

        H, W, F, C = height, width, n_filters, n_input_channels
        Tt, N = input_var.shape[0],input_var.shape[1]
        self.n_filters = n_filters


        self.Wx = shared(self.glorot_init(H*W*F,4*F,C,H,W), name='Wx'+layerid)
        self.Wh = shared(self.glorot_init(4*H*W*F,4*F,F,H,W),name='Wh'+layerid)
        self.b = shared(np.zeros(4*F,dtype=np.float32),name='b'+layerid)

        self.params = {
            self.Wx.name: self.Wx,
            self.Wh.name: self.Wh,
            self.b.name: self.b
            }

        [h,c], _ = scan(self.step,
            sequences=[X],
            outputs_info=[
                        T.alloc(np.cast['float32'](0), N,F,imH,imW),
                        T.alloc(np.cast['float32'](0), N,F,imH,imW)
                        ])

        self.output = h





        
