# theano_pong_rnn.py
# Dave Gottlieb (dmg1@stanford.edu) 2016
#
# Pong RNN model reimplemented more correctly + flexibly directly in Theano

from theano import *
import theano.tensor as T
import numpy as np

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
        
        self.W = shared(self.he_init(F,C,H,W,n_in=(H*W*C)), name='W'+layerid)
        self.b = shared(np.zeros(F,dtype=np.float32), name='b'+layerid)
        self.params = [self.W,self.b]
        
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
        
        self.W = shared(self.he_init(H*W*C,F,C,H,W), name='W'+layerid)
        self.b = shared(np.zeros(F,dtype=np.float32), name='b'+layerid)
        self.params = [self.W,self.b]
        
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


    def __init__(self, input_var, num_units=128, layerid=None, sequence=8,
                        in_dim=16386): 

        X = input_var
        Tt,N = X.shape[0], X.shape[1]
        H = num_units
        D = in_dim
        self.H = H

        self.Wx = shared(self.glorot_init(4*H,D,4*H),name='Wx'+layerid)
        self.Wh = shared(self.glorot_init(4*H,H,4*H),name='Wh'+layerid)
        self.b = shared(np.zeros(4*H,dtype=np.float32),name='b'+layerid)
        self.h0 = shared(np.zeros(H,dtype=np.float32),name='h0'+layerid)
        self.params = [self.Wx, self.Wh,self.b,self.h0]

        [h,c], _ = scan(self.step,
                sequences=[X],
                outputs_info=[T.alloc(self.h0,X.shape[1],H),T.zeros((X.shape[1],H))]
                )

        # Output: 
        # =======
        #
        # T x N x H matrix of hidden state activations
        #
        # These can be transformed with learned weights afterward or w/e.
        self.output = h

        # Sample output
        seed = T.ivector('seed')
        
        # [h_pred,c_pred], _ = scan(self.step,)


class TemporalReluFC(Layer): 

    def __init__(self, input_var, num_units=512, layerid=None): 

        X = input_var
        Tt,N,D = X.shape[0], X.shape[1], X.shape[2]
        H = num_units

        self.W = shared(self.he_init((D,H)),name='W'+layerid)
        self.b = shared(np.zeros(H,dtype=np.float32),name='b'+layerid)
        self.params = [self.W,self.b]

        preact = T.dot(X,self.W) + self.b

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
        self.params = [self.W,self.b]



        self.output = T.dot(X,self.W) + self.b





        
