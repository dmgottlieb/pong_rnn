# pong_lookback_net.py
# Dave Gottlieb (dmg1@stanford.edu) 2016
#
# Pursuing a different kind of recurrent architecture. 
# The network learns a function from a fixed time window of previous frames, 
# convolving over stacks of historical frames. 
#
# My previous approach (LSTM that gets transformations of a single frame as input)
# was not really working -- maybe because it was not using previous frames enough. 
# This might force it to learn from previous frames better. 
#

from my_layers import * 

class LookbackNet(Object): 

	def __init__(self, window=4,batch_size=128,height=32,width=32): 
        
        self.Q = T.tensor4('Q',dtype=config.floatX) # (batch,window,H,W)
        self.P = T.tensor3('P',dtype=config.floatX) # (batch,D=2)
        self.Y = T.tensor4('Y',dtype=config.floatX) # (batch,H,W)

        self.alpha = T.scalar('alpha',dtype=config.floatX) # learning rate
        
        self.CONV1 = ConvReluLayer(input_var=self.Q,
        	height=3,width=3,
        	n_filters=8,
        	n_input_channels=window,
        	layerid='CONV1')
        self.CONV2 = TemporalConvReluLayer(input_var=self.CONV1.output,
            n_input_channels=8, n_filters=16,
            height=3,width=3,
            layerid='CONV2')

        self.POOL = T.signal.pool.pool_2d(self.CONV2.output,(2,2))

        PandQ = T.concatenate([self.POOL.reshape((batch_size,height*width/4*16)), 
                    self.P],
                    axis=1)

        self.FC1 = FC(input_var=PandQ,num_units=512,layerid='FC1',in_dim=(height*width/4*16 + 2))

        self.FC2 = TemporalFC(input_var=self.FC1.output,
                    num_units=height*width,
                    layerid='FC2',
                    in_dim=512)



        Y_pred = T.nnet.sigmoid(self.FC2.output.reshape((N,H,W)))

        #self.loss = T.nnet.binary_crossentropy(Y_pred,self.Y).mean(dtype=config.floatX)
        #self.loss = (T.abs_(Y_pred - self.Y)).mean(dtype=config.floatX)
        #self.loss = ((Y_pred - self.Y) ** 2).mean(dtype=config.floatX) 
        self.loss = -(self.Y * T.log(Y_pred)*14 + (1-self.Y)* T.log(1-Y_pred)).mean(dtype=config.floatX)
        #self.compute_loss = function([self.Q,self.P,self.Y],outputs=self.loss)

        self.params = (self.CONV1.params + 
                    self.CONV2.params + 
                    self.FC1.params + 
                    self.FC2.params)

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

