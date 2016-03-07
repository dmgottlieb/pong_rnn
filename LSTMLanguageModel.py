# LSTMLanguageModel.py
# Dave Gottlieb (dmg1@stanford.edu) 2016
#
# Want to see if this can "verify" my LSTM implementation by working.

from theano import *
import theano.tensor as T
import numpy as np
from my_layers import *
import h5py

class LSTMLanguageModel(object): 
    

    
    def __init__(self, seq_length=25,vocab_size=100): 
        
        self.X = T.tensor3('X',dtype=config.floatX) # (T,N,D), 
        self.Y = T.tensor3('Y',dtype=config.floatX) # (T,N,D)
        self.vocab_size = vocab_size

        Tt,N = self.X.shape[0], self.X.shape[1]

        self.alpha = T.scalar('alpha',dtype=config.floatX) # learning rate

        

        self.LSTM1 = LSTMLayer(input_var=self.X,num_units=512,
                    layerid='LSTM1', sequence=seq_length,
                    in_dim=vocab_size)

        self.LSTM2 = LSTMLayer(input_var=self.LSTM1.output,num_units=512,
                    layerid='LSTM2', sequence=seq_length,
                    in_dim=512)

        self.LSTM3 = LSTMLayer(input_var=self.LSTM2.output,num_units=512,
                    layerid='LSTM3', sequence=seq_length,
                    in_dim=512)


        #self.LSTM2 = LSTMLayer(input_var=self.LSTM.output,num_units=512,layerid='LSTM2',in_dim=(512))
        #self.LSTM3 = LSTMLayer(input_var=self.LSTM2.output,num_units=512,layerid='LSTM3',in_dim=(512))

        self.FC = TemporalFC(input_var=self.LSTM3.output,
                    num_units=vocab_size,
                    layerid='FC',
                    in_dim=512)



        #Y_pred = T.nnet.softmax(self.FC.output.reshape((Tt*N,D))).reshape((Tt,N,H,W))*14.0
        Y_pred = T.nnet.softmax(self.FC.output.reshape((Tt*N,vocab_size),ndim=2)).reshape(self.X.shape)

        # don't use *binary* cross-entropy
        self.loss = T.nnet.categorical_crossentropy(Y_pred,self.Y).mean(dtype=config.floatX)
        #self.loss = (T.abs_(Y_pred - self.Y)).mean(dtype=config.floatX)
        #self.loss = ((Y_pred - self.Y) ** 2).mean(dtype=config.floatX) 
        #self.loss = -(self.Y * T.log(Y_pred)*14 + (1-self.Y)* T.log(1-Y_pred)).mean(dtype=config.floatX)
        #self.compute_loss = function([self.Q,self.P,self.Y],outputs=self.loss)

        self.params = {}
        self.params.update(self.LSTM1.params)
        self.params.update(self.LSTM2.params)
        self.params.update(self.LSTM3.params)
        self.params.update(self.FC.params)

        self.grads = {}
        for k in self.params.keys():
            g = T.grad(cost=self.loss,wrt=self.params[k])
            self.grads[k] = g

        #self._grad = function([self.X,self.Y],outputs=self.grads)

        self.updates=[]
        i = shared(np.cast['float32'](0))
        #i = shared(np.zeros(1).astype(np.float32))
        i_t = i + 1.0
        self.updates.append((i,i_t))
        alpha_t = T.sqrt(1 - 0.999 ** i_t) / (1 - 0.9 ** i_t) * self.alpha

        for k in self.params.keys():
            p = self.params[k]
            g = self.grads[k]
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

        self._train = function([self.X,self.Y,self.alpha],
                    outputs=self.loss,
                    updates=self.updates)
    
        self._predict = function([self.X],outputs=self.FC.output[-1])

        #self.print_y_softmax = function([self.Q,self.P],Y_pred)

    def train(self, x, y,alpha=1e-2): 
        return self._train(x,y,alpha)

    def predict(self, x): 
        """
        Returns scores by vocab item (before softmax).
        """
        return self._predict(x)

    def sample(self, seed, num, char_to_ix,ix_to_char,temp=1.0): 
        """
        Sample num chars from learned distribution, starting from seed. 

        Works by iteratively predicting sequences of length up to num, 
        which optimizes for correctness but sacrifices performance at large
        num. 

        An alternative approach would limit sequence length to s
        and construct the sample using a sliding window of length
        s over num chars. 
        """
        out = ""

        row = np.zeros((self.vocab_size),dtype=np.float32)

        if not seed is None:
            ix = char_to_ix[seed]
            row[ix] = 1.

        seq = row.reshape(1,1,-1)

        for i in range(num):
            scores = self._predict(seq).squeeze() / temp
            nonnorm = np.exp(scores)
            denom = nonnorm.sum()
            prob = (nonnorm / denom).squeeze()
            c = np.random.choice(np.arange(self.vocab_size),p=prob)
            row = np.zeros((self.vocab_size),dtype=np.float32)
            row[c] = 1.
            row = row.reshape(1,1,-1)
            seq = np.concatenate((seq,row),axis=0)

        

        ixes = np.argmax(seq,axis=2)

        for j in ixes: 
            out = out + ix_to_char[int(j)]

        return out


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

    def restore_weights(self,weight_dict): 
        """
        Takes a dictionary of pre-trained weight values, 
        with keys corresponding to the parameter names. 
        """
        for k in weight_dict.keys(): 
            self.params[k].set_value(weight_dict[k])

    def save_weights(self, file, mode='HDF5'): 
        """ 
        Saves the dictionary of parameters into a HDF5, 
        at the highest level of hierarchy
        """
        f = h5py.File(file,'w')
        
        for k in self.params.keys():
            f[k] = self.params[k].get_value()

        f.close()

    def load_weights(self, file, mode='HDF5'): 
        """
        Loads dictionary of parameters from HDF5,
        at the highest level of hierarchy.

        No enforcement of data type. 
        """

        f = h5py.File(file,'r')

        for k in self.params.keys():
            self.params[k].set_value(np.array(f[k]))

        f.close()

    



