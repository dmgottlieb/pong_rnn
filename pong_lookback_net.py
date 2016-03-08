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

from theano import *
import theano.tensor as T
import numpy as np
from my_layers import * 

class LookbackNet(Model.Model): 

    def __init__(self, window=4,batch_size=128,height=32,width=32): 
        
        self.Q = T.tensor4('Q',dtype=config.floatX) # (batch,window,H,W)
        self.P = T.matrix('P',dtype=config.floatX) # (batch,D=2)
        self.Y = T.tensor3('Y',dtype=config.floatX) # (batch,H,W)

        self.alpha = T.scalar('alpha',dtype=config.floatX) # learning rate
        
        self.CONV1 = ConvReluLayer(input_var=self.Q,
            height=3,width=3,
            n_filters=8,
            n_input_channels=window,
            layerid='CONV1')
        self.CONV2 = ConvReluLayer(input_var=self.CONV1.output,
            n_input_channels=8, n_filters=16,
            height=3,width=3,
            layerid='CONV2')

        self.POOL = T.signal.pool.pool_2d(self.CONV2.output,(2,2))

        PandQ = T.concatenate([self.POOL.reshape((batch_size,height*width/4*16)), 
                    self.P],
                    axis=1)

        self.FC1 = FCRelu(input_var=PandQ,num_units=512,layerid='FC1',in_dim=(height*width/4*16 + 2))

        self.FC2 = FC(input_var=self.FC1.output,
                    num_units=height*width,
                    layerid='FC2',
                    in_dim=512)



        Y_pred = T.nnet.sigmoid(self.FC2.output.reshape((batch_size,height,width)))

        #self.loss = T.nnet.binary_crossentropy(Y_pred,self.Y).mean(dtype=config.floatX)
        #self.loss = (T.abs_(Y_pred - self.Y)).mean(dtype=config.floatX)
        #self.loss = ((Y_pred - self.Y) ** 2).mean(dtype=config.floatX) 
        self.loss = -(self.Y * T.log(Y_pred)*14 + (1-self.Y)* T.log(1-Y_pred)).mean(dtype=config.floatX)
        #self.compute_loss = function([self.Q,self.P,self.Y],outputs=self.loss)

        self.params = {}
        self.params.update(self.CONV1.params)
        self.params.update(self.CONV2.params)
        self.params.update(self.FC1.params)
        self.params.update(self.FC2.params)

        self.train_args = [self.Q,self.P,self.Y,self.alpha]
        self.predict_args = [self.Q,self.P]

        self.output = Y_pred

        super(LookbackNet, self).__init__()


    def train(self, q, p, y,alpha=1e-2): 
        return self._train(q,p,y,alpha)

    def train_multibatch(self,q,p,y,alpha=1e-2,lr_decay=1.0,batch_size=128,num_epochs=1):
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
        N = batch_size
        sum_loss = 0
        frames_per_epoch = int(q.shape[1] / N)
        stops = 0

        while (epochs < num_epochs): 
            if (i + N > q.shape[1]): 
                epochs += 1
                avg_loss = sum_loss / (1.0*i / N)
                print "\nLoss after %i epochs: %f" % (epochs, avg_loss)
                sum_loss = 0
                i = 0
                alpha = alpha * lr_decay
            q_batch = q[i:i+N]
            p_batch = p[i:i+N]
            y_batch = y[i:i+N]

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

            sys.stdout.write("\b"*stops)
            stops = int(35*i / (frames_per_epoch))
            sys.stdout.write("#"*stops)
            sys.stdout.flush()


            i = i + N



        self.restore_weights(best_params)

        return history


    def validate(self,q,p,y): 
        return self._validate(q,p,y)

    def predict(self, q,p): 
        return self._predict(q,p)

    def sample(self, seed, num=25): 
        pass

    def sym_grads(self,q,p,y): pass
        # return self._grad(q,p,y)

    def num_grads(self, q,p,y,eps=1e-6): pass
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

