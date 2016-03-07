"""
Model.py
Dave Gottlieb (dmg1@stanford.edu) 2016

Abstract base class for models. 
""" 

import theano.tensor as T
from theano import *
import numpy as np
import h5py


class Model(object):

    def __init__(self): 
        """
        this should allow super constructor to be called 
        after self.params, self.loss, self.train_args,
        self.predict_args and self.output have been assigned.

        Gradients, _train, and _predict are then constructed. 
        """

        self.grads = {}
        for k in self.params.keys():
            g = T.grad(cost=self.loss,wrt=self.params[k])
            self.grads[k] = g

        # ADAM update
        self.updates = self._adam_update(self.params,self.grads,self.alpha)

        self._train = function(self.train_args,
                    outputs=self.loss,
                    updates=self.updates)
    
        self._predict = function(self.predict_args,
                    outputs=self.output[-1])

        # compute loss on batch w/o training
        #self._compute_loss = function(self.train_args,
        #            outputs=self.loss)


    def train(self, *args): 
        return self._train(*args)


    def _adam_update(self,ps,gs,alpha): 
        updates = []
        i = shared(np.cast['float32'](0))
        i_t = i + 1.0
        updates.append((i,i_t))
        alpha_t = T.sqrt(1 - 0.999 ** i_t) / (1 - 0.9 ** i_t) * self.alpha

        for k in ps.keys():
            p = ps[k]
            g = gs[k]
            m = shared(p.get_value()*0.)
            v = shared(p.get_value()*0.)
            m_t = 0.9*m + 0.1*g
            v_t = 0.999*v + 0.001*g**2
            p_t = p - m_t*alpha_t / T.sqrt(v_t + 1e-7)
            updates.append((m,m_t))
            updates.append((v,v_t))
            updates.append((p,p_t))

        return updates



    def predict(self, x): 
        return self._predict(x)


    def sym_grads(self,q,p,y): 
        pass
        #return self._grad(q,p,y)

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

    def get_weights(self): 
        """ 
        Returns a dictionary of values stored in self.params. 
        """
        weights = {}
        for k in self.params.keys(): 
            weights[k] = self.params[k].get_value()

        return weights

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

    



