diary.md
Dave Gottlieb (dmg1@stanford.edu) 2016

Progress diary for project. 

# Feb. 29 2016

After realizing that the recurrency was not working, and struggling fruitlessly with the Keras Extra layers to do the job, I set out to learn Theano so I could implement the needed functions directly. 

So yesterday and today I went through the Theano tutorial. 

I've just finished implementing a basic convolutional network in Theano. 
It works. 
(Reminder: weight initialization is REALLY important. Remember those He et al. and Glorot et al. papers.)

However, my implementation is missing all the cool convenience stuff provided by Keras and by the CS231N packages. 

I also haven't tried to implement recurrency yet. 

Still, this is very good progress for learning a new technology in one day. 

I like Theano! (Hopefully this will help with TensorFlow as well. I would like to use TensorFlow but right now the compatibility and performance headaches don't seem to be worth it.) 

Convenience stuff that needs to be implemented: 

* optimizers other than SGD
* don't throw an error when batch is not of expected size(!?)
* keep the best set of weights for when updates make them worse (need to figure out how this works in real implementations -- do you abandon updates that make loss worse? or just keep a running best? probably the latter)
* Run for multiple epochs(! lol)

# Feb. 28 2016

## Testing recurrency

Set out to prove that recurrent layer is actually incorporating past hidden states instead of devolving to degenerate behavior of T=1. 
Conclusion: some evidence that hidden states are being used during training, *but* they are not working properly in the Theano function I made to test it -- and they may not be working at all. 

Wrote Theano function that takes model inputs and outputs activations at the RNN layer -- wrapped in hidden_state_viz.py. 
Took a lot of debugging but apparently works. 

Used this function to compute and visualize hidden activations over a time series. 
It kind of looks like there is recurrency but it's not easy to be sure. 
I also created a spurious time series of unconnected frames, so that any consistency between activations over time would be due to recurrency. 
It kind of looks like there's some recurrency in this time series but I'm not sure and I don't know a good statistical test to check. 

In addition, I extracted the weights from the recurrent layer of the trained network. 
The recurrency weights $W_hh$ have $\sigma \approx 0.05$, and they were initialized to glorot uniform with $\sigma \approx 0.05$ *I think*. 
So this is not good evidence of updating during training. 
*But* the max weight is about 0.2 and the min -0.2, which shouldn't be possible in the given glorot uniform initialization *I think*, because the initialization is uniform and bounded by absolute values about 0.08 *I think*. 
So that seems like decent evidence that these weights were updated during training, which can't happen unless they are contributing some value to the output. 

The weights seem *small* but that is I think a general symptom of the model not training that well. 

Another test of recurrency would be to feed the model a few frames of zeros and see if any activation persists. I'll try that now! 
**DAMMIT** okay this test seems to clearly show that the hidden states are not being added in in my test. 
But the other results *seem* to show that they are being used during training. 

A final test: train a new model and see if the hidden weights get updated during training. 
*Ok this turned out to be a big pain and I didn't do it*. 

## Re-implement in pure Theano

(tk)


