% Milestone: A Recurrent Neural Game Engine
% Dave Gottlieb
% 20 Feb. 2016

---
abstract: |
	Recent work has demonstrated successes in learning probabilistic generative models with deep neural networks. 
	A game engine can be characterized as a transition graph from previous states of a game, and player inputs, to future states of the game. 
	Accordingly, a game engine is a probabilistic generative model.
	Using a simple implementation of the classic game Pong, I attempt to train a neural network model to learn the transitions of the Pong game engine. 
	The model is trained on previous screen output and present control inputs of the game engine, and produces the next screen output.

email: dmg1@stanford.edu
address: |
	Stanford University  
	Department of Philosophy

bibliography: pong.bib
...

# Introduction

I specify a model to learn the state transitions of a simple game engine end-to-end, using only screen history and player input data. 
The model has no access to the internal states of the engine. 
Internal states must be learned from screen and input. 
For this purpose, I've created a simple implementation of the Pong game engine, which is able to run fast and headlessly to generate arbitrary amounts of training data for the model. 

Although the particular application of learning to recreate a game engine may be novel, the project fits well within recent and continuing work on learning probabilistic generative models with deep neural networks. 
For example, the variational autoencoder (VAE) model introduced by @vae provides a framework for learning probabilistic generative models with deep networks. 
The DRAW network of @draw learns a recurrent generative model to produce images, in that case handwritten digits. 
The deep Q learning framework demonstrated in @atari learns to *play* games, i.e., to manipulate probabilities in a generative model, using only screen and action histories, although that model is trained to produce an optimal stream of actions rather than to predict the evolution of the screen state. 


# Problem Statement

The behavior of the Pong game engine can be characterized as a probabilistic generative model that gives the distribution of possible screen outputs, conditioned on the history of previous screen outputs and control inputs. 

My goal is to build a model that, given the history of previous screen outputs and control inputs, can correctly predict the next screen output.
The model has access only to the engine's inputs and outputs: controller presses and screen contents. 
It has no direct access to the game state. 

Furthermore, the model must learn to produce screen outputs itself, by training end-to-end from controller inputs and the screen history. 
It would be easier, but less interesting, for the model to learn a small number of parameters of the game state, like the positions and velocities of the balls and paddles, and then produce the screen from those values using hand-crafted rules. 
My hope is to avoid this solution in favor of a true end-to-end approach. 
A model that is initially agnostic about how game states are mapped to the screen will more easily generalize to learning other game engines. 

I have described the project as a "neural game engine" because any model that successfully learns the transitions of the Pong game engine would then *be* a Pong game engine. 
The tuned model could then be tested by connecting it to live control inputs and screen outputs and "playing" it as though it was the original game. 

# Technical Approach

The structure of the problem has two main morals for technical approach: 

1. Since the next Pong frame depends on information from multiple previous frames as well as inputs, the model should be a recurrent model. 
2. Since the desired output is rich and structured rather than simple classification, an autoencoder approach might be helpful. 

Although Pong is a very simple game, its state still has dependencies that can go back several frames in the history. 
Most notably, both ball and paddles have velocity and inertia, so that at least two previous frames are needed to deduce the internal state of the engine at any time. 
Some kind of recurrency seems to be the natural way to capture these dependencies across time. 
Since the dependencies do not extend too far in time, though, I hope that they can be captured using a model with a short time window -- this could help avoid the vanishing and exploding gradient problems for RNN units without having to use more complicated recurrent models.
My preliminary results, reported below, are achieved using simple RNNs with a 9-frame time window.

Reliably generating a whole image from a learned distributed representation is a challenge. 
Here, I hope to apply the insights of the VAE literature.
VAEs have been used to sample novel images from a learned conditional distribution, essentially the same task I am working on. 
However, this part of my model is still in the early planning stage, and the preliminary results below don't reflect a VAE architecture. 

# Preliminary results

I have completed substantial work in both data generation and model prototyping. 
I've completed implementation of a Pong game engine, including AI, that is able to run headlessly and quickly generate arbitrary amounts of training data for the model. 
I've also specified, trained, and evaluated a first-pass prototype model that, while comparatively simple, successfully recovers some regularities of the game engine -- notably, the dependency between control input and paddle movement direction. 

## Game engine and data 

I've implemented in `numpy` a simple version of the Pong game engine. 
The screen output is $32\times32\times1$ bit of color. 
Each paddle has a stream of control inputs, which can take on values of $\{-1, 0, 1\}$ for "down," "no input," and "up" respectively. 

To generate training data, each input stream is produced by a simple AI algorithm and the game engine is run for arbitrary time. 
The preliminary results reflect running the game about 500 times for about 99 frames each, for a total of 49,000 frames -- something like 1,000 points scored. 
Since the game engine and AI run extremely quickly, this training set can be scaled up with ease. 


## Prototype model

The prototype model is implemented in Keras, running on top of Theano. 

The architecture of the prototype model is detailed in Table \ref{arch}. 
The screen history inputs are passed through two convolutional layers before being combined with the control inputs at the RNN hidden layer. 
The RNN state is then propagated through two affine layers before a softmax loss. 

\begin{table}[htb]
\caption{Architecture of prototype model. \label{arch}}
\resizebox{\columnwidth}{!}{%
\begin{tabular}[c]{@{}ccc@{}}
Screen input ($32\times32\times1$) & & Control input ($2\times1$) \\
CONV $3\times3$, $F=8$ & & \\
RELU & & \\
CONV $3\times3$, $F=16$  & & \\
RELU & & \\
&  RNN (512 units $\times$ 9 time-steps) & \\
& FC (1024 units) & \\
& RELU & \\
& OUTPUT (1024 values) & \\
& SOFTMAX &
\end{tabular}
}
\end{table}


Although this model is the wrong model and I plan to devise something more appropriate, it has produced some promising preliminary results. 
After training and running on just 49,000 frames of input data, the model successfully captures some of the broad phenomena of the Pong game. 
Notably, 

1. The ball has some velocity, and 
2. The paddles are controlled by the inputs.

![Overlay of frame with predicted succeeding frame. Note that the ball position updates according to a model of its velocity. \label{ball}](ball-movement-overlay.png) 

Figure \ref{ball} shows a training frame overlaid with the next frame predicted by the model. 
The overlaid ball positions show that the model has extrapolated a legal ball movement in the game, in effect imputing a velocity of $[1,1]$ to the ball. 
Since the model never has direct access to game states including velocity information, this transition has been learned directly from the screen. 

Figure \ref{paddles} shows the results over several frames of repeatedly feeding the model either up- or down-button presses. 
Again, the results show the model has learned the correct direction of dependence -- up-button presses lead to a distribution of paddle positions near the top of the screen, while down-button presses lead to a distribution of paddle positions near the bottom of the screen. 

\begin{figure}[ht]
\centering
\begin{minipage}[b]{0.45\linewidth}
\includegraphics{paddles-down}
\end{minipage}
\quad
\begin{minipage}[b]{0.45\linewidth}
\includegraphics{paddles-up}
\end{minipage}
\caption{Averages over several succeeding frames. At left, the average after feeding the model repeated presses of the down button, showing paddles near the bottom of the screen. At right, the average after repeated presses of the up button, showing paddles near the top. \label{paddles}}
\end{figure}


# Next steps

The most important next step is to generate more data. 
I have so far trained on 49,000 frames, but the Pong process runs very quickly and it would be simple to scale up to 1,000,000 frames. 
I've planned several other improvements in the data pipeline as well: 

* each training frame should appear in $T$ different training sequences, where $T$ is the time window; 
* each sequence should be initialized with $h_0 =$ the last state of the prior sequence; 
* change paddle AI (e.g. add randomness) so that left and right paddles move differently, making it easier for the model to tell which control inputs move which paddle. 

Apart from data, the major remaining challenge is working out an autoencoder architecture for the model, to make it easier to generate screen outputs given previous states. 


# References

