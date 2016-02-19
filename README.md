% README.md
% Dave Gottlieb (dmg1@stanford.edu)

**Pong RNN**: a deep learning model to learn the state transitions of a Pong game engine. 

A **work in progress**(!!). 
Current results are borderline at best (the model knows inputs control the paddles, but it doesn't know which inputs control which paddles,likely a data problem; it also seems to grossly overestimate how fast the controls move the paddles, and it either loses track of the ball or thinks the paddles move it or both).

Using: 

* [Keras](http://keras.io) (model specification),
* Theano (model specification and GPU computation), 
* numpy (data handling and Pong engine),
* matplotlib (visualization),
* [images2gif.py](https://grass.osgeo.org/grass70/manuals/libpython/_modules/imaging/images2gif.html) (visualization),
* pandas (data handling). 