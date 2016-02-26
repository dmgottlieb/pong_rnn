% todo.md
% Dave Gottlieb (dmg1@stanford.edu)

Model changes to explore 
=====

1. Fewer layers in convolution layers
2. Add a convolution layer
3. Add a FC layer after convolutions
4. BIG CHANGE: work out encoder-decoder architectures and try to apply one
4. BLUE SKY: use the 1-bit technique from Courbariaux & Bengio (2016) and add more layers (good fit because my data is ~1-bit already)
5. Implement dropout (read http://arxiv.org/pdf/1409.2329.pdf; apply dropout to output but not recurrent wires)
	a. See if Keras implementation works this way
6. Use LSTMs


Data generation pipeline
=====

1. Put each frame into T different sequences
2. Add more randomness
3. Have fewer restarts (maybe 0)
4. Feed last previous frame into new sequence
5. POSSIBLY: augment training data with noise so that it looks a bit more like the predictions will 

Live prediction pipeline: 
=====

1. Feed model a live (1xTxD) stream instead of (1x1xD)
2. See if this improves test performance


AI updates
=====

1. Build screen-reading routine (estimate ball and paddle positions from pixel averages)
2. Build wrapper for AI that takes in screen and puts out moves
	a. OR: separate screen-reading from AI for efficiency 
3. Build smarter AI (e.g., model ball momentum and extrapolate its intersection with your goal line)

Model tuning pipeline
=====

1. Visualize CONV weights
2. Find a way to test recurrency

Training pipeline
=====

1. Write routines to train on EC2 spot instances with checkpoints in case of instance loss. 
2. SPECULATIVE: curriculum learning? See (http://ronan.collobert.com/pub/matos/2009_curriculum_icml.pdf), (http://arxiv.org/pdf/1410.4615v3.pdf)
	-- Curriculum learning regime: many examples where only one thing is moving. E.g., ball moving and paddles still; only one paddle moving at a time; ball moving and hitting paddle(?). 
	-- Test whether simple relationships are being learned? 
	-- Introduce combinations of simple relationships and then fully general data. 
	-- The Zaremba / Sutskever combination curriculum learning strategy is: mix in curriculum-difficulty examples with examples of random difficulty. 
