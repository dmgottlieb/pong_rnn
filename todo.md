% todo.md
% Dave Gottlieb (dmg1@stanford.edu)

Model changes to explore 
=====

1. Fewer layers in convolution layers
2. Add a convolution layer
3. Add a FC layer after convolutions


Data generation pipeline
=====

1. Put each frame into T different sequences
2. Add more randomness
3. Have fewer restarts (maybe 0)
4. Feed last previous frame into new sequence

Live prediction pipeline: 
=====

1. Feed model a live (1xTxD) stream instead of (1x1xD)
2. See if this improves test performance


AI updates
=====

1. Build screen-reading routine (estimate ball and paddle positions from pixel averages)
2. Build wrapper for AI that takes in screen and puts out moves
3. OR: separate screen-reading from AI for efficiency 

Model tuning pipeline
=====

1. Visualize CONV weights
2. Find a way to test recurrency