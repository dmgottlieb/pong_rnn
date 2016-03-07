% todo.md
% Dave Gottlieb (dmg1@stanford.edu)

Priority items: 
=====

1. Write a routine to keep the best set of weights if an update makes them worse
2. Research initialization schemes for dropout
2. Implement dropout
3. Randomize inputs in training data
4. Build live prediction


Model changes to explore 
=====

1. Implement dropout or other regularization (if the model overfits)

Training Pipeline
=====

1. Write a routine to keep the best set of weights if an update makes them worse


Data generation pipeline
=====


1. Inputs should be randomized to ensure that the model is not learning the AI. 
1. Some sequences should begin with empty frames so the model can learn how to start up.
2. Ensure that sequences where the ball goes off-screen can occur -- since my time window is only 4 frames, I need to make sure all interesting cases are within 4 frames of start. 
5. POSSIBLY: augment training data with noise so that it looks a bit more like the predictions will 

Live prediction pipeline: 
=====

1. Build live prediction / feedback code
2. Test it against fast-playing AIs and visually inspect: how long does it take for entropy to overtake game? 

AI updates
=====

1. Build screen-reading routine (estimate ball and paddle positions from pixel averages)
2. Build wrapper for AI that takes in screen and puts out moves
	a. OR: separate screen-reading from AI for efficiency 
3. Build smarter AI (e.g., model ball momentum and extrapolate its intersection with your goal line)

Model tuning pipeline
=====

1. Visualize weights
2. Visualize loss

Blue Sky possibilities
=====

1. Test the model on data from other games, possibly ones I didn't make -- this should be somewhat doable

