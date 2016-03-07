# data_generation.py
# Dave Gottlieb 2016 (dmg1@stanford.edu)

from numpy_pong import * 


def generate_data_lookback(mode=None,seq_length=4,num_seq=10): 

    y_list = []
    q_list = []
    p_list = []


    for i in range(num_seq): 
        start_ball = (np.random.choice(np.arange(30)),
            np.random.choice(np.arange(24))+2)
        seq = headless_mode(num_frames=seq_length+1, 
                ballloc=start_ball)
        y = seq[seq_length,0:1024].reshape((32,32))
        q = seq[0:seq_length,0:1024].reshape((-1,32,32))
        p = seq[seq_length-1,1024:].reshape((2))

        y_list.append(y)
        p_list.append(p)
        q_list.append(q)

        if (i+1) % 10000 == 0: print "%i sequences generated" % (i+1)

    q_data = np.stack(q_list,axis=0)
    p_data = np.stack(p_list,axis=0)
    y_data = np.stack(y_list,axis=0)

    return q_data,p_data,y_data

def generate_data_RNN(mode=None,seq_length=8,num_seq=10): 
    """
    Returns num_seq sequences of simulated data (with y 
    offset by one frame), each of length seq_length. 

    (T,N,H,W)
    (T,N,D)
    (T,N,H,W)
    """

    y_list = []
    q_list = []
    p_list = []


    for i in range(num_seq): 
        start_ball = (np.random.choice(np.arange(30)),
            np.random.choice(np.arange(24))+2)
        seq = headless_mode(num_frames=seq_length+1, 
                ballloc=start_ball)
        y = seq[1:seq_length+1,0:1024].reshape((-1,1,32,32))
        q = seq[0:seq_length,0:1024].reshape((-1,1,32,32))
        p = seq[0:seq_length,1024:].reshape((-1,1,2))

        y_list.append(y)
        p_list.append(p)
        q_list.append(q)

        if (i+1) % 10000 == 0: print "%i sequences generated" % (i+1)

    q_data = np.concatenate(q_list,axis=1)
    p_data = np.concatenate(p_list,axis=1)
    y_data = np.concatenate(y_list,axis=1)

    return q_data,p_data,y_data