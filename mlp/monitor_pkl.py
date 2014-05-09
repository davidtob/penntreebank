import sys
import numpy
import cPickle
import pickle
import LM_dataset
import theano
import time
import os

#import pylab as plt
import matplotlib.pyplot as plt

def one_hot( x ):
    ret = numpy.zeros( (x.shape[0], x.shape[1]*50) )
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            ret[i,j*50 + x[i,j]] = 1
    return ret

def training_graph( train_bpc, valid_bpc ):
    plt.title('bits per character')
    plt.plot(train_bpc, color='b', label="Train")
    plt.plot(valid_bpc, color='g', label="Valid")
    plt.plot( train_bpc*0 + 4.321, color='r', label="One char. dist. bpc" )
    plt.plot( train_bpc*0 + 3.36,  color='y', label="Two char. cond dist. bpc" )    
    plt.draw()
    plt.show()
    
def generate_sentence( mlp ):
    inputs = mlp.get_input_space().get_total_dimension()/50

    X = mlp.get_input_space().make_batch_theano()
    y = mlp.fprop(X)
    pred_next_char = theano.function( [X], y )

    # Sample example sentence
    dictionary = numpy.load( "/nobackup_home/belius/Dropbox/pylearn2_data/PennTreebankCorpus/dictionaries.npz" )['unique_chars']    
    valid = LM_dataset.LMIterator( 1, seq_len=200, mode="valid", chunks="chars", path="/nobackup_home/belius/Dropbox/pylearn2_data/PennTreebankCorpus/penntree_char_and_word.npz" )
    valid.next()
    init = (valid.next()[0])[-inputs:]
    print "init", "".join(dictionary[init])
    # sample
    sampled = list(init)
    for i in range(20):
        one_hot_inp = one_hot( numpy.array( sampled[-inputs:] ).reshape( (1,inputs) ) )
        dist = pred_next_char( one_hot_inp )
        new_char = numpy.digitize(numpy.random.uniform( 0,1,1), numpy.cumsum(dist))[0]
        sampled.append(new_char)
    print "sample", "".join(dictionary[sampled])


plt.ion()
pklfn = sys.argv[1]
while True:
    # Load MLP
    print "loading"
    load_time = os.stat(pklfn).st_mtime
    try:
        mlp = pickle.load(open(pklfn))
    except:
        pass
    else:
        # Load training history
        valid_bpc = mlp.monitor.channels['valid_softmax_nll'].val_record/numpy.log(2)
        train_bpc = mlp.monitor.channels['train_softmax_nll'].val_record/numpy.log(2)

        # Best valid bpc
        print "best valid bpc", numpy.min(valid_bpc)

        # Make graph
        training_graph( train_bpc, valid_bpc )
        
        # sentence
        generate_sentence( mlp )

    curr_time = load_time
    while curr_time==load_time:
        try:
            curr_time = os.stat(pklfn).st_mtime
        except:
            pass
        plt.pause( 0.1 ) 
