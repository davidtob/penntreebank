import numpy
import LM_dataset
import sys

numpy.random.seed(0)

dictionary = numpy.load( "/nobackup_home/belius/Dropbox/pylearn2_data/PennTreebankCorpus/dictionaries.npz" )['unique_chars']    


for k in range(1,10):
    train = LM_dataset.LMIterator( 1, seq_len=k, mode="train", chunks="chars", path="/nobackup_home/belius/Dropbox/pylearn2_data/PennTreebankCorpus/penntree_char_and_word.npz" )
    train = LM_dataset.LMIterator( train.get_length(), seq_len=k, mode="train", chunks="chars", path="/nobackup_home/belius/Dropbox/pylearn2_data/PennTreebankCorpus/penntree_char_and_word.npz" )
    train = (train.next()[0]).transpose()
    
    counts = numpy.zeros( (50,)*k )
    
    print "counting train"
    for i in range(train.shape[0]):
        counts[tuple(train[i,:])] += 1
        if i%100000==0:
            print "%.2f"%(float(i)/train.shape[0]),
            sys.stdout.flush()
    print
    
    sums = numpy.sum(counts,k-1)
    sums = (sums==0) + sums
    conddist = counts/sums.reshape( (50,)*(k-1) + (1,) )
        
    num_valid = LM_dataset.LMIterator( 1, seq_len=k, mode="valid", chunks="chars", path="/nobackup_home/belius/Dropbox/pylearn2_data/PennTreebankCorpus/penntree_char_and_word.npz" ).get_length()
    valid = LM_dataset.LMIterator( num_valid, seq_len=k, mode="valid", chunks="chars", path="/nobackup_home/belius/Dropbox/pylearn2_data/PennTreebankCorpus/penntree_char_and_word.npz" )
    valid = (valid.next()[0]).transpose()
    res = []
    print "evaluation on valid"
    for i in range(valid.shape[0]):         
        res.append ( -numpy.log2( conddist[ tuple( valid[i,:] ) ] ) )
    print "cond on prev", k,"gives bpc",numpy.mean(res)

    valid = LM_dataset.LMIterator( 1, seq_len=200, mode="valid", chunks="chars", path="/nobackup_home/belius/Dropbox/pylearn2_data/PennTreebankCorpus/penntree_char_and_word.npz" )
    valid.next()
    if k==1:
        init = numpy.zeros( (0,) )
    else:
        init = (valid.next()[0])[-(k-1):]
    if k>1:
        print "".join(dictionary[init])
    # sample
    sampled = list(init)
    for i in range(20):
        if k==1:
            dist = conddist
        else:
            dist = conddist[ tuple(sampled[-(k-1):]), tuple(range(50)) ]
        u = numpy.random.uniform( 0,1,1)
        new_char = numpy.digitize(u, numpy.cumsum(dist))[0]
        sampled.append(new_char)
    print "sameple", "".join(dictionary[sampled])

