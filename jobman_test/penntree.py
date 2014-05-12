__authors__ = "Bart van Merrienboer"
__copyright__ = "Copyright 2010-2014, Universite de Montreal"
__license__ = "3-clause BSD"

import numpy
from numpy.lib.stride_tricks import as_strided
from pylearn2.datasets.dense_design_matrix import DenseDesignMatrix
#from dense_design_matrix import DenseDesignMatrix
from pylearn2.utils import serial
from pylearn2.utils.iteration import resolve_iterator_class


class PennTreebank(SparseDataset):
    """
    Loads the Penn Treebank corpus.
    """
    def __init__(self, which_set, chars_or_words, ngram_size, shuffle=True):
        """
        Parameters
        ----------
        which_set : {'train', 'valid', 'test'}
            Choose the set to use
        ngram_size : int
            The size of the n-grams
        shuffle : bool
            Whether to shuffle the samples or go through the dataset linearly
        """
        assert( chars_or_words=='chars' or chars_or_words=='words' )
        path = "${PYLEARN2_DATA_PATH}/PennTreebankCorpus/"
        path = serial.preprocess(path)
        npz_data = numpy.load(path + 'penntree_char_and_word.npz')
        if which_set == 'train':
            self._raw_data = npz_data['train_'+chars_or_words]
        elif which_set == 'valid':
            self._raw_data = npz_data['valid_'+chars_or_words]
        elif which_set == 'test':
            self._raw_data = npz_data['test_'+chars_or_words]
        elif which_set == 'train_train' or which_set == 'train_valid':
            self._raw_data = npz_data['train_'+chars_or_words]
            sentences = numpy.split( self._raw_data, numpy.where( self._raw_data==0 )[0] )
            import random
            random.seed(3)
            valid_idx = random.sample( range(len(sentences)), int(len(sentences)*0.5) )
            train_idx = range(len(sentences))
            for idx in valid_idx:
                train_idx.remove( idx )
            
            if which_set=='train_train':
                idcs = train_idx
            elif which_set=='train_valid':
                idcs = valid_idx
            else:
                assert False
            idcs.sort()
            print idcs
      
            keep_sentences = []
            for idx in idcs:
                keep_sentences.append( sentences[ idx ] )
            self._raw_data = numpy.hstack( keep_sentences )
        else:
            raise ValueError("Dataset must be one of 'train', 'valid' "
                             "or 'test'")
        del npz_data  # Free up some memory?
        
        self._sparse_data = self.one_hot( self._raw_data )

        self._data = as_strided(self._sparse_data,
                                shape=(len(self._raw_data)/50 - ngram_size + 1,
                                       ngram_size*50),
                                strides=(self._raw_data.itemsize*50,
                                         self._raw_data.itemsize))
        #n = len(self._raw_data)/50 - ngram_size
        #X = numpy.zeros( (n, ngram_size*50),dtype='int32')
        #y = numpy.zeros( (n, 50),dtype='int32')
        #for i in range( n ):
        #    X[i,:] = self._raw_data[ i*50: (i+ngram_size)*50 ]
        #    y[i,:] = self._raw_data[ (i+ngram_size)*50: (i+ngram_size+1)*50 ]

        if chars_or_words=='chars':
          num_labels = 50
        else:
          num_labels = 10000
        super(PennTreebank, self).__init__(
            #X=X,
            #y=y,
            X=self._data[:, :-50],
            y=self._data[:, -50:],
            #X_labels=50,
            #y_labels=50
            #X_labels=num_labels, y_labels=num_labels
        )

        if shuffle:
            self._iter_subset_class = \
                resolve_iterator_class('shuffled_sequential')
    

    def one_hot( self, x ):
        ret = numpy.zeros( (50*x.shape[0],), dtype='int' )
        for j in range(x.shape[0]):
                ret[j*50 + x[j]] = 1
        return ret
