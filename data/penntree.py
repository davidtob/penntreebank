__authors__ = "Bart van Merrienboer"
__copyright__ = "Copyright 2010-2014, Universite de Montreal"
__license__ = "3-clause BSD"

import numpy
from numpy.lib.stride_tricks import as_strided
#from pylearn2.datasets.dense_design_matrix import DenseDesignMatrix
from dense_design_matrix import DenseDesignMatrix
from pylearn2.utils import serial
from pylearn2.utils.iteration import resolve_iterator_class


class PennTreebank(DenseDesignMatrix):
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
        else:
            raise ValueError("Dataset must be one of 'train', 'valid' "
                             "or 'test'")
        del npz_data  # Free up some memory?
        
        self._raw_data = self.one_hot( self._raw_data )

        self._data = as_strided(self._raw_data,
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
