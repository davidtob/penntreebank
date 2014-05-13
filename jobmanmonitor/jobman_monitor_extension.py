"""
.. todo::

    WRITEME
"""
import numpy
np = numpy
from pylearn2.train_extensions import TrainExtension
import theano
import theano.tensor as T
from pylearn2.utils import serial


class JobmanMonitor(TrainExtension):
    def __init__( self, result_extractor ):
        self.__dict__.update(locals())
        self.train_obj = None
        self.channel = None
        self.state = None

    def set_train_obj( self, train_obj ):
        self.train_obj = train_obj

    def set_jobman_channel( self, channel ):
        self.channel = channel

    def set_jobman_state( self,state ):
        self.state = state

    def on_monitor(self, model, dataset, algorithm):
        if train_obj!=None and self.state!=None:
          state.results = jobman.tools.resolve(state.extract_results)(train_obj)
          if self.channel!=None:
             channel.save()
