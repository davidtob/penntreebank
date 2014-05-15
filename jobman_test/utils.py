from jobman.tools import DD
import numpy

def results_extractor(train_obj):
    channels = train_obj.model.monitor.channels
    train_bpc = channels['train_softmax_nll'].val_record[-1]/numpy.log(2)
    valid_bpc = channels['valid_softmax_nll'].val_record[-1]/numpy.log(2)

    return DD(train_bpc=train_bpc, valid_bpc=valid_bpc)
