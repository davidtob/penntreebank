from jobman.tools import DD
import numpy
import json

def results_extractor(train_obj):
    channels = train_obj.model.monitor.channels
    train_bpc_history = list( channels['train_softmax_nll'].val_record/numpy.log(2) )
    valid_bpc_history = list( channels['valid_softmax_nll'].val_record/numpy.log(2) )

    print train_bpc_history
    print valid_bpc_history


    bpc_history =  ('graph', 'BPC', 'epochs', { 'train': train_bpc_history,
                                                'valid': valid_bpc_history } ) 
    latest_train_bpc = int(1000*bpc_history[3]['train'][-1])/1000.0
    latest_valid_bpc = int(1000*bpc_history[3]['valid'][-1])/1000.0
    best_valid_bpc = int(1000*numpy.min( bpc_history[3]['valid'] ))/1000.0
    best_valid_epoch = numpy.argmin( bpc_history[3]['valid'] )
    #print channels.keys()
    examples_seen = 0#train_obj.channels['examples_seen']
    epochs_seen = 0#channels['epochs_seen']
    total_seconds_last_epoch = int(10*channels['total_seconds_last_epoch'].val_record[-1])/10.0

    print bpc_history
    print total_seconds_last_epoch
    return DD(best_valid_bpc=best_valid_bpc,
              best_valid_epoch=best_valid_epoch,
              latest_train_bpc=latest_train_bpc,
              latest_valid_bpc=latest_valid_bpc, 
              bpc_history=json.dumps(bpc_history),
              examples_seen=examples_seen,
              epochs_seen=epochs_seen,
              total_seconds_last_epoch=total_seconds_last_epoch )
