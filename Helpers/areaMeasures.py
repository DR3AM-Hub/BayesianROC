# areaMeasures.py
#
# Copyright 2020 André Carrington, Ottawa Hospital Research Institute
# Use is subject to the Apache 2.0 License
# Written by André Carrington
#
import Helpers.bayesianAUC      as ba
from   sklearn.metrics  import roc_curve, auc

def roc_data_and_area_measures(y_true, y_pred_proba, optimal_score_pt, neg, pos,
                               partial_area_settings, measures, costs):
    # do_pArea   = partial_area_settings['do_pArea']
    # area_type  = partial_area_settings['area_type']
    # area_range = partial_area_settings['area_range']
    
    fpr, tpr, thr = roc_curve(y_true, y_pred_proba)
    roc_data      = (fpr, tpr, thr)
    
    if ('bayesian_auc' or 'my_auc') in measures:
        
        AUCpi, pAUCpi, myAUC = ba.bayesian_auc(fpr, tpr, neg, pos, partial_area_settings, costs)

        if 'my_auc' in measures:
            measures['bayesian_auc'] = AUCpi 
        #endif
        if 'my_auc' in measures:
            measures['bayesian_pauc']= pAUCpi 
        #endif
        if 'my_auc' in measures:
            measures['my_auc']       = myAUC
        #endif        
    #endif
    
    if 'sklearn_auc' in measures:    
        measures['sklearn_auc']      = auc(fpr, tpr)
    #endif
    
    return roc_data, measures
#enddef
