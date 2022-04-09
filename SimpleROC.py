#!/usr/bin/env python
# SimpleROC.py
# by André Carrington

class SimpleROC(object):
    
    # __init__    constructor
    # getAUC()
    # getC()
    # get()
    # plot()
    # set_scores_labels()
    # set_fpr_tpr()
    # __str__     to string
    
    # for attributes see the constructor 
    
    def __init__(self, predicted_scores=None, labels=None, poslabel=None, quiet=False):
        '''SimpleROC constructor. If predicted_scores and labels are
           empty then it returns an empty object.'''
        from Helpers.ROCFunctions import checkFixLabels
        from Helpers.ROCFunctions import C_statistic
        from sklearn import metrics

        if predicted_scores is not None and labels is not None:
            self.predicted_scores               = predicted_scores
            self.labels                         = labels
            self.poslabel                       = poslabel
            self.newlabels, self.newposlabel    = checkFixLabels(labels, poslabel)
            self.fpr, self.tpr, self.thresholds = metrics.roc_curve(self.newlabels,
                                                                    self.predicted_scores,
                                                                    pos_label=self.newposlabel)
            self.AUC                            = metrics.roc_auc_score(self.newlabels,
                                                                        self.predicted_scores)
            self.C                              = C_statistic(self.predicted_scores, self.newlabels)
        else:
            self.predicted_scores = None
            self.labels           = None
            self.poslabel         = None
            self.newlabels        = None
            self.newposlabel      = None
            self.fpr              = None
            self.tpr              = None
            self.thresholds       = None
            self.AUC              = None
            self.C                = None
        #endif
        self.optimalpoints        = None
    #enddef

    def getAUC(self):
        ''' Computes and returns the AUC or AUROC (a continuous measure)'''
        import sklearn.metrics as metrics

        if self.predicted_scores is None and self.newlabels is None:
            if self.fpr is not None and self.tpr is not None:
                self.AUC = metrics.auc(self.fpr, self.tpr)
            else:
                SystemError('Predicted scores and labels, or FPR and TPR, are required to ' +
                            'compute the AUC.')
            #endif
        else:  # self.predicted_scores and self.newlabels are populated
            self.AUC     = metrics.roc_auc_score(self.newlabels, self.predicted_scores)
        #endif
        return self.AUC
    #enddef

    def getC(self):
        ''' Computes and returns the C statistic (a discrete measure)'''
        from Helpers.ROCFunctions import C_statistic

        if self.predicted_scores is None or self.newlabels is None:
            SystemError('Actual labels and predicted scores are required to compute the C statistic.')
        else:
            self.C = C_statistic(self.predicted_scores, self.newlabels)
            return self.C
        #endif
    #enddef
    
    def get(self):
        '''get returns the arrays for predicted_scores, labels, fpr, tpr, thresholds.'''
        import sklearn.metrics as metrics

        if self.predicted_scores is not None and self.newlabels is not None:            
            if self.fpr is None and self.tpr is None:
                self.fpr, self.tpr, self.thresholds = metrics.roc_curve(
                                                        self.newlabels, 
                                                        self.predicted_scores, 
                                                        pos_label=self.newposlabel)
            #endif
        #endif

        if self.__class__.__name__ == 'SimpleROC':  # as opposed to a subclass
            msg = 'sklearn metrics.roc_curve sets the highest threshold ' + \
                  'to max+1, when it should/may be any threshold above max: (max, infinity].'
            print(f'Warning from get(): {msg}')
        #endif

        # note: we return newlabels (not labels) in the following
        return self.predicted_scores, self.newlabels, self.fpr, self.tpr, self.thresholds
    #enddef

    def plot(self, plotTitle, showThresholds=True, showOptimalROCpoints=True, costs=None,
             saveFileName=None, numShowThresh=30, showPlot=True, labelThresh=False, full_fpr_tpr=False):
        '''plot provides an ROC plot with full data (including a point for each tie), and
           optional labels for threshold percentiles or thresholds, and optional optimal ROC points.'''
        '''plotWholeROC plots the whole curve with thresholds labeled and the Metz optimal ROC point(s) indicated'''
        from   Helpers.ROCPlot      import plotROC
        from   Helpers.ROCPlot      import plotOpt
        from   Helpers.ROCFunctions import getSkew
        from   Helpers.ROCFunctions import optimal_ROC_point_indices
        import matplotlib.pyplot as plt
        import math

        if self.__class__.__name__ == 'SimpleROC':  # as opposed to a subclass
            msg = 'sklearn metrics.roc_curve sets the highest threshold ' + \
                  'to max+1, when it should/may be any threshold above max: (max, infinity].'
            print(f'Warning from plot(): {msg}')
        #endif

        if self.__class__.__name__ == 'SimpleROC' or not full_fpr_tpr:
            fpr         = self.fpr
            tpr         = self.tpr
            thresholds  = self.thresholds
            newlabels   = self.newlabels
            newposlabel = self.newposlabel
        else:
            fpr         = self.full_fpr
            tpr         = self.full_tpr
            thresholds  = self.full_thresholds
            newlabels   = self.full_newlabels
        #endif

        newposlabel = self.newposlabel  # [sic] there is no full version for this

        fig, ax     = plotROC(fpr, tpr, plotTitle, numShowThresh, thresholds, labelThresh)

        if showOptimalROCpoints:
            # get optimal points here...
            skew           = getSkew(newlabels, newposlabel, costs)
            optimalpoints  = optimal_ROC_point_indices(fpr, tpr, skew)
            fpr_opt        = fpr[optimalpoints]
            tpr_opt        = tpr[optimalpoints]
            thresholds_opt = thresholds[optimalpoints]

            # for plotOpt...
            if not math.isinf(thresholds[0]):
                maxThreshold = thresholds[0]  # if first (max) thresh is not infinite, then use it for label
            else:
                maxThreshold = thresholds[1]  # otherwise, use the next label which should be finite
            # endif

            plotOpt(fpr_opt, tpr_opt, thresholds_opt, maxThreshold, labelThresh)  # add the optimal ROC points
        # endif

        if showPlot:
            plt.show()
        #modeShort = mode[:-3]  # training -> train, testing -> test
        #fig.savefig(f'output/ROC_{modeShort}_{testNum}-{index}.png')

        if saveFileName is not None:
            fig.savefig(saveFileName)

        return fig, ax
    #enddef

    def set_scores_labels(self, predicted_scores=None, labels=None, poslabel=None):
        from Helpers.ROCFunctions import checkFixLabels
        from Helpers.ROCFunctions import C_statistic
        from sklearn import metrics

        if self.predicted_scores is not None and self.newlabels is not None:
            SystemError('predicted_scores and labels are already set.')

        if predicted_scores is None or labels is None:
            SystemError('predicted_scores or labels cannot be empty.')
        else:
            self.predicted_scores               = predicted_scores
            self.labels                         = labels
            self.newlabels, self.newposlabel    = checkFixLabels(labels, poslabel=poslabel)
            self.poslabel                       = poslabel
            self.newlabels, self.newposlabel    = checkFixLabels(labels, poslabel)
            self.fpr, self.tpr, self.thresholds = metrics.roc_curve(self.newlabels,
                                                                    self.predicted_scores,
                                                                    pos_label=self.newposlabel)
            self.AUC                            = metrics.roc_auc_score(self.newlabels,
                                                                        self.predicted_scores)
            self.C                              = C_statistic(self.predicted_scores, self.newlabels)
        #endif
    #enddef
    
    def set_fpr_tpr(self, fpr=None, tpr=None):
        '''The set_fpr_tpr method is allowed if the object is empty.'''
        from sklearn import metrics

        if fpr is None or tpr is None: 
            SystemError('fpr or tpr cannot be empty')
            
        if self.predicted_scores is not None or self.newlabels is not None:
            SystemError('Not allowed to set fpr and tpr ' +
                        'when predicted_scores and labels are already set.')
        
        self.fpr        = fpr 
        self.tpr        = tpr
        self.thresholds = None
        self.AUC        = metrics.auc(self.fpr, self.tpr)
    #enddef

    def __str__(self):
        '''This method prints the object as a string of its content re 
           predicted_scores, labels, fpr, tpr, thresholds.'''

        if self.__class__.__name__ == 'SimpleROC':  # as opposed to a subclass
            msg = 'sklearn metrics.roc_curve sets the highest threshold ' + \
                  'to max+1, when it should/may be any threshold above max: (max, infinity].'
            print(f'Warning from __str__(): {msg}')
        #endif
        rocdata = f'score, label\n'
        for a, b in zip(self.predicted_scores, self.labels):
            rocdata = rocdata + f'{a:0.3f}, {b:<5d}\n'
        #endfor
        rocdata = rocdata + f'\nfpr  , tpr  , thresh\n'
        for c, d, e in zip(self.fpr, self.tpr, self.thresholds):
            rocdata = rocdata + f'{c:0.3f}, {d:0.3f}, {e:0.3f}\n'
        #endfor
        return rocdata
    #enddef

#enddef
