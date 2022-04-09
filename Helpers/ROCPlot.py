#!/usr/bin/env python
# ROCPlot.py
# by André Carrington

# plotROC(fpr, tpr, plotTitle, numThresh, thresh, fancyLabel)
# plotSimpleROC(fpr,tpr,title):
# addPoints(fpr, tpr, numThresh, thresh, fancyLabel)
# get_ROC_Curve_Label_Offset_Fontsize(x, y, t, maxThresh, fancyLabel)
# plotOpt(fpr_opt, tpr_opt, thresh_opt, maxThresh, fancyLabel)

# imports are locally defined in each function

def plotROC(fpr, tpr, plotTitle, numThresh, thresh, fancyLabel):
    ''' returns fig, ax '''
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker

    fig = plt.figure()
    ax  = fig.add_subplot(1, 1, 1)
    # ax  = plt.add_subplot(1, 1, 1, xticks=[], yticks=[])

    plt.plot(fpr, tpr, color='blue', lw=2)

    # add threshold labels and circles
    # allow up to numThresh labels per plot, or numThresh+4 in the first multiple
    if 'thresh' in locals():
        if 'fancyLabel' not in locals():
            fancyLabel = False
        addPoints(fpr, tpr, numThresh, thresh, fancyLabel)
    #endif

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(plotTitle)
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.0)
    ax.axes.xaxis.set_major_locator(ticker.MultipleLocator(0.1))
    ax.axes.yaxis.set_major_locator(ticker.MultipleLocator(0.1))
    return fig, ax
#enddef

def plotSimpleROC(fpr,tpr,title):
    import matplotlib.pyplot as plt
    plt.plot(fpr, tpr)
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.0)
    plt.title(title)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.show()
#enddef

def addPoints(fpr, tpr, numThresh, thresh, fancyLabel):
    import math
    import matplotlib.pyplot as plt
    #import matplotlib.ticker as ticker

    # add threshold labels and circles
    # allow up to numThresh labels per plot, or numThresh+4 in the first multiple
    if not math.isinf(thresh[0]):
        maxThresh = thresh[0]  # if first (max) thresh is not infinite, then use it for label
    else:
        maxThresh = thresh[1]  # otherwise, use the next label which should be finite
    #endif

    stepfactor = round((len(thresh)-4) / numThresh)
    if stepfactor == 0:
        stepfactor = 1
    #endif

    for i in range(0, len(thresh), stepfactor):
        label, offset, fontsize = \
            get_ROC_Curve_Label_Offset_Fontsize(fpr[i], tpr[i], thresh[i], maxThresh, fancyLabel)
        if fancyLabel:
            plt.annotate(label, (fpr[i], tpr[i]), textcoords="offset points",
                         xytext=offset, ha='left', fontsize=fontsize)
        #endif
        plt.scatter(fpr[i], tpr[i], s=8, color='blue')
    #endfor
#enddef

def get_ROC_Curve_Label_Offset_Fontsize(x, y, t, maxThresh, fancyLabel):
    import math

    if 'fancyLabel' not in locals():
        fancyLabel = False

    inf_symbol = '\u221e'
    fontsize   = 'x-small'
    # fontsize: float or {'xx-small', 'x-small', 'small', 'medium', 'large', 'x-large', 'xx-large'}
    # label offsets are dependent on the fontsize

    # setup prefix and suffix for (0,0)
    if math.isinf(t) and t > 0:  # for positive infinity threshold
        t      = maxThresh       # replace it with greater than maxThresh
        if fancyLabel:
            prefix = '('; suffix = f',{inf_symbol}]'
        else:
            prefix = '>'; suffix = ''
    else:
        prefix = ''; suffix = ''
    #endif

    # setup prefix and suffix for (1,1)
    if x == 1 and y == 1:
        if fancyLabel:
            prefix = f'[-{inf_symbol},'; suffix = ']'
        else:
            prefix = ''; suffix = ''
    #endif

    # number formatting
    if t < 10:
        # label    = "{:.2f}".format(t)
        label = f'{prefix}{t:.2g}{suffix}'
    else:
        t          = int(round(t))
        label = f'{prefix}{t:.2d}{suffix}'
    #endif

    # setup width and offset
    if x == 1 and y == 1:
        if t < 1 and t > 0:   # if decimal included
            numberWidth  = 5 * (len(label)-1)
            decimalWidth = 1
            if fancyLabel:
                width    = numberWidth + decimalWidth -10
            else:
                width    = numberWidth + decimalWidth
        else:
            if fancyLabel:
                width    = 5 * len(label) - 10
            else:
                width    = 5 * len(label)
        #endif
        if fancyLabel:
            offset = (-width, 2)
        else:
            offset = (-width, -7)
    elif x == 0 and y == 0:
        offset = (7, 2)
    else:
        offset = (1, -7)
    # endif
    return label, offset, fontsize
#enddef

def plotOpt(fpr_opt, tpr_opt, thresh_opt, maxThresh, fancyLabel):
    import matplotlib.pyplot as plt

    # plot optimal ROC points
    plt.scatter(fpr_opt, tpr_opt, s=30, marker='o', alpha=1, facecolors='w', edgecolors='r')
    for fpr, tpr, thresh in zip(fpr_opt, tpr_opt, thresh_opt):
        label, offset, fontsize = get_ROC_Curve_Label_Offset_Fontsize(fpr, tpr, thresh, maxThresh, fancyLabel)
        if fancyLabel:
            plt.annotate(label, (fpr, tpr), textcoords="offset points",
                         color='r', xytext=offset, ha='left', fontsize=fontsize)
    #endfor
    return
#enddef