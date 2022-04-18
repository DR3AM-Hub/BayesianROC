# ChanceROC
  
Please read 'Chance ROC Code Documentation.docx' for much better documentation.  
  
BayesianROC (is a DeepROC), measures and plots with Chance and Bayesian iso performance baselines   
DeepROC     (is a FullROC), does group-wise partial measures and plots  
FullROC     (is a SimpleROC), includes all ROC points for all instances including tied scores  
Simple ROC, a class for ROC and mean ROC plots and analysis  
 
ConcordanceMatrixPlot (has an ROC object, any of the above), plots the Concordance Matrix  

TestChanceROC.py, creates a BayesianROC object and performs tests for the Chance ROC paper    
Alter parameters at the start of the file:  
  
quiet         = False  
testNum       = 1  
useFile       = False  
resultType    = 'matlab'  # matlab or python  
matlabResults = ['040', '356', '529', '536', '581', '639', '643']  
pythonResults = ['009']  
resultIndex   = 5  
pythonMinor   = ['nnReLU', 'svcRBF', 'plogr1']  
indexMinor    = 0  
groupAxis     = 'TPR'  
groups        = [[0.0, 0.8], [0.8, 1]]  
#groups       = [[0.0, 1/3], [1/3, 2/3], [2/3, 1.0]]  
#groups       = [[0, 1], [0.0, 1/3], [1/3, 2/3], [2/3, 1.0]]  
costs         = dict(cFP=1, cFN=4, cTP=0, cTN=0, costsAreRates=False)  # depends on the dataset  
popPrevalence = 0.3  # None uses sample prevalence. population depends on the dataset,  
                     # 0.297 for LBC data in 536 and 639, 0.345 for WBC data.  
