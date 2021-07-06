

# import necessarily libraries 
#!pip install pyts
#!pip install imblearn
import warnings
warnings.filterwarnings("ignore")
from pyts.approximation import SymbolicAggregateApproximation, SymbolicFourierApproximation
from sklearn.metrics import accuracy_score
from sklearn.ensemble import  RandomForestClassifier
import pandas as pd
import numpy as np
from pyts.preprocessing import StandardScaler
import random
from collections import Counter
from imblearn.over_sampling import BorderlineSMOTE, SMOTE
from sklearn.model_selection import LeaveOneOut
import heapq
from sklearn.model_selection import cross_val_score
import os


# ### Define functions required for Co-eye

#Find the most confident label across lenses in one representation (either SAX or SFA)
def mostConf_one(matrcies, labels):
    #Marerices is of length L: number of lenses/forests
    #Each matrix in matrices is the probalistic accuracy of test data 
    #Matrix of size m X n: where m is the number of test instances,n is the number of classes  
    predLabels=[]
    # For each test data
    for row in range(len(matrcies[0])):
        maxConf= 0
    # For each Forest/lense
        for mat in matrcies: 
            for col in range(len(labels)):
                if(mat[row][col]>maxConf):
                    maxConf= mat[row][col]
                    Conflabel= labels[col]    
        #Return the most confident lable across lenses 
        predLabels.append(Conflabel)
    return predLabels


# ### Implementation of Vote function described in Section 4.3 in the paper


# #Find the most confident labeles across lenses of the two representations ( SAX and SFA)

def mostConf_multi(matrcies, labels, sfa_n):
    #Marerices is of length L: number of lenses/forests (total for SAX and SFA)
    #Each matrix in matrices is the probalistic accuracy of test data 
    #Matrix of size m X n: where m is the number of test instances,n is the number of classes (labels)
    #Sax_n number of lenses in SFA
    
    predLabel=[]
    multiSFA = False
    multiSAX= False
    
    # For each test data, we define the most confident and second-most confident for each presentation, then vote between them
    for row in range(len(matrcies[0])):
        #Initialise
        maxConf_SFA= 0
        SFAConf=[]
        SFALables= []
        SAXConf=[]
        SAXLables= []
        #Look into SFA only matrices
        #Find the max confidence/ prbability in SFA
        for mat in matrcies[:sfa_n]: 
            for col in range(len(labels)):
                SFAConf.append(mat[row][col])
                SFALables.append(labels[col])
                if(mat[row][col]>maxConf_SFA):
                    maxConf_SFA= mat[row][col]
                    Conflabel_SFA= labels[col]
        
        #Second best flag
        SB1= False
        #Special case: when multiple lenses have the same accuracy (most confident)
        if(SFAConf.count(maxConf_SFA)>1): 
            indices = [i for i, x in enumerate(SFAConf) if x == maxConf_SFA]                
            l= [SFALables[i] for i in indices]
            if(len(set(l))!=1): 
                maxIr=max(l.count(y) for y in set(l))
                k= [l.count(y) for y in set(l)]
                        
                # Find the most common label
                if(k.count (maxIr)==1): 
                    cnt = Counter(l)
                    Conflabel_SFA= cnt.most_common(2)[0][0] 
                    secondBestSFALabel= cnt.most_common(2)[1][0] 
               
            #If no common label, in case of tie, the label is chosen randomly.
                else: 
                    shuff=[]
                    for it in set(l): 
                        if (l.count(it)==maxIr): shuff.append(it)
                    shuff= list(set(l))
                    random.shuffle(shuff)
                    Conflabel_SFA= shuff[0]
                    secondBestSFALabel= shuff[1]
                SB1= True
                #Set the second best 
                secondBestSFA= maxConf_SFA
        
        if(SB1== False):
            secondBestSFA=  max(n for n in SFAConf if n!=maxConf_SFA)      
            secondBestSFALabel=  SFALables[SFAConf.index(secondBestSFA)]
        
        #Same steps for SAX
        maxConf_SAX= 0
        SB2= False
        for mat in matrcies[sfa_n:]: 
            for col in range(len(labels)):
                SAXConf.append(mat[row][col])
                SAXLables.append(labels[col])
                if(mat[row][col]>maxConf_SAX):    
                    maxConf_SAX= mat[row][col]
                    Conflabel_SAX= labels[col]
        if(SAXConf.count(maxConf_SAX)>1): 
            indices = [i for i, x in enumerate(SAXConf) if x == maxConf_SAX]                
            l= [SAXLables[i] for i in indices]
            if(len(set(l))!=1): 
                print ("Conflict on max confident", l) 
                maxIr=max(l.count(y) for y in set(l))
                k= [l.count(y) for y in set(l)]
#                 How many equal items with max confident value  
                if(k.count (maxIr)==1):  
                    cnt = Counter(l)
                    Conflabel_SAX= cnt.most_common(2)[0][0] 
                    secondBestSAXLabel= cnt.most_common(2)[1][0] 
                else:
                    shuff=[]
                    for it in set(l): 
                        if (l.count(it)==maxIr): shuff.append(it)
                    print ("tie", shuff)
                    random.shuffle(shuff)
                    Conflabel_SAX= shuff[0]
                    secondBestSAXLabel= shuff[1]
                   
                secondBestSAX= maxConf_SAX
#                 print("most common", Conflabel_SAX, "Second Best", secondBestSAXLabel)
                SB2= True
#             print("indecies", indices, "labels" ,labels)
        if(SB2== False):
            secondBestSAX=  max(n for n in SAXConf if n!=maxConf_SAX)      
            secondBestSAXLabel=  SAXLables[SAXConf.index(secondBestSAX)]
        
#         print ("Best SAX",maxConf_SAX , Conflabel_SAX)    
#         print ("Second Best SAX",secondBestSAX , secondBestSAXLabel)    
        
   #-----------------------------------
        # In case of agreement between most Conf SAX and SFA
        if(Conflabel_SAX==Conflabel_SFA): 
            best= Conflabel_SAX
        # If no agreement, then second best is testes
        elif(secondBestSAX>secondBestSFA): best= secondBestSAXLabel
        else: best= secondBestSFALabel
        #Accumulate labels with the best choice
        predLabel.append(best)
#         print("Best SFA, SAX", Conflabel_SAX, Conflabel_SFA,"Second best SFA, SAX", secondBestSAXLabel, secondBestSFALabel,"Best",  best)
#         print ("----------------------------------------")

    return predLabel


# ### searchLenses [Algorithm 2 (Co-eye paper)]



# Finding best pairs for SFA transformation 
def searchLense_SFA(X_train, y_train): 
    #Input is training data (X_train, y_train)
    #Returns selected pairs for SFA transformation based on cross validation
   
    #Set ranges (Seg, alpha) parameters
    print("\n Generating SFA parameters....")
    maxCoof= 130
    if(X_train.shape[1]<maxCoof): maxCoof = X_train.shape[1]-1
    if(X_train.shape[1]<100): n_segments=list(range(5, maxCoof,5))
    else: n_segments=list(range(10, maxCoof,10))
    
    maxBin= 26
    if(X_train.shape[1]<maxBin): maxBin = X_train.shape[1]-2
    if(X_train.shape[0]<maxBin): maxBin = X_train.shape[0]-2
    alphas= range(3,maxBin)
    
    pairs= []

    # Learning parameteres using 5 folds cross validation

    for alpha in alphas: 
        s= []
        for seg in n_segments:  
            SFA= SymbolicFourierApproximation( n_coefs=seg, n_bins= alpha, alphabet= 'ordinal')
            X_SFA = SFA.fit_transform(X_train)        
            scores= 0
            RF_clf= RandomForestClassifier(n_estimators=100, random_state=0)
            scores = cross_val_score(RF_clf, X_SFA, y_train, cv=5)  
            s.append(scores.mean())
        winner = np.argwhere(s >= np.amax(s)-0.01)
        for i in winner.flatten().tolist(): 
            bestCof= n_segments[i]
            pairs.append((bestCof, alpha)) 
    print("No of selected pairs: ", len(pairs))
    print("SFA pamaeter selection, done!")
    return pairs


# In[13]:


# Finding best pairs for SAX transformation 

def searchLense_SAX(X_train, y_train): 
    #Input is training data (X_train, y_train)
    #Returns selected pairs for SAX transformation based on cross validation    
    
    #Set range (alpha) parameter
    print("Generating SAX parameters....")

    maxBin= 26
    if(X_train.shape[1]<maxBin): maxBin = X_train.shape[1]
    alphas= range(3,maxBin)
    s= []
    pairs= []

    #Learning parameters using 5 folds cross validation
    for alpha in alphas: 
        SAX = SymbolicAggregateApproximation(strategy= 'uniform', n_bins=alpha, alphabet= 'ordinal')
        X_train_SAX = SAX.fit_transform(X_train)           
        scores= 0
        RF_clf= RandomForestClassifier(n_estimators=100, random_state=0)
        scores = cross_val_score(RF_clf, X_train_SAX, y_train, cv=5)  
        s.append(scores.mean())
    winner = np.argwhere(s >= np.mean(s)-0.01)
    for i in winner.flatten().tolist(): 
        bestCof= alphas[i]
        pairs.append(bestCof)    
    
    print("No of selected pairs: ", len(pairs))
    print("SAX pamaeter selection, done!")
    
    return pairs


# ### Co-eye Main Run


#Main Run
#Initialisation
All_Acc=[]
SFA_acc=[]
SAX_acc=[]
SFA_pairs= []
SAX_pairs= []

#List names of datasets to use/test
files= ['Coffee', 'Beef']

for f in files: 
    #Import and balance data 
    dirname = os.getcwd()
    dftrain= pd.read_csv(dirname+'/UCR/'+f+'/'+f+'_TRAIN.tsv',header=None,  sep='\t')
    dftest= pd.read_csv(dirname+'/UCR/'+f+'/'+f+'_TEST.tsv',header=None,  sep='\t')
    print("\n-------------------\nDataset: ", f)
    
    #Split data to X_train, y_train and X_test, y_test
    X_train= dftrain.drop([0], axis=1).values   
    y_train= dftrain[0]
    
    X_test= dftest.drop([0], axis=1).values
    y_test= dftest[0]
    print ( X_train.shape, X_test.shape, y_train.shape, y_test.shape)

    
    #Check for class imbalance
    print ("Class distribution ",sorted(Counter(y_train).items()))
    min_neighbours= min(Counter(y_train).items(), key=lambda k: k[1])[1]
    max_neighbours= max(Counter(y_train).items(), key=lambda k: k[1])[1]
    
    
    if(min_neighbours==max_neighbours): 
        print ("Data has an equal distribution among classes \n")
        SMOTE_Xtrain= X_train
        SMOTE_ytrain= y_train
        
    #Apply SMOTE if data is imbalanced
    else:
        if (min_neighbours>5): min_neighbours= 6
        SMOTE_Xtrain, SMOTE_ytrain = SMOTE(sampling_strategy="all", k_neighbors= min_neighbours-1, random_state=42).fit_resample(X_train, y_train )
        print("Class distribution after balance", sorted(Counter(SMOTE_ytrain).items()))
        print("---------------------")
        print ("After Balance",  SMOTE_Xtrain.shape, X_test.shape, SMOTE_ytrain.shape, y_test.shape,"\n")
    
    #Initialise parameters for each dataset
    #Random forest probability matrices for all lenses 
    RFmatrices= [] 
   

    #Training Phase
    #--------------
    pairs= searchLense_SFA (SMOTE_Xtrain, SMOTE_ytrain)
    for n_coefs, n_bins  in pairs: 
        SFA= SymbolicFourierApproximation( n_coefs=n_coefs, n_bins= n_bins, alphabet= 'ordinal')
        #Transform to SFA
        X_train_SFA = SFA.fit_transform(SMOTE_Xtrain)
        X_test_SFA = SFA.fit_transform(X_test)
        # Build RF on each lense
        RF_clf= RandomForestClassifier(n_estimators=100, random_state=0)
        RF_clf.fit(X_train_SFA,SMOTE_ytrain)
        #Store prediction probability for test data 
        model_pred=  RF_clf.predict_proba(X_test_SFA)
        #accumulate RFmatrices for a lense
        RFmatrices.append(model_pred)

    
    length= len(RFmatrices)
    #Best probablistic accuracy using SFA only 
    acc= accuracy_score(y_test, mostConf_one(RFmatrices,RF_clf.classes_))
    SFA_acc.append(acc)
    SFA_pairs.append(length)
    print("*SFA Correct classification rate: {:2.4f} ".format(acc), "Error rate", 1-acc,"\n")
    
    SAXacc=[]
    # Transform to SAX Lenses

    saxPairs= searchLense_SAX(SMOTE_Xtrain, SMOTE_ytrain)
    for n_bins in saxPairs:
        sax = SymbolicAggregateApproximation(strategy= 'uniform', n_bins=n_bins, alphabet= 'ordinal')
        #Transform to SAX
        X_sax = sax.fit_transform(SMOTE_Xtrain)
        X_test_sax = sax.fit_transform(X_test)
        #Build RF for each SAX lense
        RF_clf= RandomForestClassifier(n_estimators=100, random_state=0)
        RF_clf.fit(X_sax,SMOTE_ytrain)
        #Store prediction probability for test data 
        model_pred= RF_clf.predict_proba(X_test_sax)
        #Sax only lenses
        SAXacc.append(model_pred)
        #accumulate RFmatrices for all lenses
        RFmatrices.append(model_pred)
        
    #Classification Phase
    #---------
    saxAcc= accuracy_score(y_test, mostConf_one(SAXacc,RF_clf.classes_))
    SAX_acc.append(saxAcc)
    SAX_pairs.append(len(SAXacc))
    print("*SAX only Correct classification rate: {:2.4f} ".format(saxAcc), "Error rate", 1-saxAcc)

    #Combining both SAX and SFA using mostConf_multi method
    acc= accuracy_score(y_test, mostConf_multi(RFmatrices,RF_clf.classes_,length))
    print("\n Co-eye Correct classification rate: {:2.4f} ".format(acc), "Error rate", 1-acc)
    print(" -----------------------------------------")
    All_Acc.append(acc)



