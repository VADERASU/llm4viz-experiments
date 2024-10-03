import json
import numpy as np
import pandas as pd
from os import listdir
from os.path import isfile, join
from multiprocessing import Pool

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn import metrics

from scipy.integrate import quad, trapezoid, simpson
from scipy.optimize import root, bisect, newton
from mpmath import appellf1, power, mp, linspace
from mpmath import quad as mpmathQuad
#from scipy.special import gamma, beta
from mpmath import gamma,beta,findroot

def resultsFunc(fileLoc,
                qaDict):
    resultFiles = [f for f in listdir(fileLoc) if isfile(join(fileLoc, f))]
    
    row = 0
    resultsDF = pd.DataFrame()
    for resultFile in resultFiles:
        nameList = resultFile.split('_')
        chartType = nameList[1]
        qNum = nameList[2].split('.')[0]
        
        inputDict = qaDict[chartType]['QAs'][qNum]
        
        resultsDF.at[row,'index'] = int(nameList[0])
        resultsDF.at[row,'chartType'] = chartType
        resultsDF.at[row,'qNum'] = qNum
        resultsDF.at[row,'question'] = inputDict['Q']
        resultsDF.at[row,'answer'] = inputDict['A']
        resultsDF.at[row,'qType'] = inputDict['Type']
        
        f = open(fileLoc+'/'+resultFile)
        dataDict = json.load(f)
        
        resultsDF.at[row,'qaInput'] = dataDict['qaInput']
        resultsDF.at[row,'response'] = dataDict['response']
        resultsDF.at[row,'time'] = dataDict['time']
        
        if (fileLoc.split('/')[0] == 'reVLAT_Viz+Choices')|(fileLoc.split('/')[0] == 'reVLAT_NoViz+Choices'):
            qaInputList = dataDict['qaInput'].split('\n')
            answerList = [x for i, x in enumerate(qaInputList) if (inputDict['A'] in x)&(i > 1)]
            
            if len(answerList) == 1:
                respList = answerList[0].split(' ')
                respLetter = respList[0]

                if len(dataDict['letterResp']) == 1:
                    resultsDF.at[row,'correct'] = respLetter==dataDict['letterResp'][0]
                else:
                    resultsDF.at[row,'correct'] = False
            else:
                print('STAHHHHHP',answerList,dataDict)
        
        row += 1
    
    return resultsDF

def numCorrCalc(resultDF,
                filDict,
                qaDict):
    llmFil = resultDF['LLM']==filDict['LLM']
    expTypeFil = resultDF['expType']==filDict['expType']
    filResultDF = resultDF.loc[llmFil&expTypeFil,:]
    
    row = 0
    outputDF = pd.DataFrame()
    for chartType in qaDict.keys():
        tempDict = qaDict[chartType]['QAs']
        
        for qNumStr in tempDict.keys():
            chartTypeFil = filResultDF['chartType']==chartType
            qNumFil = filResultDF['qNum']==qNumStr
            filFilResultDF = filResultDF.loc[chartTypeFil&qNumFil,:]
            filFilResultDF.reset_index(drop=True,
                                       inplace=True)
            
            outputDF.at[row,'chartType'] = chartType
            outputDF.at[row,'qNum'] = qNumStr
            outputDF.at[row,'qType'] = filFilResultDF.at[0,'qType']
            outputDF.at[row,'qType_simple'] = filFilResultDF.at[0,'qType_simple']
            outputDF.at[row,'numCorr'] = sum(filFilResultDF['correct'])
            outputDF.at[row,'totalNum'] = len(filFilResultDF['correct'])
            outputDF.at[row,'percentCorr'] = sum(filFilResultDF['correct'])/len(filFilResultDF['correct'])
            
            row += 1
    
    return outputDF

def perCorrDiffCalc(firstDF,
                    secondDF,
                    qaDict):
    
    row = 0
    outputDF = pd.DataFrame()
    for chartType in qaDict.keys():
        tempDict = qaDict[chartType]['QAs']
        
        for qNumStr in tempDict.keys():
            firstChartTypeFil = firstDF['chartType']==chartType
            firstQNumFil = firstDF['qNum']==qNumStr            
            filFirstDF = firstDF.loc[firstChartTypeFil&firstQNumFil,:]
            filFirstDF.reset_index(drop=True,
                                   inplace=True)
            
            secondChartTypeFil = secondDF['chartType']==chartType
            secondQNumFil = secondDF['qNum']==qNumStr
            filSecondDF = secondDF.loc[secondChartTypeFil&secondQNumFil,:]
            filSecondDF.reset_index(drop=True,
                                    inplace=True)
            
            outputDF.at[row,'chartType'] = chartType
            outputDF.at[row,'qNum'] = qNumStr
            outputDF.at[row,'qType'] = filFirstDF.at[0,'qType']
            outputDF.at[row,'qType_simple'] = filFirstDF.at[0,'qType_simple']
            outputDF.at[row,'percentDiff'] = filFirstDF.at[0,'percentCorr']-filSecondDF.at[0,'percentCorr']
            
            row += 1
    
    return outputDF

def varColDF(corrDF,
             perCorrDF,
             llmList=[],
             expList=[]):
    corrLogisticDF = corrDF.copy()
    colList = []
    for i in perCorrDF.index:
        chartType = perCorrDF.at[i,'chartType']
        qType = perCorrDF.at[i,'qType_simple']

        chartTypeCol = chartType.replace('100%','OneHundred').replace(' ','_')
        qTypeCol = qType.replace(' ','_').replace('/','_')

        interactCol = chartTypeCol+'&'+qTypeCol

        chartTypeFil = corrLogisticDF['chartType'] == chartType
        qTypeFil = corrLogisticDF['qType_simple'] == qType

        if interactCol not in colList:
            colList.append(interactCol)
            corrLogisticDF[interactCol] = chartTypeFil&qTypeFil
            corrLogisticDF[interactCol] = corrLogisticDF[interactCol].apply(lambda x:int(x))    

        if chartTypeCol not in colList:
            colList.append(chartTypeCol)
            #chartTypeFil = corrLogisticDF['chartType'] == chartType
            corrLogisticDF[chartTypeCol] = chartTypeFil
            corrLogisticDF[chartTypeCol] = corrLogisticDF[chartTypeCol].apply(lambda x:int(x))

        if qTypeCol not in colList:
            colList.append(qTypeCol)
            #qTypeFil = corrLogisticDF['qType_simple'] == qType
            corrLogisticDF[qTypeCol] = qTypeFil
            corrLogisticDF[qTypeCol] = corrLogisticDF[qTypeCol].apply(lambda x:int(x))
        
        if llmList:
            for llm in llmList:
                llmFil = corrLogisticDF['LLM'] == llm
                
                llmChartTypeCol = chartTypeCol+'&'+llm
                llmQTypeCol = qTypeCol+'&'+llm
                fullInterCol = chartTypeCol+'&'+qTypeCol+'&'+llm
                
                if llm not in colList:
                    colList.append(llm)
                    corrLogisticDF[llm] = llmFil
                    corrLogisticDF[llm] = corrLogisticDF[llm].apply(lambda x:int(x))
                
                if llmChartTypeCol not in colList:
                    colList.append(llmChartTypeCol)
                    corrLogisticDF[llmChartTypeCol] = chartTypeFil&llmFil
                    corrLogisticDF[llmChartTypeCol] = corrLogisticDF[llmChartTypeCol].apply(lambda x:int(x))
                    
                if llmQTypeCol not in colList:
                    colList.append(llmQTypeCol)
                    corrLogisticDF[llmQTypeCol] = qTypeFil&llmFil
                    corrLogisticDF[llmQTypeCol] = corrLogisticDF[llmQTypeCol].apply(lambda x:int(x))
                
                if fullInterCol not in colList:
                    colList.append(fullInterCol)
                    corrLogisticDF[fullInterCol] = chartTypeFil&qTypeFil&llmFil
                    corrLogisticDF[fullInterCol] = corrLogisticDF[fullInterCol].apply(lambda x:int(x))
                
                if expList:
                    for exp in expList:
                        expFil = corrLogisticDF['expType'] == exp
                        expCol = exp.replace('+','')

                        expLLM = expCol+'&'+llm
                        expLLMChartTypeCol = chartTypeCol+'&'+expCol+'&'+llm
                        expLLMQTypeCol = qTypeCol+'&'+expCol+'&'+llm
                        fullInterCol = chartTypeCol+'&'+qTypeCol+'&'+expCol+'&'+llm

                        if expLLM not in colList:
                            colList.append(expLLM)
                            corrLogisticDF[expLLM] = expFil&llmFil
                            corrLogisticDF[expLLM] = corrLogisticDF[expLLM].apply(lambda x:int(x))

                        if expLLMChartTypeCol not in colList:
                            colList.append(expLLMChartTypeCol)
                            corrLogisticDF[expLLMChartTypeCol] = expFil&llmFil&chartTypeFil
                            corrLogisticDF[expLLMChartTypeCol] = corrLogisticDF[expLLMChartTypeCol].apply(lambda x:int(x))
                            
                        if expLLMQTypeCol not in colList:
                            colList.append(expLLMQTypeCol)
                            corrLogisticDF[expLLMQTypeCol] = expFil&llmFil&qTypeFil
                            corrLogisticDF[expLLMQTypeCol] = corrLogisticDF[expLLMQTypeCol].apply(lambda x:int(x))
                        
                        if fullInterCol not in colList:
                            colList.append(fullInterCol)
                            corrLogisticDF[fullInterCol] = expFil&llmFil&chartTypeFil&qTypeFil
                            corrLogisticDF[fullInterCol] = corrLogisticDF[fullInterCol].apply(lambda x:int(x))
        
        if expList:
            for exp in expList:
                expFil = corrLogisticDF['expType'] == exp
                expCol = exp.replace('+','')
                
                expChartTypeCol = chartTypeCol+'&'+expCol
                expQTypeCol = qTypeCol+'&'+expCol
                fullInterCol = chartTypeCol+'&'+qTypeCol+'&'+expCol
                
                if expCol not in colList:
                    colList.append(expCol)
                    corrLogisticDF[expCol] = expFil
                    corrLogisticDF[expCol] = corrLogisticDF[expCol].apply(lambda x:int(x))

                if expChartTypeCol not in colList:
                    colList.append(expChartTypeCol)
                    corrLogisticDF[expChartTypeCol] = chartTypeFil&expFil
                    corrLogisticDF[expChartTypeCol] = corrLogisticDF[expChartTypeCol].apply(lambda x:int(x))
                    
                if expQTypeCol not in colList:
                    colList.append(expQTypeCol)
                    corrLogisticDF[expQTypeCol] = qTypeFil&expFil
                    corrLogisticDF[expQTypeCol] = corrLogisticDF[expQTypeCol].apply(lambda x:int(x))
                
                if fullInterCol not in colList:
                    colList.append(fullInterCol)
                    corrLogisticDF[fullInterCol] = chartTypeFil&qTypeFil&expFil
                    corrLogisticDF[fullInterCol] = corrLogisticDF[fullInterCol].apply(lambda x:int(x))

    corrLogisticDF = corrLogisticDF.loc[:,['corrInt']+colList]
    exogFormStr = '+'.join(colList)
    corrLogisticFormStr = 'corrInt ~ '+exogFormStr
    
    return corrLogisticDF,corrLogisticFormStr

def logisticModRun(inputDict):
    paramDict = inputDict['paramDict']
    
    kfoldNum = inputDict['kfoldNum']
    dfFileLoc = inputDict['dfFileLoc']

    corrLogisticDF = pd.read_csv(dfFileLoc)
    
    train_index = inputDict['train_index']
    test_index = inputDict['test_index']

    trainX = corrLogisticDF.loc[train_index,corrLogisticDF.columns != 'corrInt']
    trainY = corrLogisticDF.loc[train_index,'corrInt']

    testX = corrLogisticDF.loc[test_index,corrLogisticDF.columns != 'corrInt']
    testY = corrLogisticDF.loc[test_index,'corrInt']
    
    clf = LogisticRegression(**paramDict).fit(trainX,
                                              trainY)
    predProbY = clf.predict_proba(testX)
    predY = clf.predict(testX)

    fpr, tpr, rocThresholds = metrics.roc_curve(testY,predProbY[:,1])
    precision, recall, prThresholds = metrics.precision_recall_curve(testY,predProbY[:,1])
    
    auc_roc = metrics.auc(fpr,tpr)
    auc_prc = metrics.auc(recall,precision)
    f1_score = metrics.f1_score(testY,predY)
    aps = metrics.average_precision_score(testY,predProbY[:,1])
    score = clf.score(testX, testY)

    metricsDict = {'auc_roc':auc_roc,
                   'auc_prc':auc_prc,
                   'f1_score':f1_score,
                   'aps':aps,
                   'score':score}

    clf = None
    print(paramDict,score,kfoldNum)
    
    return paramDict,metricsDict,kfoldNum

def logisticTests(corrLogisticDF,
                  corrLogisticFileLoc,
                  randSeedList,
                  cList,
                  penaltyList,
                  l1RatioList,
                  maxIter=10000):
    
    inputList = []
    for penaltyDict in penaltyList:
        penalty = penaltyDict['penalty']
        solverList = penaltyDict['solvers']

        for solver in solverList:
            for randSeed in randSeedList:
                kf = KFold(n_splits=10,
                           shuffle=True,
                           random_state=randSeed)
                for i, (train_index, test_index) in enumerate(kf.split(corrLogisticDF)):
                    trainX = corrLogisticDF.loc[train_index,corrLogisticDF.columns != 'corrInt']
                    trainY = corrLogisticDF.loc[train_index,'corrInt']

                    testX = corrLogisticDF.loc[test_index,corrLogisticDF.columns != 'corrInt']
                    testY = corrLogisticDF.loc[test_index,'corrInt']

                    if penalty == 'elasticnet':
                        for cVal in cList:
                            for l1Ratio in l1RatioList:
                                inputList.append({'paramDict':{'random_state':randSeed,
                                                               'solver':solver,
                                                               'C':cVal,
                                                               'max_iter':maxIter,
                                                               'l1_ratio':l1Ratio,
                                                               'penalty':penalty},
                                                  'kfoldNum':i,
                                                  'train_index':train_index,
                                                  'test_index':test_index,
                                                  'dfFileLoc':corrLogisticFileLoc})
                    elif penalty is None:
                        inputList.append({'paramDict':{'random_state':randSeed,
                                                       'solver':solver,
                                                       'max_iter':maxIter,
                                                       'penalty':penalty},
                                          'kfoldNum':i,
                                          'train_index':train_index,
                                          'test_index':test_index,
                                          'dfFileLoc':corrLogisticFileLoc})
                    else:
                        for cVal in cList:
                            inputList.append({'paramDict':{'random_state':randSeed,
                                                           'solver':solver,
                                                           'C':cVal,
                                                           'max_iter':maxIter,
                                                           'penalty':penalty},
                                              'kfoldNum':i,
                                              'train_index':train_index,
                                              'test_index':test_index,
                                              'dfFileLoc':corrLogisticFileLoc})
    
    return inputList

def nudgeFactor(x,
                nudge=0.001):
    if x == 0:
        return x+nudge
    elif x == 1:
        return x-nudge
    else:
        return x

# Functions for Beta-difference distribution
def appellFunc(a,
               b1,
               b2,
               c,
               x1,
               x2,
               intLimit=50,
               altInt='simpson',
               nPts=1000,
               dps=15,
               builtinBool=False):
    mp.dps = dps
    const = gamma(c)/(gamma(a)*gamma(c-a))
    try:
        if builtinBool:
            result = appellf1(a,
                              b1,
                              b2,
                              c,
                              x1,
                              x2)
            return result
        else:
            integral = mpmathQuad(lambda u: power(u,a-1)*power(1-u,c-a-1)*power(1-u*x1,-b1)*power(1-u*x2,-b2),
                                  [0,1],
                                  error=True)
            return const*integral[0]
    except:
        integral = quad(lambda u: (u**(a-1))*((1-u)**(c-a-1))*((1-u*x1)**(-b1))*((1-u*x2)**(-b2)),0,1,limit=intLimit)
        #print('scipy',integral)
        if np.isnan(integral[0]):
            xList = np.linspace(0,1,nPts)
            yList = [(u**(a-1))*((1-u)**(c-a-1))*((1-u*x1)**(-b1))*((1-u*x2)**(-b2)) for u in xList]
        
            if (max(yList) == np.inf)|(sum([np.isnan(y) for y in yList]) > 0):
                newXlist = [xList[i] for i,y in enumerate(yList) if (y != np.inf)&(not np.isnan(y))]
                newYlist = [y for y in yList if (y != np.inf)&(not np.isnan(y))]
                xList = newXlist
                yList = newYlist
        
            if altInt == 'trap':
                trapInt = trapezoid(yList,
                                    x=xList)
                return const*trapInt
            elif altInt == 'simpson':
                simpsonInt = simpson(yList,
                                    x=xList)
                return const*simpsonInt
        else:
            return const*integral[0]

def dDiffProp(x,
              alpha1,
              beta1,
              alpha2,
              beta2,
              intLimit=50,
              altInt='simpson',
              nPts=1000,
              dps=15,
              outputType='float'):
    mp.dps = dps
    if (alpha1+alpha2 > 1)&(beta1+beta2 > 1)&(x == 0):
        return float(beta(alpha1+alpha2-1,beta1+beta2-1)/(beta(alpha1,beta1)*beta(alpha2,beta2)))
    elif (x >= -1)&(x < 0):
        const = beta(alpha1,beta2)/(beta(alpha1,beta1)*beta(alpha2,beta2))
        xExp = power(-x,beta1+beta2-1)*power(1+x,alpha1+beta2-1)
        appellVal = appellFunc(beta2,
                               1-alpha2,
                               alpha1+alpha2+beta1+beta2-2,
                               alpha1+beta2,
                               1-x**2,
                               1+x,
                               intLimit,
                               altInt,
                               nPts,
                               dps)
        
        if outputType == 'float':
            return float(const*xExp*appellVal)
        elif outputType == 'mpf':
            return const*xExp*appellVal
    elif (x >= 0)&(x <= 1):
        const = beta(alpha2,beta1)/(beta(alpha1,beta1)*beta(alpha2,beta2))
        xExp = power(x,beta1+beta2-1)*power(1-x,alpha2+beta1-1)
        appellVal = appellFunc(beta1,
                               alpha1+alpha2+beta1+beta2-2,
                               1-alpha1,
                               beta1+alpha2,
                               1-x,
                               1-x**2,
                               intLimit,
                               altInt,
                               nPts,
                               dps)
        
        if outputType == 'float':
            return float(const*xExp*appellVal)
        elif outputType == 'mpf':
            return const*xExp*appellVal
    else:
        print('x is outside of the domain.  Please set it to a number between -1 and 1.')
    
def pDiffProp(x,
              alpha1,
              beta1,
              alpha2,
              beta2,
              intType='mpmath',
              appellLimit=50,
              intLimit=50,
              altBool=True,
              altInt='simpson',
              appellNPts=1000,
              nPts=50,
              maxdegree=0,
              dps=15,
              outputType='float',
              verbose=False):
    mp.dps = dps
    if (intType=='scipy')&altBool:
        tList = np.linspace(-1,x,nPts)
        yList = [dDiffProp(t,alpha1,beta1,alpha2,beta2,appellLimit,altInt,appellNPts,dps+2,'float') for t in tList]
        
        if (max(yList) == np.inf)|(sum([np.isnan(y) for y in yList]) > 0):
            newTlist = [tList[i] for i,y in enumerate(yList) if (y != np.inf)&(not np.isnan(y))]
            newYlist = [y for y in yList  if (y != np.inf)&(not np.isnan(y))]
            tList = newTlist
            yList = newYlist
        
        if altInt == 'trap':
            trapInt = trapezoid(yList,
                                x=tList)
            
            if verbose:
                print('scipy, trapInt',trapInt)
            return trapInt
        elif altInt == 'simpson':
            simpsonInt = simpson(yList,
                                    x=tList)
            
            if verbose:
                print('scipy, simpsonInt',simpsonInt)
            return simpsonInt
    elif (x >= -1)&(x <= 0):
        if x == 0:
            x = -10**(-(dps+2))
        
        const = beta(alpha1,beta2)/(beta(alpha1,beta1)*beta(alpha2,beta2))
        if intType=='mpmath':
            integral = mpmathQuad(lambda t: power(-t,beta1+beta2-1)*power(1+t,alpha1+beta2-1)*appellFunc(beta2,
                                                                                                         1-alpha2,
                                                                                                         alpha1+alpha2+beta1+beta2-2,
                                                                                                         alpha1+beta2,
                                                                                                         1-t**2,
                                                                                                         1+t,
                                                                                                         appellLimit,
                                                                                                         altInt,
                                                                                                         appellNPts,
                                                                                                         dps+2),
                                  #[-1, x],
                                  linspace(-1,x,nPts),
                                  error=True,
                                  maxdegree=maxdegree)
            if verbose:
                print('mpmath',const,integral,mp.dps)

            if outputType == 'float':
                return float(const*integral[0])
            elif outputType == 'mpf':
                return const*integral[0]
        elif intType=='scipy':
            integral = quad(lambda t: ((-t)**(beta1+beta2-1))*((1+t)**(alpha1+beta2-1))*appellFunc(beta2,
                                                                                                   1-alpha2,
                                                                                                   alpha1+alpha2+beta1+beta2-2,
                                                                                                   alpha1+beta2,
                                                                                                   1-t**2,
                                                                                                   1+t,
                                                                                                   appellLimit,
                                                                                                   altInt,
                                                                                                   appellNPts,
                                                                                                   dps+2),
                            -1,
                            x,
                            limit=intLimit)
            if verbose:
                print('scipy',integral)
            return const*integral[0]
        
    elif (x > 0)&(x <= 1):
        const = beta(alpha2,beta1)/(beta(alpha1,beta1)*beta(alpha2,beta2))
        if intType=='mpmath':
            integral = mpmathQuad(lambda t: power(t,beta1+beta2-1)*power(1-t,alpha2+beta1-1)*appellFunc(beta1,
                                                                                                        alpha1+alpha2+beta1+beta2-2,
                                                                                                        1-alpha1,
                                                                                                        beta1+alpha2,
                                                                                                        1-t,
                                                                                                        1-t**2,
                                                                                                        appellLimit,
                                                                                                        altInt,
                                                                                                        appellNPts,
                                                                                                        dps+2),
                                  #[0, x],
                                  linspace(x,1,nPts),
                                  error=True,
                                  maxdegree=maxdegree)
            if verbose:
                print('mpmath (upperCDF)',const,integral,mp.dps)
            
            if outputType == 'float':
                return float(1-const*integral[0])
            elif outputType == 'mpf':
                return 1-const*integral[0]
        elif intType=='scipy':
            integral = quad(lambda t: (t**(beta1+beta2-1))*((1-t)**(alpha2+beta1-1))*appellFunc(beta1,
                                                                                                alpha1+alpha2+beta1+beta2-2,
                                                                                                1-alpha1,
                                                                                                beta1+alpha2,
                                                                                                1-t,
                                                                                                1-t**2,
                                                                                                appellLimit,
                                                                                                altInt,
                                                                                                appellNPts,
                                                                                                dps+2),
                            x,
                            1,
                            limit=intLimit)
            if verbose:
                print('scipy (upperCDF)',integral)
            return 1-const*integral[0]
    else:
        print('x is outside of the domain.  Please set it to a number between -1 and 1.')

def qDiffProp(q,
              alpha1,
              beta1,
              alpha2,
              beta2,
              rootType='mpmath',
              biTol=0.0001,
              newtonTol=0.0000001,
              biMaxN=5,
              newtonMaxN=1000,
              intType='mpmath',
              appellLimit=50,
              intLimit=50,
              altBool=True,
              altInt='simpson',
              appellNPts=1000,
              nPts=50,
              maxdegree=0,
              dps=15,
              dOutType='float',
              pOutType='float',
              outputType='float',
              verbose=False):
    if rootType == 'mpmath':
        dOutType='mpf'
        pOutType='mpf'
    else:
        dOutType='float'
        pOutType='float'

    # Set up function and derivative function
    func = lambda x: pDiffProp(x,
                               alpha1,
                               beta1,
                               alpha2,
                               beta2,
                               intType,
                               appellLimit,
                               intLimit,
                               altBool,
                               altInt,
                               appellNPts,
                               nPts,
                               maxdegree,
                               dps,
                               pOutType,
                               verbose)-q
    dfFunc = lambda x: dDiffProp(x,
                                 alpha1,
                                 beta1,
                                 alpha2,
                                 beta2,
                                 appellLimit,
                                 altInt,
                                 appellNPts,
                                 dps+2,
                                 dOutType)
    if rootType == 'mpmath':
        mp.dps = dps
        # Bisection method to find initial value for x
        if verbose:
            print('Start of Bisection Method')
        xPrev = findroot(func,
                         [-1,1],
                         solver='bisect',
                         tol=biTol,
                         maxsteps=biMaxN,
                         verify=False)
        
        # Newton-Raphson method to find a better x
        if verbose:
            print('Start of Newton-Raphson Method')
        xCurr = findroot(func,
                         xPrev,
                         solver='newton',
                         tol=newtonTol,
                         maxsteps=newtonMaxN,
                         df=dfFunc)
        
        if outputType == 'float':
            return float(xCurr)
        elif outputType == 'mpf':
            return xCurr
    elif rootType == 'scipy':
        # Bisection method to find initial value for x
        if verbose:
            print('Start of Bisection Method')
        xPrev = bisect(func,
                       -1,
                       1,
                       maxiter=biMaxN,
                       xtol=biTol,
                       disp=False)
        
        # Newton-Raphson method to find a better x
        if verbose:
            print('Start of Newton-Raphson Method')
        xCurr = newton(func,
                       xPrev,
                       fprime=dfFunc,
                       maxiter=newtonMaxN,
                       rtol=newtonTol)
        return xCurr
    elif rootType == 'manual':
        # Bisection method to find initial value for x
        a = -1
        b = 1
        n = 0
        keepLoop = True

        while (n < biMaxN)&keepLoop:
            xPrev = (a+b)/2
            funcVal = func(xPrev)
            aFuncVal = func(a)
            if (funcVal == 0)|(funcVal**2 < biTol):
                keepLoop = False
                
            n += 1
            if np.sign(aFuncVal) == np.sign(funcVal):
                a = xPrev
            else:
                b = xPrev
        
        # Newton-Raphson method to find a better x
        n = 0

        pdf = dfFunc(xPrev)
        funcVal = func(xPrev)
        
        xCurr = xPrev - (funcVal)/pdf
        while (funcVal**2 > newtonTol)&(n < newtonMaxN):
            pdf = dfFunc(xPrev)
            funcVal = func(xPrev)
            
            xCurr = xPrev - (funcVal)/pdf
            if xCurr < -1:
                print('xCurr assigned to -1 (need to investigate)')
                xCurr = -1
            elif xCurr > 1:
                print('xCurr assigned to 1 (need to investigate)')
                xCurr = 1

            n += 1
            xPrev = xCurr
    
        return xCurr

def pdfDiffProp(x,
                *args,
                **kwargs):
    return [dDiffProp(xVal,*args,**kwargs) for xVal in x]

def cdfDiffProp(x,
                *args,
                **kwargs):
    return [pDiffProp(xVal,*args,**kwargs) for xVal in x]

def ppfDiffProp(x,
                *args,
                **kwargs):
    return [qDiffProp(xVal,*args,**kwargs) for xVal in x]