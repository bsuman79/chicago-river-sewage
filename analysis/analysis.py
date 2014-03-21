import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import datetime
from sklearn import preprocessing, svm, metrics, cross_validation, linear_model
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.pipeline import Pipeline


def add_months(sourcedate,months):
     month = sourcedate.month - 1 + months
     year = sourcedate.year + month / 12
     month = month % 12 + 1
     day = min(sourcedate.day,calendar.monthrange(year,month)[1])
     return datetime.date(year,month,day)

def read_data(directory='../data/', file='clean-waterway-measurements.csv', file1='cso_events_timestamped.csv', file2='mwrd_rain_measurements.csv',datetosplit='2010-02-01'):
   format = '%Y-%m-%d' 
   next_month= datetime.datetime.strptime(datetosplit, format)+datetime.timedelta(weeks=6)
   
   data= pd.read_csv(directory+file) 
   #print data.columns.values.tolist()
   
   data.replace(to_replace=['<','\*','\'','_____','3..1','n/a','t/x'],value=['','','',np.nan,np.nan,np.nan,np.nan],inplace=True,regex=True)
   data.replace(to_replace=['na'],value=[np.nan],inplace=True)
   data['fec_col(cts/100ml)'][np.array(data['fec_col(cts/100ml)'],dtype=float) <200]=1
   data['fec_col(cts/100ml)'][np.array(data['fec_col(cts/100ml)'],dtype=float) >=400]=0
   data['fec_col(cts/100ml)'][(np.array(data['fec_col(cts/100ml)'],dtype=float) <400) & (np.array(data['fec_col(cts/100ml)'],dtype=float) >=200)]=1

   # create train data, take data till datetosplit 
   data_split= data[data['collect(date)']<datetosplit]
   # create test data, take data start from datetosplit 
   data_split_test= data[(data['collect(date)']>=datetosplit) & (data['collect(date)']<str(next_month))]
   #print data.ix[:,30:40][(data['collect(date)']>='2012-07-30') & (data['collect(date)']<'2012-07-31')].fillna(0.0)
   #exit()
   return data_split,data_split_test

def prepare_data(data):   
   data_grouped= data.groupby(['north_latitude','west_longitude'])  
   group={}
   ct=1
   data_full, response_full=[],[]
   for key, val in data_grouped:
       #print key
       val=val.fillna(0.0)
       group[key]=val
       data2=np.hstack((val.ix[:,1:19],val.ix[:,20:31],val.ix[:,32:52],val.ix[:,53:69]))
       #data2=val[['do(mg/l)','ph()','turb(ntu)','temp(deg_c)']]
       data_full.append(data2)
       response_full.extend(val['fec_col(cts/100ml)'])
       #print "---------location= ",ct," done"
       ct+=1
   return np.vstack(data_full), response_full 

def scale_data(train_data, test_data):
    scaler= preprocessing.MinMaxScaler().fit(train_data)
    #print scaler.mean_, scaler.std_
    return scaler.fit_transform(train_data), scaler.transform(test_data)

def cv_loop(X, y, model, N=8,SEED=25):
    mean_auc = 0.
    for i in range(N):
        X_train, X_cv, y_train, y_cv = cross_validation.train_test_split(
                                       X, y, test_size=.20, 
                                       random_state = i*SEED)
        model.fit(X_train, y_train)
        preds = model.predict_proba(X_cv)[:,1]
        auc = metrics.roc_auc_score(y_cv, preds)
        print "AUC (fold %d/%d): %f" % (i + 1, N, auc)
        mean_auc += auc
    return mean_auc/N

def hyperparam(Xt,y,model,N=5):    
    print "Performing hyperparameter selection..."
    # Hyperparameter selection loop
    score_hist = []
    Cvals = np.logspace(2, 6, 15, base=2)
    for C in Cvals:
        model.C = C
        score = cv_loop(Xt, y, model, N)
        score_hist.append((score,C))
        print "C: %f Mean AUC: %f" %(C, score)
    bestC = sorted(score_hist)[-1][1]
    print "Best C value: %f" % (bestC)
    return bestC 

if __name__=="__main__":
    data,data1=read_data()
    train_data, train_response=prepare_data(data)
    test_data,test_response=prepare_data(data1)
    #print train_data[0,:],test_data[0,:]

    #train_data,test_data= scale_data(train_data.astype(float),test_data.astype(float))
    #model=svm.SVC(probability=True)

    model=RandomForestClassifier(n_estimators=500, criterion="entropy") 
    imp=preprocessing.Imputer(missing_values=0,strategy="median",axis=0)
    imp.fit(train_data)
    train_data=imp.transform(train_data)
    test_data=imp.transform(test_data)
    #Pipeline([("imputer",preprocessing.Imputer(missing_values=0.0,strategy="median",axis=0)),("forest", RandomForestClassifier(n_estimators=500, criterion="entropy"))])
    #model=ExtraTreesClassifier(n_estimators=500,criterion="entropy")
    print '-------find optimze hyperparameter--------'
    model.C=35.33 #hyperparam(train_data,train_response,model)
    print '---- doing cv now ------------'
    cv_score= cv_loop(train_data, train_response, model)
    print 'cv mean AUC score= ',cv_score
    print '------ fitting full model -------------'
    model.fit(train_data, train_response)
    preds=model.predict_proba(test_data)[:,1]
    #print model.accuracy_score(test_scaled, test_response)
    print 'test data AUC score= ', metrics.roc_auc_score(test_response, preds)
    print '------------------------------'
    print 'entries in test data= ',len(preds)
    ct=0
    for pred in preds:
        #print pred
        if pred>=0.5: 
            ct+=1
    print 'ratio of no of 1s to total test size= ',  float(ct)/len(preds)
    ct=0      
    for i in xrange(len(test_response)):
        #print test_response[i], preds[i]
        if test_response[i]==1 and preds[i]>=0.5:
            ct+=1
        if test_response[i]==0 and preds[i]<0.5:
            ct+=1  
    print 'how many predictions match with test data= ',float(ct)/len(test_response)     
              
