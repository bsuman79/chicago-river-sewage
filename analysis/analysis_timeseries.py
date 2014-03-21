__author__ = 'suman bhattacharya'
"""
Goal: given the water, rain and sewage measurements of the chicago river at different locations
and at various times over the past few years, can we predict which locations are less/more polluted in future?

Description: The code has two parts- data wrangling part and the analysis part. For the data wrangling part,
 the code reads in 3 files containing measurements of clean-water, rain fall and cso-events
  (note that the rain fall and the cso files have been modified to include the lat/long information
  , see data_wrangling.py), then for each location and date of the clean-water, the nearest cso-event location
  and the rainfall location is found. Also the  date of rain-fall/cso-event >0 that are closest to each
  clean-water date is found. The end result is out.csv file that contain the clean-water measurements plus
   the rain-fall (number of days before the clean-water) and the quantity plus cso event (same columns as
   rainfall). Note that we are always interested to predict the 1 month into the future (or the next event), the data
   (both past and future) is prepared such that the outcome is offset by 1 month.

   The next part of the code deals with the time series analysis of the chicago river. We divide the data
   into test and train set at a particular date. E.g. if datetosplit=2010-01-01, all data before that (aka past data)
   will be used for  training , and the data from 2010-01-01 to next 6 weeks (roughly 1 month) (aka future data)
   will be used for testing. We then loop over the datatosplit advancing by 1 month, so the next date would be 2010-02-01
   and so on. We then repeat these steps over the past 4 years from 2012-2009. The final outcome is a set of locations that
   are most predictable over the past 4 years month after month both in terms of pollution. We then show these places in the
   Chicago map.


"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import datetime
from sklearn import preprocessing, svm, metrics, cross_validation, linear_model
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
import orange
from collections import defaultdict, OrderedDict
from dateutil.relativedelta import relativedelta



def date_to_days(dates, divide_by=30):
  """
  take list of dates and convert them to months (roughly where month=30 days)

  """
  list_of_days=[]
  for date in dates:
     d0 = datetime.date(date.year, date.month, date.day)
     d1 = datetime.date(date.year, 01, 01)
     delta = d0 - d1
     list_of_days.append(np.floor(delta.days/divide_by))
  return list_of_days


def add_months(sourcedate, months):
    """
    take a source date and add a month to that, then convert it back to the  date format
    """
    month = sourcedate.month - 1 + months
    year = sourcedate.year + month / 12
    month = month % 12 + 1
    day = min(sourcedate.day, calendar.monthrange(year, month)[1])
    return datetime.date(year, month, day)

def read_mwrd_rain(directory='', file='mwrd_rain_measurements.csv'):
    """
    read mwrd_rain file, slice and dice the data such that it can be included to the main clean-water
    measure file. also compute the monthly mean and median rainfall measurements such that it can be added to
    clean water dates when the rain fall data dont exist.
    """
    data = pd.read_csv(directory + file)
    del data['Gauge Number']
    del data['Gauge Name']
    del data['Address']
    data['duration'] = data['Rain (inches)']
    data['start'] = data['Date']
    data = data[(np.isfinite(data['north_latitude'])) & (np.isfinite(data['west_longitude']))]
    data = data[data['duration']>0]
    dates = list(data['Date'])
    dates = [datetime.datetime.strptime(date.split()[0],'%Y-%m-%d') for date in dates]
    data['months'] = date_to_days(dates)
    rain_monthly_median = defaultdict(int)
    data_grouped= data.groupby(['months'])
    for key,val in data_grouped:
         rain_monthly_median[key] = (val['duration'].median(),val['duration'].mean())
    return data, rain_monthly_median

def read_cso_events(directory='', file='cso_events_timestamped.csv'):
    """
    read cso file, do the same with cso file as with the rain file
    """
    data = pd.read_csv(directory + file)
    data = data[(np.isfinite(data['north_latitude'])) & (np.isfinite(data['west_longitude']))]
    data['duration']= (pd.to_datetime(data['end'])-pd.to_datetime(data['start'])).astype(int)/60.0/1e9
    dates = list(data['start'])
    dates = [datetime.datetime.strptime(date.split()[0],'%Y-%m-%d') for date in dates]
    data['months'] = date_to_days(dates)
    del data['location']
    del data['end']
    duration_monthly_median = defaultdict(int)
    data_grouped= data.groupby(['months'])
    for key,val in data_grouped:
         duration_monthly_median[key] = (val['duration'].median(),val['duration'].mean())
    return data, duration_monthly_median

def get_cso_rain_duration(data, data_cso, monthly_duration):
    """
    this method reads in clean-water and cso/rain data, group them by location. Then find the closest rain gauge/cso
     location to the clean-water location, find the latest date the rain/cso happened , if found attach that rain/cso
      measurement to clean-water measuement, if not then use the monthly median/mean
    """
    data_cso_grouped = data_cso.groupby(['north_latitude', 'west_longitude'])
    data_grouped = data.groupby(['north_latitude', 'west_longitude'])
    cso_duration_full=defaultdict(float)
    for loc,val in  data_grouped:
       cso_duration, days_between_cso_water=[],[]
       min_dist=1e6
       min_loc1= 0.0
       for loc1,val1 in data_cso_grouped:
           dist= (loc[0]-loc1[0])**2+(loc[1]-loc1[1])**2
           if min_dist > dist:
              min_dist= dist
              min_loc1=loc1 # find approx nearest cso location to water measurement loc
       for date in val['collect(date)']: # for each measurement loc loop over collect date
           min=1e6 # initialize a min value
           for date_cso in data_cso_grouped.get_group(min_loc1)['start']: # find the closest cso date to measure date
                date_cso_tmp = str(date_cso).split()[0]
                date_tmp= str(date).split()[0] #, date - datetime.timedelta(weeks=6)
                date_delta = datetime.datetime.strptime(date_tmp, '%Y-%m-%d') - datetime.timedelta(weeks=12)
                date_tmp = datetime.datetime.strptime(date_tmp,'%Y-%M-%d')
                date_cso_tmp = datetime.datetime.strptime(date_cso_tmp,'%Y-%M-%d')

                if (date_cso_tmp < date_tmp) and (date_cso_tmp >= date_delta):
                    diff=(datetime.date(date_tmp.year, date_tmp.month, date_tmp.day)-
                          datetime.date(date_cso_tmp.year, date_cso_tmp.month, date_cso_tmp.day)).days
                    if min> diff:
                          min=diff
                          min_date= date_cso

           if min ==1e6: # if closest cso date not found , assign average to the measure
                 month,=val['months'][val['collect(date)']==date]
                 cso_duration.append(monthly_duration[month][1])
                 days_between_cso_water.append(np.nan)
           else:
                 csoduration = data_cso_grouped.get_group(min_loc1)[['start','duration']]
                 cso_duration.append(list(csoduration[csoduration['start']==min_date].ix[:,1])[0])
                 days_between_cso_water.append(min)


       ind= list(val.ix[:,0])
       for i in xrange(len(ind)):
           cso_duration_full[ind[i]]= (cso_duration[i],days_between_cso_water[i])

    return cso_duration_full

def read_water_measurements(directory='../data/', file='clean-waterway-measurements.csv'):
    """
    reads in water measurements, call the previous two methods to get the nearest location, closest date
    to the clean-water measurements, add the rain,cso columns to clear water, write the new dataset to a file.
    (useful to do this to not repeat this step for every datetosplit).
    """

    data = pd.read_csv(directory + file)

    data.replace(to_replace=['<', '\*', '\'', '_____', '3..1', 'n/a', 't/x'],
                 value=['', '', '', np.nan, np.nan, np.nan, np.nan], inplace=True, regex=True)
    data.replace(to_replace=['na'], value=[np.nan], inplace=True)
    data['fec_col_cat'] = data['fec_col(cts/100ml)']
    data['fec_col_cat'][np.array(data['fec_col(cts/100ml)'], dtype=float) < 200] = 1
    data['fec_col_cat'][np.array(data['fec_col(cts/100ml)'], dtype=float) >= 400] = 0
    data['fec_col_cat'][(np.array(data['fec_col(cts/100ml)'], dtype=float) < 400) & (
        np.array(data['fec_col(cts/100ml)'], dtype=float) >= 200)] = 1

    data= data[data['collect(date)']==data['collect(date)']]
    dates = pd.DatetimeIndex(list(data['collect(date)']))

    data['months'] = date_to_days(dates)
    data['west_longitude']= -1*data['west_longitude'] # fix the longitude sign


    data['cso_duration'] = np.nan
    data['cso_days'] = np.nan
    data_cso, monthly_duration=read_cso_events()
    cso_duration= get_cso_rain_duration(data,  data_cso, monthly_duration)
    for key,val in cso_duration.items():
        data['cso_duration'][key] = val[0]
        data['cso_days'][key] = val[1]

    #print set(data['cso_duration']), set(data['cso_days'])

    data['rain_measure'] = np.nan
    data['rain_days'] = np.nan
    data_cso, monthly_duration=read_mwrd_rain()
    cso_duration= get_cso_rain_duration(data,  data_cso, monthly_duration)
    for key,val in cso_duration.items():
        data['rain_measure'][key] = val[0]
        data['rain_days'][key] = val[1]

    del data['Unnamed: 0']
    data.to_csv('out.csv')
    exit()
    #print set(data['rain_measure']), set(data['rain_days'])
    #return 0

def split_data(file='out.csv', datetosplit=''):
    """
    split the dataset to past and future one month
    """
    data = pd.read_csv(file)
    format = '%Y-%m-%d'
    next_month = datetime.datetime.strptime(datetosplit, format) + datetime.timedelta(weeks=6)
    # create train data, take data till datetosplit
    data_split = data[data['collect(date)'] < datetosplit]
    # create test data, take data start from datetosplit
    data_split_test = data[(data['collect(date)'] >= datetosplit) & (data['collect(date)'] < str(next_month))]
    #print data.ix[:,30:40][(data['collect(date)']>='2012-07-30') & (data['collect(date)']<'2012-07-31')].fillna(0.0)
    return data_split, data_split_test


def prepare_data(data):
    """
    make the data time-series-y, namely, offset the response (y) by one month both in the train and the test data
    compared to X (features )

    """
    data_grouped = data.groupby(['north_latitude', 'west_longitude'])
    group = {}
    ct = 1
    data_full, response_full = [], []
    for key, val in data_grouped:
          val = val.fillna(np.nan)
          group[key] = val
          data2 = np.hstack((val.ix[:,1:19], val.ix[:,20:52], val.ix[:,53:69],
                             val.ix[:,73:78],  val.ix[:,69:72]))
          data2=data2[:-1]
          data_full.append(data2)
          response_full.extend(val['fec_col_cat'][1:])
          #print data_full,response_full
          #print "---------location= ",ct," done"
          ct += 1
    return np.vstack(data_full), response_full

def scale_data(train_data, test_data):
    """
    needed to scale the data if using svm
    """
    scaler= preprocessing.MinMaxScaler().fit(train_data)
    #print scaler.mean_, scaler.std_
    return scaler.fit_transform(train_data), scaler.transform(test_data)

def dates_to_split(years_to_subtract):
    """
    return the list of 12 months to loop over by subtracting n years from 2012, used to loop over years
    """
    dates=['2012-01-01','2012-02-01','2012-03-01','2012-04-01','2012-05-01','2012-06-01',
                    '2012-07-01','2012-08-01','2012-09-01','2012-10-01','2012-11-01','2012-12-01']
    dates_new=[]
    for date in dates:
        dates_new.append(str(int(str(date)[0:4])-years_to_subtract)+date[4:])
    return dates_new

def predict_by_location(years_to_avg):
    """
    the main method that does most of the analysis work, declare which ML algorithm to use, then predict by location
    loop over years and months of a year, predict by month-to-month. return aggregated prediction of each location by
    years and months.
    """
    model=RandomForestClassifier(n_estimators=100, criterion="entropy")
    #model=svm.SVC(probability=True)
    #model = ExtraTreesClassifier(n_estimators=250, random_state=0, criterion='entropy', n_jobs=-1)
    imp=preprocessing.Imputer(missing_values=np.nan, strategy="median",axis=0)
    prediction_by_location=defaultdict(list)
    location_lat_long=defaultdict(float)
    for year in xrange(years_to_avg):
        for date_to_split in dates_to_split(year):
            print "------------------------------------------"
            print "date to split train and test= ",date_to_split
            print "-----------------------------------------"
            data,data1=split_data(datetosplit=date_to_split)
            train_data, train_response = prepare_data(data)
            test_data, test_response = prepare_data(data1)
            print test_data.shape,train_data.shape
            long_test= list(test_data[:,test_data.shape[1]-1])
            lat_test= list(test_data[:,test_data.shape[1]-2])
            locations_test= list(test_data[:,test_data.shape[1]-3])
            imp.fit(train_data[:,:-3])
            train_data=imp.transform(train_data[:,:-3])
            test_data=imp.transform(test_data[:,:-3])

            #train_data,test_data= scale_data(train_data.astype(float),test_data.astype(float))

            print '------ fitting full model -------------'
            model.fit(train_data, train_response)
            preds=model.predict_proba(test_data)[:,1]
            preds_binary=model.predict(test_data)
            #print model.accuracy_score(test_scaled, test_response)
            #auc_score.append(metrics.roc_auc_score(test_response, preds))
            #print '------------------------------'
            #print 'entries in test data= ',len(preds) #, preds

            #print 'ratio of no of 1s PREDICTED to total test size= ', 1.0*len(list(preds[preds>=0.5]))/len(p
            ct,ct1=0,0
            for i in xrange(len(test_response)):
                  prediction_by_location[locations_test[i]].append(
                      (test_response[i], preds[i],preds_binary[i]))
                  location_lat_long[locations_test[i]]=(lat_test[i],long_test[i])
                  if test_response[i]==1 and preds[i]>=0.5:
                      ct += 1
                      ct1 += 1
                  if test_response[i]==0 and preds[i]<0.5:
                      ct+=1
                  #print 'ratio of no of ACTUAL 1s to total test size= ',float(ct1)/len(test_response)
                  #print 'how many predictions match with test data= ',float(ct)/len(test_response)
    return prediction_by_location, location_lat_long

def est_accuracy_by_location(prediction_by_location):
    """
    accuracy of each location summed over years and months
    """
    accuracy_by_location=defaultdict(float)
    test_response1, train_response1=[],[]
    for loc, preds in prediction_by_location.items():
        count,count_test_1=0,0
        for pred in preds:
           if pred[0]==pred[0]:
              test_response1.append(pred[0])
              train_response1.append(pred[1])
           if pred[0]==pred[2]:
                count+=1
           if pred[0]==1:
               count_test_1+=1
        if len(preds)>5:
             accuracy_by_location[loc]=(1.0*count/len(preds),len(preds),metrics.roc_auc_score(test_response1,
                                                                train_response1),1.0*count_test_1/len(preds))
    return accuracy_by_location



if __name__ == "__main__":
    """ the main driver routine, calls read_water_measurements, sets how many years to avg , get prediction
     by location , then accuracy by location, then print out the locations which matches the ground
      truth and predictions> 70% of the time and have AUC > 0.75"""
    # set some basic parameters here
    years_to_avg=4
    how_many_match=0.7
    AUC_threshold=0.75

    #read_water_measurements() # comment out if you already have out.csv
    prediction_by_location, location_lat_long=predict_by_location(years_to_avg)
    accuracy_by_location=est_accuracy_by_location(prediction_by_location)
    print "-------------------------------"
    accurate_locations= list(OrderedDict(sorted(accuracy_by_location.items(),key=lambda x: x[1][0])))[::-1]

    ct=0
    for loc in accurate_locations:
        if accuracy_by_location[loc][0]>how_many_match and accuracy_by_location[loc][2]>AUC_threshold:
            print ct, loc, location_lat_long[loc], accuracy_by_location[loc] #,prediction_by_location[loc]
            print "---------------------------------"
            ct+=1
