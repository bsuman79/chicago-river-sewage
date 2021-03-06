import pandas as pd
import calendar
import re

pieces = []

months = [calendar.month_name[i].lower() for i in range(1, 13)]
years = range(2005, 2013)

for year in years:
    for month in months:
        path = '%s/%s%s.csv' % (year, month, year)
        frame = pd.read_csv(path)
        columns_to_remove = (x for x in frame.columns if (re.search('^\(\)\.\d+$|^\(\)$|^$|Unnamed|code|qual_data|bod|time',x))) # Remove columns that don't have headers, can't join correctly or really tell what the data is
        for i in columns_to_remove:
            frame = frame.drop(i,1)
        pieces.append(frame)
        print('%s/%s%s.csv' % (year, month, year))
flat = pd.concat(pieces, ignore_index=True, join='outer')

location_frame = pd.read_csv('waterwayslocations.csv')
flat = flat.merge(location_frame, how='left', on='sampling(point)')

flat.to_csv("clean-waterway-measurements.csv")
