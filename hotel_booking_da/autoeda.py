# auto eda an bird overview of datasets
import pandas as pd 
from pathlib import Path
from ydata_profiling import ProfileReport


#load the datset 
path=Path('./hotel_bookings.csv')
# df 
df=pd.read_csv(path)

#auto eda 
profile=ProfileReport(df,title="eda auto report")

profile.to_file("autoeda.html")
