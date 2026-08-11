import numpy as np 
import pandas as pd
import matplotlib.pylab as plt 
import seaborn as sns 
import plotly as ep
from pathlib import Path

def overview(df):
    'overview of the dataset'
    df.head()
    return df



if __name__=='__main__':
    path=Path('./hotel_bookings.csv')
    df=pd.read_csv(path)
    overview(df)