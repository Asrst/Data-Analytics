import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from scipy.stats import pointbiserialr, spearmanr 

# null hypothesis H0 = there is no correlation between the two variables
# p < 0.05 reject null hypothesis

def test_correlation(x, y, method = 'spearman'):
    
    """
    function to test spearman/pointbiseralr correlation b/w two variables.
    
    :param x: Pandas series
    :param y: Pandas series
    :param method: choose from 'spearman' (spearmanr) or 'pbs'(pointbiserialr)
    """
    
    # spearman rank correlation is robust to outliers due to ranking
    plt.figure(figsize = (6,4))
    plt.scatter(x, y)
    plt.title('Scatter plot b/w {} & {}'.format(x.name, y.name))
    plt.xlabel(x.name)
    plt.ylabel(y.name)
    plt.show()
    
    if method == 'spearman':
        result = spearmanr(x, y)
    elif method == 'pbs':
        result = pointbiserialr(x, y)
        
    return result



if __name__ == "__main__": 
    
    # read the csv
    df = pd.read_csv('telecom_battery.csv')
  
    # converting to seconds (standard posix time)
    df['Timestamp'] = df['Timestamp'].apply(lambda x: x/1000)
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit = 's')

    diff = df['Timestamp'].values[2] - df['Timestamp'].values[1]
    print('time delta for each observation(row):', diff.astype('timedelta64[s]'))

    result = test_correlation(df['Grid status'], df['SOC'], method = 'pbs')
    print('correlation b/w Grid status v/s SOC', result)

    result = test_correlation(df['Equivalent cycle'], df['SOH'], method = 'spearman')
    print('correlation b/w Equivalent cycle v/s SOH', result)


    result = test_correlation(df['SOC'], df['Temperature'], method = 'spearman')
    print('correlation b/w SOC v/s Temperature', result)
