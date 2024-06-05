import scipy.stats
import pandas as pd

# Function to preprocess data
def preprocecss(data):
    return data

# CBD
def anova_steps(data, a):
    data = preprocecss(data)
    
    # Find mean of each group
    mean = data.mean()
    print(mean)
    
    # Total mean
    total_mean = mean.mean()
    print("Total Mean: ",total_mean)
    
    # Calculate SSC
    SSC = 0
    for i in data.columns:
        count = data[i].count()
        SSC += count * ((mean[i] - total_mean) ** 2)
    print("SSR: ", SSC)
    df_c = data.columns.size - 1
    print("df_c: ", df_c)
    
    #Caclulate SSE
    SSE = 0
    for i in data.columns:
        for j in data[i]:
            SSE += (j - mean[i]) ** 2
    print("SSE: ", SSE)
    df_e = data.size - data.columns.size
    print("df_e: ", df_e)
    
    #Calculate SST
    SST = SSC + SSE
    print("SST: ", SST)
    df_t = data.size - 1
    print("df_t: ", df_t)
    
    #Calculate MSC
    MSC = SSC / df_c
    print("MSC: ", MSC)
    
    #Calculate MSE
    MSE = SSE / df_e
    print("MSE: ", MSE)
    
    #Calculate F-Value
    F = MSC / MSE
    print("F: ", F)
    
    #Get F-Critical Value
    F_critical = scipy.stats.f.ppf(1-a, df_c, df_e)
    print("F Critical: ", F_critical)
    
    return mean, total_mean, SSC, df_c, SSE, df_e, SST, df_t, MSC, MSE, F, F_critical
    
# data = pd.read_csv('data.csv')
# anova_steps(data)