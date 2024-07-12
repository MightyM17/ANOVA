import scipy.stats
import pandas as pd

# Function to preprocess data
def preprocecss(data):
    return data

# CBD
def anova_steps_cbd(data, a):
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
    
#RBD
def anova_steps_rbd(data, a):
    data = preprocecss(data)
    
    # Find mean of each group
    mean = data.mean()
    print(mean)
    
    row_means = data.mean(axis=1)
    print(row_means)
    
    # Total mean
    total_mean = mean.mean()
    print("Total Mean: ",total_mean)
    
    # Calculate SSC
    SSC = 0
    for i in data.columns:
        count = data[i].count()
        SSC += ((mean[i] - total_mean) ** 2)
    SSC = SSC * data.index.size
    print("SSC: ", SSC)
    df_c = data.columns.size - 1
    print("df_c: ", df_c)
    
    # Calculate SSR
    SSR = 0
    for i in data.index:
        SSR += ((row_means[i] - total_mean) ** 2)
    SSR = data.columns.size * SSR
    print("SSR: ", SSR)
    df_r = data.index.size - 1
    print("df_r: ", df_r)
    
    # Calculate SSE
    SSE = 0
    for i in data.index:
        for j in data.columns:
            SSE += (data.loc[i, j] - row_means[i] - mean[j] + total_mean) ** 2
    print("SSE: ", SSE)
    df_e = data.size - data.columns.size - data.index.size + 1
    print("df_e: ", df_e)
    
    # Calculate SST
    SST = SSC + SSR + SSE
    print("SST: ", SST)
    df_t = data.size - 1
    print("df_t: ", df_t)
        
    # Calculate MSC
    MSC = SSC / df_c
    print("MSC: ", MSC)
    
    # Calculate MSR
    MSR = SSR / df_r
    print("MSR: ", MSR)
    
    # Calculate MSE
    MSE = SSE / df_e
    print("MSE: ", MSE)
    
    # Calculate F-Value
    F_treat = MSC / MSE
    print("F_treat: ", F_treat)
    F_block = MSR / MSE
    print("F_block: ", F_block)
    
    # Get F-Critical Value
    F_critical_treat = scipy.stats.f.ppf(1-a, df_c, df_e)
    print("F Critical Treat: ", F_critical_treat)
    F_critical_block = scipy.stats.f.ppf(1-a, df_r, df_e)
    print("F Critical Block: ", F_critical_block)

    return mean, total_mean, SSC, df_c, SSR, df_r, SSE, df_e, SST, df_t, MSC, MSR, MSE, F_treat, F_block, F_critical_treat, F_critical_block