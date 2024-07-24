import scipy.stats
import pandas as pd

# CBD
def anova_steps_cbd(data, a):
    
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

#FD
def anova_steps_fd(data, a, n, row_treat):
    col_treat = data.columns.size
    N = data.size
    
    df_r = row_treat - 1
    print("df_r: ", df_r)
    df_c = col_treat - 1
    print("df_c: ", df_c)
    df_i = df_r * df_c
    print("df_i: ", df_i)
    df_e = row_treat * col_treat * (n - 1)
    print("df_e: ", df_e)
    df_t = N - 1
    print("df_t: ", df_t)
    
    # Grand Total
    GT = data.sum().sum()
    corr_fact = (GT ** 2) / N
    
    print("GT: ", GT)
    print("corr_fact: ", corr_fact)
    
    # SST
    SST = (data ** 2).sum().sum() - corr_fact
    print("SST: ", SST)
    
    # SSC
    column_sums = data.sum(axis=0)
    SSC = (column_sums ** 2).sum() / (n*row_treat) - corr_fact
    print("SSC: ", SSC)
    
    # SSR
    total_rows = len(data)
    block_sum = [0] * (total_rows // n)
    i=0
    for start_row in range(0, total_rows, n):
        end_row = start_row + n
        block_sum[i] = data.iloc[start_row:end_row].sum().sum()
        i+=1
    print(f"Sum of rows: {block_sum}")
    SSR=0
    for i in range(len(block_sum)):
        SSR = SSR + (block_sum[i] ** 2)/(n*data.columns.size)
    SSR = SSR - corr_fact
    print("SSR: ", SSR)
    
    # SSI
    block_sums = {col: [] for col in data.columns}  
    for col in data.columns:
        for start_row in range(0, len(data), n):
            end_row = start_row + n
            block_sum = data[col][start_row:end_row].sum()
            block_sums[col].append(block_sum)
    
    print("Block sums: ", block_sums)

    SSI = 0
    for col, sums in block_sums.items():
        for sum_val in sums:
            SSI += (sum_val ** 2) / (n)
    SSI = SSI - corr_fact - SSC - SSR
    print("SSI: ", SSI)
    
    # SSE
    SSE = SST - SSI - SSC - SSR
    print("SSE: ", SSE)
    
    # Calculate MSR
    MSR = SSR / df_r
    print("MSR: ", MSR)
    
    # Calculate MSC
    MSC = SSC / df_c
    print("MSC: ", MSC)
    
    # Calculate MSI
    MSI = SSI / df_i
    print("MSI: ", MSI)
    
    # Calculate MSE
    MSE = SSE / df_e
    print("MSE: ", MSE)
    
    # Calculate F-Value
    F_r = MSR / MSE
    print("F_r: ", F_r)
    F_c = MSC / MSE
    print("F_c: ", F_c)
    F_i = MSI / MSE
    print("F_i: ", F_i)
    
    # Get F-Critical Value
    F_critical_row = scipy.stats.f.ppf(1-a, df_r, df_e)
    print("F Critical Treat: ", F_critical_row)
    F_critical_col = scipy.stats.f.ppf(1-a, df_c, df_e)
    print("F Critical Block: ", F_critical_col)
    F_critical_iter = scipy.stats.f.ppf(1-a, df_i, df_e)
    print("F Critical Iter: ", F_critical_iter)
    
    return SST, df_t, SSC, df_c, SSR, df_r, SSI, df_i, SSE, df_e, MSR, MSC, MSI, MSE, F_r, F_c, F_i, F_critical_row, F_critical_col, F_critical_iter

# daaa = pd.DataFrame({'col1': [2,1,2,1,2,3,1,2], 'col2': [2,3,3,2,3,3,2,4], 'col3':[4,3,4,3,4,4,3,4]})
# no_of_groups = 4
# anova_steps_fd(daaa, 0.05, 4, 2)