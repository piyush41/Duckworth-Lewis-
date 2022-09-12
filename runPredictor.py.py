import math
import numpy as np

import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import matplotlib.pyplot as plt
    
    
    
def pre_process(df):

    #Use to preprocess the data
    data=df[df["Innings"]==1]
    data=data.sort_values(by=['Match',"Over"],ascending=True)
    gmatch=data.groupby("Match")
    count=0

    # Calculate the actual_total_runs and actual_total_innings from the data 
    actual_total_runs=[]
    actual_total_innings=[]
    match_to_remove=[]

    for i,j in gmatch:       

        sum=0
        for k in range(j.shape[0]):
            if k==0:
                if( j.iloc[k]['Over']!=1):
                    match_to_remove.append(i)
            sum=sum + j.iloc[k]["Runs"]
            actual_total_runs.append(sum)
        total_innings=[sum]*j.shape[0]
        actual_total_innings.extend(total_innings)


    actual_total_runs=np.array(actual_total_runs)
    actual_total_innings=np.array(actual_total_innings)
    actual_runs_remaining = actual_total_innings-actual_total_runs
    data.insert(5,'Actual_runs_remaining',actual_runs_remaining)
    data.insert(5,'Actual_total_runs',actual_total_runs)
    data.insert(5,'Actual_total_Innings',actual_total_innings)
    # Remove the matches in which the 50 overs has not finished and
    # wickets_remainings is greater than 0.
    mat=data[(data['Over']==50) | (data['Wickets.in.Hand']==0)]['Match'].unique()
    data=data[data['Match'].isin(mat)]
    mat=data[(data['Error.In.Data']==0)]["Match"].unique()
    data=data[data['Match'].isin(mat)]
    data=data[~data['Match'].isin(np.array(match_to_remove))]
   
    return data


def Loss_fn(params, args):
   
    L = params[10]
    innings=args[0]
    overs_remaining = args[1]
    runs_remaining = args[2]
    wickets_remaining = args[3]
    loss=0   
    
    for i in range(len(wickets_remaining)):
        predicted_run = params[wickets_remaining[i]-1] * (1.0 - np.exp((-1*L*overs_remaining[i])/(params[wickets_remaining[i]-1])))
        tmp=(predicted_run - runs_remaining[i])**2
        loss=loss+tmp

    return loss



def optimizer(method_name,innings,overs_remaining,runs_remaining,wickets_remaining):    
    

    W=[]
    # Take the mean the initial parameters for each of the wickets remaining repectively
    W.extend(list(data.groupby('Wickets.in.Hand')['Actual_runs_remaining'].mean()[1:]))
    W.append(1)
    print("The initial value of model parameters are ")
    for i in range(len(W)):
        if i==10:
            print("The value of L is",W[i])
        else:
            print("The value of Z["+str(i+1)+'] is ', W[i])
    
    param = minimize(Loss_fn, W,
                      args=[innings,
                            overs_remaining,
                            runs_remaining,
                            wickets_remaining
                            ],
                      method=method_name)
    
    return param['x'], param['fun']



def y_hat(Z, L, U):
    return Z * (1 - np.exp(-L*U/Z))

def plot():
    plt.figure(figsize=(10,7)) #Fig Size
    plt.xlabel('Overs remaining')
    
    plt.ylabel('Percentage(%) of resources remaining')
    plt.xlim((0, 50))
    plt.ylim((0, 100))
    plt.xticks([0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50])
    plt.yticks([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    highest = y_hat(final_param[9],final_param[10], 50)
    overs = np.arange(0, 51, 1)
    line=[]



    for i in range(10):
        
        y=100*(y_hat(final_param[i], final_param[10], overs)/highest)
        plt.plot(overs, y, label='Z['+str(i+1)+']')
        plt.legend()
    plt.show()
    
    
df=pd.read_csv("../Data/04_cricket_1999to2011.csv")
data=pre_process(df)
innings = data['Innings'].values
overs_completed = data['Over'].values
total_overs = data['Total.Overs'].values
overs_remaining = total_overs - overs_completed
innings_total_score = data['Actual_total_Innings'].values
current_score = data['Actual_total_runs'].values
runs_remaining=data['Actual_runs_remaining'].values

wickets_remaining = data['Wickets.in.Hand'].values

final_param, loss= optimizer('L-BFGS-B',innings,overs_remaining,runs_remaining,wickets_remaining)
print("The final value of model paramaters are")
for i in range(len(final_param)):
    if i==10:
        print("The value of L after optimization is",final_param[i])
    else:
        print("The value of Z["+str(i+1)+'] after optimization is ', final_param[i])
        
print("The loss value after Normalization is ",loss/data.shape[0])
plot()