#tutorial:
#https://github.com/QuantScientist/Deep-Learning-Boot-Camp/blob/master/day01/Intro_ml_models.ipynb
#https://github.com/QuantScientist/Deep-Learning-Boot-Camp/blob/master/day01/predicting_income_from_census_income_data.ipynb


# > this is how to install pandas
# > sudo easy_install pip
# > pip install wheel
# > pip install pandas


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook


df_AQ  = pd.read_csv('data/AQ.csv', sep=',')
df_BFI = pd.read_csv('data/BFI.csv', sep=',')
df_demographics = pd.read_csv('data/demographics.csv', sep=',')
df_tablet = pd.read_csv('data/tablet.csv', sep=',')
df_summary = pd.DataFrame ()

def process_BFI():
    # Reverse BFI:
    df_BFI["BFI_6r"] = 6 - df_BFI["BFI_6"]
    df_BFI["BFI_21r"] = 6 - df_BFI["BFI_21"]
    df_BFI["BFI_9r"] = 6 - df_BFI["BFI_9"]
    df_BFI["BFI_24r"] = 6 - df_BFI["BFI_4"]
    df_BFI["BFI_34r"] = 6 - df_BFI["BFI_34"]
    df_BFI["BFI_2r"] = 6 - df_BFI["BFI_2"]
    df_BFI["BFI_12r"] = 6 - df_BFI["BFI_12"]
    df_BFI["BFI_27r"] = 6 - df_BFI["BFI_27"]
    df_BFI["BFI_37r"] = 6 - df_BFI["BFI_37"]
    df_BFI["BFI_8r"] = 6 - df_BFI["BFI_8"]
    df_BFI["BFI_18r"] = 6 - df_BFI["BFI_18"]
    df_BFI["BFI_23r"] = 6 - df_BFI["BFI_23"]
    df_BFI["BFI_43r"] = 6 - df_BFI["BFI_43"]
    df_BFI["BFI_35r"] = 6 - df_BFI["BFI_35"]
    df_BFI["BFI_41r"] = 6 - df_BFI["BFI_41"]

    # calculate the big 5 factors:
    df_BFI["BFI_extraversion"] = df_BFI[["BFI_1","BFI_6r","BFI_11","BFI_16","BFI_21r","BFI_26","BFI_31","BFI_36"]].mean(axis=1)
    df_BFI["BFI_neuroticism"] = df_BFI[["BFI_4","BFI_9r","BFI_14","BFI_24r","BFI_29","BFI_34r","BFI_39"]].mean(axis=1)
    df_BFI["BFI_agreeableness"] = df_BFI[["BFI_2r","BFI_7","BFI_12r","BFI_17","BFI_22","BFI_27r","BFI_32","BFI_37r","BFI_42"]].mean(axis=1)
    df_BFI["BFI_concientiousness"] = df_BFI[["BFI_3","BFI_8r","BFI_13","BFI_18r","BFI_23r","BFI_28","BFI_33","BFI_38","BFI_43r"]].mean(axis=1)
    df_BFI["BFI_openness"] = df_BFI[["BFI_5","BFI_10","BFI_15","BFI_20","BFI_25","BFI_30","BFI_35r","BFI_40","BFI_41r","BFI_44"]].mean(axis=1)


def process_AQ():
    # reverse AQ (Autism Spectrum Quotient Questions)
    ## http://aspergerstest.net/interpreting-aq-test-results/

    df_AQ["AQ_3"] = 6 - df_AQ["AQ_3"]
    df_AQ["AQ_8"] = 6 - df_AQ["AQ_8"]
    df_AQ["AQ_10"] = 6 - df_AQ["AQ_10"]
    df_AQ["AQ_11"] = 6 - df_AQ["AQ_11"]
    df_AQ["AQ_14"] = 6 - df_AQ["AQ_14"]
    df_AQ["AQ_15"] = 6 - df_AQ["AQ_15"]
    df_AQ["AQ_17"] = 6 - df_AQ["AQ_17"]
    df_AQ["AQ_24"] = 6 - df_AQ["AQ_24"]
    df_AQ["AQ_25"] = 6 - df_AQ["AQ_25"]
    df_AQ["AQ_27"] = 6 - df_AQ["AQ_27"]
    df_AQ["AQ_28"] = 6 - df_AQ["AQ_28"]
    df_AQ["AQ_29"] = 6 - df_AQ["AQ_29"]
    df_AQ["AQ_30"] = 6 - df_AQ["AQ_30"]
    df_AQ["AQ_31"] = 6 - df_AQ["AQ_31"]
    df_AQ["AQ_32"] = 6 - df_AQ["AQ_32"]
    df_AQ["AQ_34"] = 6 - df_AQ["AQ_34"]
    df_AQ["AQ_36"] = 6 - df_AQ["AQ_36"]
    df_AQ["AQ_37"] = 6 - df_AQ["AQ_37"]
    df_AQ["AQ_38"] = 6 - df_AQ["AQ_38"]
    df_AQ["AQ_40"] = 6 - df_AQ["AQ_40"]
    df_AQ["AQ_44"] = 6 - df_AQ["AQ_44"]
    df_AQ["AQ_47"] = 6 - df_AQ["AQ_47"]
    df_AQ["AQ_48"] = 6 - df_AQ["AQ_48"]
    df_AQ["AQ_49"] = 6 - df_AQ["AQ_49"]
    df_AQ["AQ_50"] = 6 - df_AQ["AQ_50"]

    # Definitely agree or Slightly agree responses to questions 1, 2, 4, 5, 6, 7, 9, 12, 13, 16, 18, 19, 20, 21, 22, 23, 26, 33, 35, 39, 41, 42, 43, 45, 46 score 1 point.
    # Definitely disagree or Slightly disagree responses to questions 3, 8, 10, 11, 14, 15, 17, 24, 25, 27, 28, 29, 30, 31, 32, 34, 36, 37, 38, 40, 44, 47, 48, 49, 50 score 1 point.

    for column in df_AQ.iloc[:,1:51]:
        df_AQ[column] = (df_AQ[column]>3)*1 # Give one point to questions who score less than 3

    df_AQ["AQ_total"] = df_AQ.iloc[:,1:51].sum(axis=1)


def create_df_summary():
    # create data frame with the important data
    df_summary["id"] = df_AQ["id"]
    df_summary["demographics_age"] = df_demographics["age"]
    df_summary["demographics_gender"] = df_demographics["gender"]
    df_summary["demographics_grades"] = df_demographics["grades"]
    df_summary["demographics_psychometrics"] = df_demographics["psychometrics"]
    df_summary["demographics_control_robot"] = df_demographics["control_robot"]
    df_summary["demographics_q1"] = df_demographics["q1"]
    df_summary["demographics_q2"] = df_demographics["q2"]
    df_summary["demographics_q3"] = df_demographics["q3"]
    df_summary["tablet_transition_entropy"] = df_tablet["transition_entropy"]
    df_summary["tablet_multi_discipline_entropy"] = df_tablet["Multi_discipline_entropy"]
    df_summary["tablet_multi_discipline_entropy"] = df_tablet["Multi_discipline_entropy"]
    df_summary["tablet_psycholetrics"] = df_tablet["PSY"]
    df_summary["tablet_normalized_total_listenning_time"] = df_tablet["normalized_total_listenning_time"]
    df_summary["BFI_extraversion"] = df_BFI["BFI_extraversion"]
    df_summary["BFI_neuroticism"] = df_BFI["BFI_neuroticism"]
    df_summary["BFI_agreeableness"] = df_BFI["BFI_agreeableness"]
    df_summary["BFI_concientiousness"] = df_BFI["BFI_concientiousness"]
    df_summary["BFI_openness"] = df_BFI["BFI_openness"]
    df_summary["AQ_total"] = df_AQ["AQ_total"]
    #print(df_summary)

def correlation_matrix(df,title):
    from matplotlib import pyplot as plt
    from matplotlib import cm as cm
    print("correlation_matrix")
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    cmap = cm.get_cmap('jet', 10)
    cax = ax1.imshow(df.corr(), interpolation="nearest", cmap=cmap)
    ax1.grid(True)
    plt.title(title)
    labels=list(df) #the dataframe headers
    print(labels)
    ax1.set_xticklabels(labels,fontsize=4, rotation='vertical')
    ax1.set_yticklabels(labels,fontsize=4)
    # Add colorbar, make sure to specify tick locations to match desired ticklabels
    fig.colorbar(cax, ticks=[0,.05,.10,.15,.20,.25,.30,.35,.40,.45,.50,.55,.60,.65,.70,.75,.8,.85,.90,.95,1])
    plt.show()

def correlation_summary(df):
    corr = df.corr()
    print(corr)
    plt.matshow(corr)
    headers = list(df_summary)
    x_pos = np.arange(len(headers))
    plt.xticks(x_pos, headers, rotation='vertical', fontsize=4)
    y_pos = np.arange(len(headers))
    plt.yticks(y_pos, headers, fontsize=4)
    plt.show()


def plot_correlations(x,y):
    #plt.plot(df_summary["demographics_psychometrics"], df_summary["BFI_extraversion"], 'ro')
    #plt.axis([400, 800, 0, 5])
    #plt.show()
    fig, ax = plt.subplots()
    idx = np.isfinite(x) & np.isfinite(y)
    fit = np.polyfit(x[idx], y[idx], deg=1)
    ax.plot(x[idx], fit[0] * x[idx] + fit[1], color='red')
    ax.scatter(x, y)
    fig.show()
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

process_BFI()
process_AQ()
create_df_summary()
#correlation_summary(df_summary)
#correlation_matrix(df_summary,"correlations")

plot_correlations(df_summary["demographics_psychometrics"],df_summary["BFI_extraversion"])


#plt.plot(df_summary["demographics_grades"],df_summary["BFI_extraversion"],  'ro')
#plt.plot(df_summary["BFI_concientiousness"],df_summary["tablet_normalized_total_listenning_time"], 'ro')



