import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.cbook as cbook
from scipy.stats import pearsonr
from scipy.stats import ttest_ind



def calculate_factors_NARS(data_frame_NARS):
#data_frame_NARS: a data frame of 14 columns

#NARs Summary Analysis:
#Sub-scale 1: Negative Attitudes toward Situations and Interactions with Robots. Items - 4,7,8,9,10,12
#Sub-scale 2: Negative Attitudes toward Social Influence of Robots. Items - 1,2,11,13,14
#Sub-scale 3: Negative Attitudes toward Emotions in Interaction with Robots. Items - 3r,5r,6r
#Syrdal, D. S., Dautenhahn, K., Koay, K. L., & Walters, M. L. (2009). The negative attitudes towards robots scale and
#reactions to robot behaviour in a live human-robot interaction study. Adaptive and Emergent Behaviour and Complex Systems.
# r=reversed

    NARS_facrots = pd.DataFrame()
    NARS_facrots["NARS_sub1"] = data_frame_NARS[[3,6,7,8,9,11]].mean(axis=1)
    NARS_facrots["NARS_sub2"] = data_frame_NARS[[0,1,10,12,13]].mean(axis=1)
    NARS_facrots["NARS_sub3"] = 6 - data_frame_NARS[[2,4,5]].mean(axis=1)

    return NARS_facrots


#Godspeed Summary Analysis
#The Godspeed questionnaires defined by Bartneck, Kulic, and Croft (2009) were used to assess the children’s impressions of
#the robot, going beyond the ones already covered in the ALMERE questionnaire.

#Anthropomorphism: Items 1,2,3,4,5
#Animacy: Items 6,7,8,9,10,11
#Likeability: Items 12,13,14,15,16
#Perceived Intelligence: 17,18,19,20,21
#Perceived Safety: 22,23,24
#Bartneck, C., Kulić, D., Croft, E., & Zoghbi, S. (2009). Measurement instruments for the anthropomorphism, animacy, likeability, perceived intelligence, and perceived safety of robots. International journal of social robotics, 1(1), 71-81.