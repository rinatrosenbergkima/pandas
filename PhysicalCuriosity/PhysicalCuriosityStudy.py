#tutorial from Tor:
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

df=pd.read_csv('data/AQ.csv', sep=',',header=None)
df.values


print(df.values)