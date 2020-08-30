#!/usr/bin/env python´
# coding: utf-8

# In[17]:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# import dask.dataframe as dd
import time
start_time = time.time()

df = pd.read_csv('/Users/vineethdasary/Documents/flask-app/SampleInstPerf.csv')
print(df)


# In[18]: 


df.drop(columns = ['SUBJECT', 'CATALOG_NBR'], inplace = True, axis = 1)
print(df)


# In[19]:


# Importing module and initializing setup
from pycaret.regression import *
reg1 = setup(data = df, target = 'Average_Point', train_size = 0.6)


# In[20]:


# comparing all models
allmodels = compare_models(n_select=5)


# In[ ]:


cbr = create_model('llar')


# In[ ]:


print(cbr)


# In[ ]:


# tune model
tcbr = tune_model(cbr)

# In[ ]:
print(tcbr)
# In[ ]:

    
# ensembling decision tree model (boosting)
btcbr = ensemble_model(tcbr, method = 'Bagging')


# In[ ]:


blender = blend_models(estimator_list=[btcbr])
print(blender)

# In[ ]:

print('belnder plot\n')
plot_model(estimator=blender, plot='residuals')
plt.show()
# In[ ]:
plot_model(estimator=blender, plot='error')
plt.show()
# In[ ]:
plot_model(estimator=blender, plot='cooks')
plt.show()
# In[ ]:
plot_model(estimator=blender, plot='learning')
plt.show()
# In[ ]:
plot_model(estimator=blender, plot='manifold')
plt.show()


# In[ ]:
final_llar_model = finalize_model(blender)
# print('intrpret model\n')
# interpret = interpret_model(blender)

# In[ ]:
save_model(final_llar_model, 'llar_model')
# In[ ]:
print('predict model on test data\n')
# In[ ]:
predtest = predict_model(blender)
print('prediction results\n')
print(predtest)

# In[ ]:
data_unseen = pd.read_csv('/Users/vineethdasary/Documents//Capstone/SampleInstPerf.csv') 
# generate predictions on unseen data
predictions = predict_model(final_llar_model, data = data_unseen)
print('predictions on unseen data\n')
print(predictions)

df = pd.DataFrame(predictions)
df.to_csv('/Users/vineethdasary/Documents/Capstone/SampleInstPerf1.csv')


# %%
print("--- %s seconds ---" % (time.time() - start_time))
# %%


# %%
