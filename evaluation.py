#!/usr/bin/env python
# coding: utf-8

# # Build evaluation tables for paper
# 
# - Precomputed results must be located in `experiments_output` directory.

# In[1]:


import json
import os
import pickle
import pandas as pd
import numpy as np
import logging
import matplotlib.pyplot as plt


# In[2]:


experiments = {}
out_dir = os.environ['OUTPUT_DIR']#'experiments_output'
for name in os.listdir(out_dir):
    if name.startswith('task'):
        if not os.path.exists(os.path.join(out_dir, name, 'report.json')):
            continue
            
        experiments[name] = {}
        
        experiments[name]['task'] = 'a' if name.startswith('task-a') else 'b'
        
        # Load report
        with open(os.path.join(out_dir, name, 'report.json'), 'r') as f:
            experiments[name]['report'] = json.load(f)
         
        if os.path.exists(os.path.join(out_dir, name, 'model_config.json')):
            with open(os.path.join(out_dir, name, 'model_config.json'), 'r') as f:
                experiments[name]['config'] = json.load(f)
            
        if 'author-only' in name:
            with open(os.path.join(out_dir, name, 'report_author_vec_found.json'), 'r') as f:
                experiments[name]['report'] = json.load(f) 
    else:
        if not os.path.exists(os.path.join(out_dir, name, 'report.json')):
            continue
        experiments[name] = {}
        experiments[name]['task'] = "imdb" 
        # Load report
        with open(os.path.join(out_dir, name, 'report.json'), 'r') as f:
            experiments[name]['report'] = json.load(f)
        if os.path.exists(os.path.join(out_dir, name, 'model_config.json')):
            with open(os.path.join(out_dir, name, 'model_config.json'), 'r') as f:
                experiments[name]['config'] = json.load(f)    


# In[3]:


metrics = ['f1-score', 'precision', 'recall']
scores = []

for n, d in experiments.items():
    dd = {
        'name': n,
        'task': d['task'],
    }
    
    for metric in metrics:
        dd[metric] = d['report']['micro avg'][metric] * 100

    scores.append(dd)
    
#scores = [{'name': n, 'task': , 'micro avg f1-score': d['report']['micro avg']['f1-score']}]

df = pd.DataFrame(scores)

df = df.set_index('name')

for metric in metrics:
    df[metric + '_diff'] = 0


# In[4]:


from IPython.display import display

pd.options.display.float_format = '{:,.2f}'.format

def display_task(df, task_id):
    print(f'#### Task {task_id} ####')
    print(df) 
    df_a = df[df['task'] == task_id]
    print(df_a)
    for metric in metrics:
        max_val = df_a[metric].max()
        df_a = df_a.assign(**{metric + '_diff': max_val - df_a[metric]})
        
    for metric in metrics:
        print(f'Sorted by {metric}')
        display(df_a.sort_values([metric], ascending=False))


# In[5]:

display_task(df,"imdb")
#display_task(df, 'a')


# In[6]:


#display_task(df, 'b')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




