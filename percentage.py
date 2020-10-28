import pandas as pd

import numpy as np
import sys
pd.set_option('display.max_rows', None)
df_dir = pd.read_csv('directors.dat', delimiter='::', engine='python')
del df_dir['birthday']
del df_dir['place']

fe_dir = len(df_dir[df_dir['gender'] == 1])
m_dir = len(df_dir[df_dir['gender'] == 2])

print(fe_dir)
print(m_dir)
print(fe_dir/(m_dir+fe_dir)*100)
