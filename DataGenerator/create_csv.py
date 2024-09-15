import pandas as pd
import glob
import json
from tqdm import tqdm


dataset      = 'Nanopore_single_flowcell'    
cluster_type = 'pseudo' 

path = ''
save_path = '' 

files = glob.glob(path+'*.json')
df = pd.DataFrame()
df['path'] = files
drop_index = []

for i,file in enumerate(tqdm(files)):

    try:
        with open(file) as json_file:
            data = json.load(json_file)
        if data['data'] == 'None':
            drop_index.append(i)
        df['index'] = data['index']
    except:
        drop_index.append(i)
    break
  
df = df.drop(drop_index)
df = df.reset_index(drop=True)
df.to_csv(save_path+'.csv')
