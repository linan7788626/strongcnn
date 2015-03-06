"""
Load up the outputs and collate with the catalogs into a catalog of features
"""

import numpy as np
import pandas as pd

project_dir = '/afs/ir/users/c/p/cpd/Projects/strongcnn/'
datapath = '/farmshare/user_data/cpd/spacewarp_batches/'

project_dir = '/Users/cpd/Projects/strongcnn/'
datapath = '/Users/cpd/Desktop/batches/'



# load up catalog for cutouts
cat = pd.read_csv(project_dir + 'catalog/cluster_catalog.csv').set_index('cutoutname')

names = []
for batch_i in range(900, 903+1):
    redpath = datapath + 'data_reduced_{0}'.format(batch_i)
    reduced = np.load(redpath)  # pickle of filenames

    names_i = [filename.split('/')[-1] 
               for filename in reduced['filenames']]

    for iexp in range(0, 20):
        batchpath = datapath + 'data_batch_{0}'.format(batch_i) + '_iexp{0}'.format(iexp)
        batch = np.load(batchpath)

        if ((batch_i == 900) and (iexp == 0)):
            data = batch['data'][:len(names_i)]
        else:
            data = np.vstack((data, batch['data'][:len(names_i)]))


        names += names_i

        # cool now I have all that data let's make a dataframe
        print(batch_i, iexp, data.shape, len(names))
df = pd.DataFrame(data, index=names,
                  columns=['nn{0}'.format(i)
                           for i in xrange(data.shape[1])])

for key in cat.columns:
    df[key] = cat[key].loc[names]

df['cutoutname'] = names
df = df.set_index('Unnamed: 0')

df.to_csv(datapath + 'database.csv')

