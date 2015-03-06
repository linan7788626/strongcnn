# code for taking the databatches and saving just the filenames for collating
import cPickle as pickle
import numpy as np

input_dir = '/srv/zfs01/user_data/cpd/spacewarp_batches/'
output_dir = input_dir + 'batches/'

for i in range(900, 904):
    file_path = input_dir + 'data_batch_{0}'.format(i)
    data = np.load(file_path)
    filenames = data['filenames']
    galids = data['galids']
    pickle.dump({key: data[key] for key in ['filenames', 'galids']},
                open(output_dir + 'data_reduced_{0}'.format(i), 'w'))
