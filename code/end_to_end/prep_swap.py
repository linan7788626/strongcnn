import numpy as np
import ipdb

def main():
    make_train_and_val_listfiles()

def make_train_and_val_listfiles(frac_val=0.2):
    #from glob import glob
    #filenames = glob('data/cutouts_good/*.png')
    #np.random.shuffle(filenames)
    df = load_labels()

    df_not = df[(df.object_flavor == 'DUD')]
    df_lens = df[(df.object_flavor != 'DUD') & (df.object_flavor != 'UNKNOWN')]
    filenames_not = df_not.object_name.values
    filenames_lens = df_lens.object_name.values
    filenames = np.concatenate([filenames_not, filenames_lens])
    labels = np.concatenate([np.zeros(len(filenames_not)), np.ones(len(filenames_lens))])
    ind = np.arange(len(filenames))
    np.random.shuffle(ind)
    filenames = filenames[ind]
    labels = labels[ind]
    f_train = open('caffe_data/sw_train_listfile','w')
    f_val = open('caffe_data/sw_val_listfile','w')
    for this_filename, this_label in zip(filenames, labels):

        if (np.random.random()<frac_val):
            f_val.write('%s %i\n'%(this_filename, this_label))
        else:
            f_train.write('%s %i\n'%(this_filename, this_label))
    f_train.close()
    f_val.close()


def load_labels():
    from pandas.io.parsers import read_csv
    df = read_csv('cutout_catalog/catalog.csv') #, index_col='cutoutname')
    return df


def make_jpg_copies():
    pass
    '''
    from scipy.misc import imread, imsave
    from glob import glob
    filenames = glob('data/cutouts_good/*.png')
    for filename in filenames:
        x = imread(filename)
        savename = filename.split('.png')[0] + '.jpg'
        imsave(x, savename)
    '''

if __name__ == '__main__':
    main()
