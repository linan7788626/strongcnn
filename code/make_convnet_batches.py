"""
Process images into the batch files
"""

import ipdb
import numpy as np
import cPickle as pickle
import matplotlib.pylab as pl
pl.ion()

# Define DATAPATH, which should contain
# images_training_rev1, images_test_rev1, and training_solutions_rev1.csv,
# and should be writeable.
datapath = '/farmshare/user_data/cpd/spacewarp_batches/cutouts_good/'
#datapath = '../galzoo_old/'

n_files_per_batch = 10263
n_batches_test = 9

def make_convnet_batches_test(simple_ds, mask_strength, savepath):
    n_batches = n_batches_test
    from os import system
    from glob import glob
    from PIL import Image
    files = sorted(glob(datapath+'*.png'))
    dummy_labels = np.zeros(37)-1.
    labels = [dummy_labels for i in range(n_files_per_batch)]    
    nside = preprocess3band_convnet(files[0], simple_ds=simple_ds, mask_strength=mask_strength).shape[0]
    pickle.dump(files, open(savepath+'test_files.pkl','w'))    
    for ibatch in range(n_batches):
        data = np.zeros((nside*nside*3, n_files_per_batch), dtype=np.uint8)
        ind_min = ibatch*n_files_per_batch
        ind_max = np.min([(ibatch+1)*n_files_per_batch, len(files)])
        filenames = files[ind_min:ind_max]
        if len(filenames) == 0:
            break
        galids = []
        print(ibatch,n_batches)
        for ifile,file in enumerate(filenames):
            if (ifile%100)==0: print(ifile,n_files_per_batch)
            data[:, ifile] = preprocess3band_convnet(file, simple_ds=simple_ds, mask_strength=mask_strength).T.ravel()
            this_galid = file2galid(file)
            galids.append(this_galid)
        output = {'data':data, 'labels':labels, 'filenames':filenames,
                  'galids':galids}
        pickle.dump(output, open(savepath+'data_batch_%i'%(900+ibatch), 'w'))



def preprocess3band_convnet(filename, simple_ds=3., mask_strength=None):
    from PIL import Image
    from skimage.filters import gaussian_filter
    # open file
    x=np.array(Image.open(filename), dtype=np.float)

    if ((simple_ds>1) & (simple_ds!=None)):
        # Gaussian smooth with FWHM = "simple_ds" pixels.
        for i in range(3):
            x[:,:,i] = gaussian_filter(x[:,:,i], 1.*simple_ds/2.355)
        # subsample by simple_ds.
        x = x[0::int(simple_ds), 0::int(simple_ds), :]
        # take inner 96x96.
        ntmp = x.shape[0]-96
        if (ntmp % 2)==0:
            nside=ntmp/2
            x = x[nside:-nside, nside:-nside, :]
        else:
            nside=(ntmp-1)/2
            x = x[nside+1:-nside, nside+1:-nside, :]
    else:
        # If we're not doing simple down-sampling, proceed here.
        #for i in range(3): x[:,:,i] -= np.min(x[0,:,i])
        # get mask
        avg = np.mean(x,axis=-1)
        mask = get_mask(avg, thresh=25)
        # measure "width"=second spatial moment
        nside = x.shape[0]
        d1d = np.arange(nside)[:,np.newaxis]; d1d-=d1d.mean()
        rsq = np.transpose(d1d**2) + d1d**2
        width = np.sqrt(np.mean(rsq*(mask*x.mean(2))))
        from scipy.interpolate import RectBivariateSpline
        this_zoom = 100./width
        if this_zoom<1: this_zoom=1
        if this_zoom>4: this_zoom=4
        this_stride = nside/96./this_zoom
        tmp_in = np.arange(nside)
        tmp_out = np.arange(96)*this_stride
        tmp_out -= tmp_out.mean(); tmp_out += tmp_in.mean()
        tmp_out_x = np.zeros((96,96)) + tmp_out[:,np.newaxis]
        tmp_out_y = np.zeros((96,96)) + tmp_out[np.newaxis,:]
        xnew = np.zeros((96,96,3))
        for i in range(3):
            rb = RectBivariateSpline(tmp_in, tmp_in, x[:,:,i], kx=2,ky=2,s=0)
            xnew[:,:,i] = rb.ev(tmp_out_x.ravel(), tmp_out_y.ravel()).reshape(96,96)
        rb = RectBivariateSpline(tmp_in, tmp_in, mask, kx=1,ky=1,s=0)
        mask = rb.ev(tmp_out_x.ravel(), tmp_out_y.ravel()).reshape(96,96)
        x = xnew      

    # If desired, apply mask.
    if (mask_strength==None) | (mask_strength=='none'):
        return x
    else:
        if (mask_strength=='weak'): mask_thresh = 15.
        if (mask_strength=='strong'): mask_thresh = 30.
        avg = np.mean(x,axis=-1)
        mask = get_mask(avg, thresh=mask_thresh)
        x *= mask[:,:,np.newaxis]
        return x

def get_mask(img, thresh=25):
    # add color cut in here?
    from skimage.filters import gaussian_filter
    from scipy import ndimage
    (nx,ny)=img.shape
    sm = gaussian_filter(img, 4.0)
    #sm = gaussian_filter(img, 3.0)    
    notdark = sm>thresh
    label_objects, nb_labels = ndimage.label(notdark)
    mask = label_objects == label_objects[np.round(nx/2.), np.round(ny/2.)]
    return mask

def file2galid(file):
    return int(file.split('/')[-1].split('_')[0])


def do_preprocessing(savepath):
    from os import makedirs, path

    simple_ds = None  # no downsampling for me!
    mask_strength = None
    if not path.exists(savepath):
        makedirs(savepath)
    print(simple_ds, mask_strength, savepath)
    make_convnet_batches_test(simple_ds, mask_strength, savepath)

if __name__ == '__main__':
    savepath = '/farmshare/user_data/cpd/spacewarp_batches/'
    do_preprocessing(savepath)
