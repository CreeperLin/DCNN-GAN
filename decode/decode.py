#!/usr/bin/python
# -*- coding=utf-8 -*-
import sys
import os
import pickle
import re
import numpy as np
import argparse
import dec_config as config

from itertools import product
import sklearn.metrics
import math

from sklearn.linear_model import Ridge

def get_img_feat(img_feat_path):

    if os.path.exists(img_feat_path):
        with open(img_feat_path,'rb') as f:
            data_feature = pickle.load(f)
        print('loaded %s' % img_feat_path)
        return data_feature
    else:
        print('image feature file not exist')
        exit(0)

def sort_img_feat(data, f_label):

    val = data.values()
    for i in val:
        feat_len = i.shape[0]
        break

    avail_idx=[]
    feat=[]
    keys=[]
    for label in f_label:
        str_lbl = str(label).split('.')
        plh=''
        if len(str_lbl[0])==7:
            plh='0'
        str_idx = (str_lbl[1] + '0'*(6-len(str_lbl[1]))).lstrip('0')
        lbl_key = 'n'+plh+str_lbl[0]+'_'+str_idx+'.JPEG'
        keys.append(lbl_key)
        if lbl_key in data:
            avail_idx.append(True)
            feat.append(data[lbl_key])
        else:
            avail_idx.append(False)
            feat.append(np.zeros(feat_len))
    
    feat = np.array(feat)
    avail_idx = np.array(avail_idx)
    keys = np.array(keys)
    print('avail %d/%d' % (np.sum(avail_idx),avail_idx.shape[0]))
    return feat, avail_idx, keys

def main():

    parser = argparse.ArgumentParser(description='Run fMRI decoder')
    parser.add_argument('--fmri_data', type=str)
    parser.add_argument('--feat_data', type=str)
    parser.add_argument('--output', type=str)

    args = parser.parse_args()
    
    subjects = config.subjects
    rois = config.rois

    results_dir = args.output
    fmri_dir = args.fmri_data
    feat_dir = args.feat_data

    data_feature = get_img_feat(os.path.join(feat_dir,config.image_feature_name))
    
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    for sbj, roi in product(subjects, rois):
        print('--------------------')
        print('Subject:    %s' % sbj)
        print('ROI:        %s' % roi)
        
        with open(os.path.join(fmri_dir,'fmri_'+sbj+'_'+roi+'.pkl'),'rb') as f:
            dat=pickle.load(f)

        datatype = dat[:,0]
        labels = dat[:,1]
        x = dat[:,2:]

        decode_id = 'decode_' + sbj + '_' + roi

        y = data_feature
        y_sorted, i_avail, keys = sort_img_feat(y, labels)

        i_train = (datatype == 1).flatten()    # Index for training
        i_test = (datatype == 2).flatten()  # Index for perception test

        i_test_f = i_test & i_avail
        i_train_f = i_train & i_avail

        x_train = x[i_train_f, :]
        x_test = x[i_test_f, :]
        y_train = y_sorted[i_train_f, :]
        y_test = y_sorted[i_test_f, :]
        lbl = keys[i_test_f]
        
        pred_y = run_decode(x_train, y_train, x_test, y_test)

        savepath = os.path.join(results_dir, decode_id+'_pred.pkl')
        with open(savepath, 'wb') as f:
            pickle.dump(pred_y, f)
        print('Saved %s' % savepath)

        savepath = os.path.join(results_dir, decode_id+'_id.pkl')
        with open(savepath, 'wb') as f:
            pickle.dump(lbl, f)
        print('Saved %s' % savepath)

# Functions ############################################################

def run_decode(x_train, y_train, x_test, y_test):
    '''Run feature prediction

    Parameters
    ----------
    x_train, y_train : array_like [shape = (n_sample, n_voxel)]
        Brain data and image features for training
    x_test, y_test : array_like [shape = (n_sample, n_unit)]
        Brain data and image features for test

    Returns
    -------
    predicted_label : array_like [shape = (n_sample, n_unit)]
        Predicted features
    '''

    print('x_train',np.shape(x_train))
    print('y_train',np.shape(y_train))
    print('x_test',np.shape(x_test))
    print('y_test',np.shape(y_test))

    model = Ridge(
        normalize=True,
        alpha=0.7,
        max_iter=1000,
        solver='auto',
    )
    model.fit(x_train,y_train)
    y_pred = model.predict(x_test)
    print('ridge: rmse:',math.sqrt(sklearn.metrics.mean_squared_error(y_test,y_pred)),'r2:',sklearn.metrics.r2_score(y_test,y_pred))

    return y_pred

if __name__ == '__main__':
    main()
