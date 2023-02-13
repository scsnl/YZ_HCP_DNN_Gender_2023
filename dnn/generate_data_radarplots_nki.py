import numpy as np
import json
from sklearn.preprocessing import normalize
import pickle
# import pickle5 as pickle
import os

#   
bn_atlas_file = 'PROJECT_DIR/scripts/features/bnatlas_tree.json'

with open(bn_atlas_file) as f:
    bn_atlas=json.load(f)


if __name__ == '__main__':

    # save np.load
    np_load_old = np.load
    # modify the default parameters of np.load
    np.load = lambda *a, **k: np_load_old(*a, allow_pickle=True, **k)


    best_split_id = [2, 2]  # correspoding to split index for S1, S1, S3, and S3
    hcp_sessions_list = ['LR_S1', 'RL_S1']  # models S1, S1, S3, S3
    test_session_list = ['645', '645']  # test S1, S2, S3, S4
    test_session_alias_list = ['nki_645', 'nki_645']
    # we want the combination of hcp_session and test_session to get the features file
    output_path = 'PROJECT_DIR/results/restfmri/dnn/radial_plots/'
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    data_path = 'PROJECT_DIR/results/restfmri/dnn/attributions/'
    for ss in range(2):
        model_session = hcp_sessions_list[ss]
        m = best_split_id[ss]
        test_session = test_session_list[ss]
        data_file = data_path + 'hcp_model_' + model_session + '_index_' + str(m) + '_test_nki_' + test_session + '.npz'
        print("data file {}".format(data_file))
        site = 'hcp_model_' + model_session + '_index_' + str(m) + '_test_nki_' + test_session

        data = np.load(data_file)  # load feature attributions (#sub x #roi x #tp), labels etc.
        data_features = data['features']  # #sub x #roi x #tp
        labels = data['labels']
        predictions = data['predictions']
        subjids = data['subjid']

        f_idcs = np.where(data['labels'] == 1)
        m_idcs = np.where(data['labels'] == 0)

        for group in range(2): # 0: male, 1: female

            if group == 0:
                group_label = 'male'
                group_features = data_features[m_idcs]
                group_labels = labels[m_idcs]
                print(type(group_labels))
                group_predictions = predictions[m_idcs]
                group_subjids = subjids[m_idcs]
                print(type(group_subjids))
                print(group_subjids)
            elif group == 1:
                group_label = 'female'
                group_features = data_features[f_idcs]
                group_labels = labels[f_idcs]
                print(type(group_labels))
                group_predictions = predictions[f_idcs]
                group_subjids = subjids[f_idcs]
                print(type(group_subjids))
                print(group_subjids)

            output_fname = output_path + group_label + '_data_for_radial_plots_HCP_model_' + model_session + '_index_' + str(m) + '_test_' + test_session_alias_list[ss] + '.npz'
            print("output file {}".format(output_fname))

            # 1 x #roi (average across subjects)
            features = np.mean(np.abs(np.median(group_features, axis=2)), axis=0)
            # #sub x #roi
            data_medians = np.abs(np.median(group_features, axis=2))
            # normalize along roi - #sub x #roi
            norm_medians = normalize(data_medians, axis=1)
            # 1 x #roi (average across subjects): in descending order in terms of feature attributions
            feature_idcs = np.flip(np.argsort(features))
            feature_alias = []
            feature_names = []
            parent_regions = []
            for idx in range(1, 247):
                for region in bn_atlas:
                    if region['id'] == str(idx):
                        feature_alias.append(region['data']['alias'])
                        feature_names.append(region['text'])
                        parent_regions.append(region['parent'])
            features_thresholded_alias = [feature_alias[x] for x in feature_idcs[0:13]]
            features_thresholded_names = [feature_names[x] for x in feature_idcs[0:13]]
            parent_regions_thresholded = [parent_regions[x].split(',')[0] for x in feature_idcs[0:13]]
            combined_names = [', '.join([parent_regions_thresholded[x], features_thresholded_alias[x].split(',')[0]]) for x
                              in range(len(parent_regions_thresholded))]
            name_reordered = ['_'.join([x.split('_')[1] + x.split('_')[0], str(feature_idcs[idx] + 1)]) for idx, x in
                              enumerate(features_thresholded_names)]
            print(feature_idcs[0:13])
            print(features_thresholded_names)
            print(features_thresholded_alias)
            print(parent_regions_thresholded)
            print(combined_names)

            np.savez(output_fname, norm_medians=norm_medians, feature_idcs=feature_idcs, combined_names=combined_names, name_reordered=name_reordered,
                     predictions=group_predictions, labels=group_labels, subjids=group_subjids)

    # restore np.load for future normal usage
    np.load = np_load_old