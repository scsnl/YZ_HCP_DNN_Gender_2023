from plotly import graph_objs as go
import numpy as np
import json
import pickle5 as pickle
from scipy.io import savemat

#   
bn_atlas_file = '/Users/zhangyuan/Desktop/Sherlock/projects/sryali/2019_DNN/scripts/features/bnatlas_tree.json'

with open(bn_atlas_file) as f:
    bn_atlas=json.load(f)

def generate_radial_plots(subj_list,subjid_list,gender,norm_medians,feature_idcs,name_reordered,output_dir,site):

    if gender == 0:
        group = 'male'
        color_list = ['#00CDCD', '#00EEEE', '#97FFFF']
    elif gender == 1:
        group = 'female'
        color_list = ['#EE3A8C','#FF82AB','#FFB5C5']

    # color_list = ['blue', 'red', 'green'] #, 'orange', 'purple']
    for ii in range(len(subj_list)):
        print(ii)
        subj_idx = subj_list[ii]
        print('subj_list {}, subj_idx {}'.format(subj_list, subj_idx))
        # feature_idcs - this is the order of features determined based on mean across subjects
        # advantage of this is that for all subjects' radarplots show the same set of features
        # disadvantage is that this fixed set of features may not be a subject's top features
        # norm_medians: #sub x #roi
        subj_features=norm_medians[subj_idx][feature_idcs[0:13]]

        fig = go.Figure(data=go.Scatterpolar(
                name="subj_%s" % subjid_list[subj_idx],
                r=subj_features,
                theta=name_reordered,
                theta0=90,
                fill='tonext',
                line=dict(width=1, color=color_list[ii])
        ))

        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                visible=False
                ),
                angularaxis=dict(showticklabels=True,ticks='',)
            ),
            showlegend=False)

        fig.write_image(output_dir + 'radarplot_site_%s_group_%s_subj_%s_%s.png'%(site,group,subjid_list[subj_idx],color_list[ii]))


if __name__ == '__main__':

    # save np.load
    np_load_old = np.load
    # modify the default parameters of np.load
    np.load = lambda *a, **k: np_load_old(*a, allow_pickle=True, **k)

    best_split_id = [2, 2]
    hcp_sessions_list = ['LR_S1', 'RL_S1']
    test_session_list = ['645', '645']
    test_session_alias_list = ['nki_645', 'nki_645']
    # we want the combination of hcp_session and test_session to get the features file
    data_path = '/Users/zhangyuan/Desktop/Sherlock/projects/menon/2021_HCP_Gender/results/restfmri/dnn/radial_plots/'
    # output_path = data_path
    output_path = '/Users/zhangyuan/Desktop/radial_plots/nki/'
    # if not os.path.exists(output_path):
    #     os.makedirs(output_path)

    data_file_1m = data_path + 'male_data_for_radial_plots_HCP_model_' + hcp_sessions_list[0] + '_index_' + str(
        best_split_id[0]) + '_test_' + test_session_alias_list[0] + '.npz'
    data_file_2m = data_path + 'male_data_for_radial_plots_HCP_model_' + hcp_sessions_list[1] + '_index_' + str(
        best_split_id[1]) + '_test_' + test_session_alias_list[1] + '.npz'
    data_file_1f = data_path + 'female_data_for_radial_plots_HCP_model_' + hcp_sessions_list[0] + '_index_' + str(
        best_split_id[0]) + '_test_' + test_session_alias_list[0] + '.npz'
    data_file_2f = data_path + 'female_data_for_radial_plots_HCP_model_' + hcp_sessions_list[1] + '_index_' + str(
        best_split_id[1]) + '_test_' + test_session_alias_list[1] + '.npz'

    data_1m = np.load(data_file_1m)
    predictions_1m = data_1m['predictions']  # predictions (num_sub) - used to assess correctness of prediction
    labels_1m = data_1m['labels']  # labels (num_sub) - used to assess correctness of prediction
    correct_idcs_1m = np.argwhere(labels_1m == predictions_1m)

    data_2m = np.load(data_file_2m)
    predictions_2m = data_2m['predictions']  # predictions (num_sub) - used to assess correctness of prediction
    labels_2m = data_2m['labels']  # labels (num_sub) - used to assess correctness of prediction
    correct_idcs_2m = np.argwhere(labels_2m == predictions_2m)

    correct_m_idcs, tmp1_ix, tmp2_ix = np.intersect1d(correct_idcs_1m, correct_idcs_2m, return_indices=True)
    print(correct_m_idcs)

    data_1f = np.load(data_file_1f)
    predictions_1f = data_1f['predictions']  # predictions (num_sub) - used to assess correctness of prediction
    labels_1f = data_1f['labels']  # labels (num_sub) - used to assess correctness of prediction
    correct_idcs_1f = np.argwhere(labels_1f == predictions_1f)

    data_2f = np.load(data_file_2f)
    predictions_2f = data_2f['predictions']  # predictions (num_sub) - used to assess correctness of prediction
    labels_2f = data_2f['labels']  # labels (num_sub) - used to assess correctness of prediction
    correct_idcs_2f = np.argwhere(labels_2f == predictions_2f)

    correct_f_idcs, tmp1_ix, tmp2_ix = np.intersect1d(correct_idcs_1f, correct_idcs_2f, return_indices=True)
    print(correct_f_idcs)



    for ss in range(2):
        for group in range(2):

            if group == 0:
                group_label = 'male'
                subj_idcs = correct_m_idcs[0:3]
            elif group == 1:
                group_label = 'female'
                subj_idcs = correct_f_idcs[0:3]

            site = 'HCP_model_' + hcp_sessions_list[ss] + '_index_' + str(best_split_id[ss]) + '_test_' + test_session_alias_list[ss]
            data_file = data_path + group_label +'_data_for_radial_plots_HCP_model_' + hcp_sessions_list[ss] + '_index_' + str(best_split_id[ss]) + '_test_' + test_session_alias_list[ss] + '.npz'

            data = np.load(data_file)

            norm_medians = data['norm_medians'] # feature attributions (num_sub x num_roi)
            print('norm_median shape {}'.format(norm_medians.shape))
            feature_idcs = data['feature_idcs'] # feature indices sorted in decreasing order of importance (num_roi)
            print('feature_idcs shape {}'.format(feature_idcs.shape))
            name_reordered = data['name_reordered'] # 25 names corresponding to top 25 (or ~10%) features
            print('name_reordered shape {}'.format(len(name_reordered)))
            full_subjids = data['subjids']
            print('full_subjids shape {}'.format(len(full_subjids)))

            if group == 0:
                print('male: {}'.format(full_subjids[subj_idcs]))
                generate_radial_plots(subj_idcs, full_subjids,0, norm_medians, feature_idcs, name_reordered, output_path, site)
            elif group == 1:
                print('female: {}'.format(full_subjids[subj_idcs]))
                generate_radial_plots(subj_idcs, full_subjids, 1, norm_medians, feature_idcs, name_reordered,output_path,site)

    # restore np.load for future normal usage
    np.load = np_load_old




