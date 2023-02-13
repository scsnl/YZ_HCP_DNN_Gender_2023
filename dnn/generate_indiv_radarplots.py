from plotly import graph_objs as go
import numpy as np
import json
import pickle5 as pickle
import os
from scipy.io import savemat

#   
bn_atlas_file = '/Users/zhangyuan/Desktop/Sherlock/projects/sryali/2019_DNN/scripts/features/bnatlas_tree.json'

with open(bn_atlas_file) as f:
    bn_atlas=json.load(f)

def generate_radial_plots(subj_list,subjid_list,gender,norm_medians,feature_idcs,name_reordered,output_dir,site):

    if gender == 0:
        group = 'male'
        # color = 'blue'
        color_list = ['#00CDCD', '#00EEEE', '#97FFFF']
    elif gender == 1:
        group = 'female'
        # color = 'pink'
        color_list = ['#EE3A8C','#FF82AB','#FFB5C5']

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


def get_common_subjects(dict,test_session_alias_list):

    session1subjids = dict[test_session_alias_list[0]] #.to_numpy()
    session2subjids = dict[test_session_alias_list[1]] #.to_numpy()
    session3subjids = dict[test_session_alias_list[2]] #.to_numpy()
    session4subjids = dict[test_session_alias_list[3]] #.to_numpy()

    commonsubjids12, session1_ix, session2_ix = np.intersect1d(session1subjids, session2subjids, return_indices=True)
    commonsubjids34, session3_ix, session4_ix = np.intersect1d(session3subjids, session4subjids, return_indices=True)
    # print('commonsubjid12 example {}'.format(commonsubjids12[0:10]))
    # print('commonsubjid34 example {}'.format(commonsubjids34[0:10]))
    commonsubjids, tmp1_ix, tmp2_ix = np.intersect1d(commonsubjids12, commonsubjids34, return_indices=True)
    # print('commonsubjid across four sessions shape {}'.format(commonsubjids.shape))
    # print('commonsubjid example {}'.format(commonsubjids[0:10]))

    return commonsubjids


def get_subjid_correct_pred(dict, hcp_sessions_list, test_session_alias_list, best_split_id):

    subjectID_female = {}
    subjectID_male = {}
    commonsubjids = get_common_subjects(dict, test_session_alias_list)

    for ss in range(4):
        #subjectID[test_session_alias_list[ss]] = {}
        subjids = dict[test_session_alias_list[ss]].to_numpy()

        data_file = data_path + 'data_for_radial_plots_HCP_model_' + hcp_sessions_list[ss] + '_index_' + str(best_split_id[ss]) + '_test_' + \
                    test_session_alias_list[ss] + '.npz'

        data = np.load(data_file)
        predictions = data['predictions']  # predictions (num_sub) - used to assess correctness of prediction
        # print('predictions shape {}'.format(predictions.shape))
        labels = data['labels']  # labels (num_sub) - used to assess correctness of prediction
        # print('labels shape {}'.format(labels.shape))

        correct_idcs = np.argwhere(labels == predictions)
        # print('correct_idcs shape {}, content {}'.format(correct_idcs.shape, correct_idcs[0:10]))
        f_idcs = np.argwhere(labels == 1)
        # print('f_idcs shape {}, content {}'.format(f_idcs.shape, f_idcs[0:10]))
        m_idcs = np.argwhere(labels == 0)
        # print('m_idcs shape {}, content {}'.format(m_idcs.shape, m_idcs[0:10]))
        correct_f_idcs, tmp1_ix, tmp2_ix = np.intersect1d(correct_idcs, f_idcs, return_indices=True)
        # print('correct_f_idcs shape {}, content {}, subjids {}'.format(correct_f_idcs.shape, correct_f_idcs[0:10], subjids[correct_f_idcs[0:10]]))
        correct_m_idcs, tmp1_ix, tmp2_ix = np.intersect1d(correct_idcs, m_idcs, return_indices=True)
        # print('correct_m_idcs shape {}, content {}, subjids {}'.format(correct_m_idcs.shape, correct_m_idcs[0:10], subjids[correct_m_idcs[0:10]]))

        tmp_f = np.intersect1d(subjids[correct_f_idcs], commonsubjids, return_indices=True)
        # print('shape {}, content {}'.format(tmp_f[0].shape, tmp_f[0][0:10]))
        tmp_m = np.intersect1d(subjids[correct_m_idcs], commonsubjids, return_indices=True)
        # print('shape {}, content {}'.format(tmp_m[0].shape, tmp_m[0][0:10]))
        subjectID_female[test_session_alias_list[ss]] = tmp_f[0]
        subjectID_male[test_session_alias_list[ss]] = tmp_m[0]

    commonsubjids_correct_female = get_common_subjects(subjectID_female, test_session_alias_list)
    commonsubjids_correct_male = get_common_subjects(subjectID_male, test_session_alias_list)
    # print('commonsubjids female correct shape {}, content {}'.format(commonsubjids_correct_female.shape, commonsubjids_correct_female[0:10]))
    # print('commonsubjids male correct shape {}, content {}'.format(commonsubjids_correct_male.shape, commonsubjids_correct_male[0:10]))

    return commonsubjids_correct_female, commonsubjids_correct_male


if __name__ == '__main__':

    best_split_id = [2, 2, 2, 2]  # correspoding to split index for S1, S1, S3, and S3
    hcp_sessions_list = ['LR_S1', 'LR_S1', 'RL_S1', 'RL_S1']  # models S1, S1, S3, S3
    test_session_list = ['REST1_LR', 'REST2_LR', 'REST1_RL', 'REST2_RL']  # test S1, S2, S3, S4
    test_session_alias_list = ['LR_S1', 'LR_S2', 'RL_S1', 'RL_S2']
    # we want the combination of hcp_session and test_session to get the features file
    data_path = '/Users/zhangyuan/Desktop/Sherlock/projects/menon/2021_HCP_Gender/results/restfmri/dnn/radial_plots/'
    # output_path = data_path
    output_path = '/Users/zhangyuan/Desktop/radial_plots_test/'
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    subjid_file = '/Users/zhangyuan/Desktop/Sherlock/projects/menon/2021_HCP_Gender/results/restfmri/dnn/basic_info/subjid_HCP_sessions.pkl'
    with open(subjid_file, 'rb') as handle:
        mydict = pickle.load(handle)

    subjids_female, subjids_male = get_subjid_correct_pred(mydict, hcp_sessions_list, test_session_alias_list, best_split_id)


    for ss in range(4):
        for group in range(2):

            if group == 0:
                group_label = 'male'
            elif group == 1:
                group_label = 'female'

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
                subj_idcs = np.squeeze(np.where(np.isin(full_subjids, subjids_male[0:3])))
                print(subj_idcs)
                generate_radial_plots(subj_idcs, full_subjids,0, norm_medians, feature_idcs, name_reordered, output_path, site)
            elif group == 1:
                subj_idcs = np.squeeze(np.where(np.isin(full_subjids, subjids_female[0:3])))
                print(subj_idcs)
                generate_radial_plots(subj_idcs, full_subjids, 1, norm_medians, feature_idcs, name_reordered,output_path,site)






