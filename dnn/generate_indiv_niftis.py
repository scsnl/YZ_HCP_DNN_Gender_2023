import warnings
warnings.filterwarnings("ignore")
# import pickle5 as pickle # this is for local run
import pickle # this is for sherlock run
import matplotlib as mpl
from utilityFunctions import *

high_res=True

parent_substitutions={
    'Amyg, Amygdala': 'Middle Temporal Gyrus',
    'CG, Cingulate Gyrus': 'Cingulate Gyrus',
    'Cun, Cuneus': 'PCC, Precuneus',
    'Frontal Lobe': 'Prefrontal Cortex',
    'FuG, Fusiform Gyrus': 'Inferior Temporal Gyrus',
    'Hipp, Hippocampus' : 'Middle Temporal Gyrus',
    'IFG, Inferior Frontal Gyrus': 'Prefrontal Cortex',
    'INS, Insular Gyrus' : 'Prefrontal Cortex',
    'IPL, Inferior Parietal Lobule': 'Inferior Parietal Lobe',
    'ITG, Inferior Temporal Gyrus': 'Inferior Temporal Gyrus',
    'Insular Lobe':'Prefrontal Cortex',
    'Limbic Lobe' : 'Middle Temporal Lobe',
    'MFG, Middle Frontal Gyrus' : 'Prefrontal Cortex',
    'MTG, Middle Temporal Gyrus' : 'Middle Temporal Gyrus',
    'OcG, Occipital Gyrus' : 'Occipital Gyrus',
    'Occipital Lobe':'Occipital Lobe',
    'OrG, Orbital Gyrus' : 'Prefrontal Cortex',
    'PCL,Paracentral Lobule' : 'Paracentral Lobule',
    'Parietal Lobe' : 'Parietal Lobe',
    'Pcun, Precuneus' : 'PCC, Precuneus',
    'PhG, Parahippocampal Gyrus' : 'Middle Temporal Gyrus',
    'PoG, Postcentral Gyrus' : 'Postcentral Gyrus',
    'PrG, Precentral Gyrus' : 'Precentral Gyrus',
    'Psts, Posterior Superior Temporal Sulcus' : 'Superior Temporal Gyrus',
    'SFG, Superior Frontal Gyrus' : 'Prefrontal Cortex',
    'SPL, Superior Parietal Lobule' : 'Superior Parietal Lobule',
    'STG, Superior Temporal Gyrus' : 'Superior Temporal Gyrus',
    'Str, Striatum' : 'Striatum',
    'Subcortical Nuclei' : 'Subcortical Nuclei',
    'Temporal Lobe' : 'Temporal Lobe',
    'Tha, Thalamus' : 'Thalamus',
              }
if high_res:
    mpl.rcParams['figure.dpi'] = 1600
    target='fsaverage'
else:
    mpl.rcParams['figure.dpi'] = 80
    target='fsaverage5'



if __name__ == "__main__":

    best_split_id = [2, 2, 2, 2]  # correspoding to split index for S1, S1, S3, and S3
    # which sessions models were trained on
    model_sessions_list = ['LR_S1', 'LR_S1', 'RL_S1', 'RL_S1']
    # which normz non-windowed entire dataset were used for generating feature attribution
    test_sessions_list = ['REST1_LR', 'REST2_LR', 'REST1_RL', 'REST2_RL']
    test_session_alias_list = ['LR_S1', 'LR_S2', 'RL_S1', 'RL_S2']

    subjid_file = 'PROJECT_DIR/results/restfmri/dnn/basic_info/subjid_HCP_sessions.pkl'
    with open(subjid_file, 'rb') as handle:
        mydict = pickle.load(handle)

    # output
    output_path = 'PROJECT_DIR/results/restfmri/dnn/niis/indiv_niis/'
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # keep consisten with radarplots
    # female_subjids = ['102311','101915','101006','103414','102816']
    # male_subjids = ['100206','102008','101107','100610','100408']
    sample_subjids = ['102311','101915','101006','103414','102816',
                      '100206','102008','101107','100610','100408']
    sample_gender = [1,1,1,1,1,0,0,0,0,0]

    for ii in range(4):
        # full_subjids = mydict[test_session_alias_list[ii]] # for local run
        full_subjids = mydict[test_session_alias_list[ii]].to_numpy()  # for sherlock
        subj_idcs = np.squeeze(np.where(np.isin(full_subjids, sample_subjids)))

        model_session = model_sessions_list[ii]
        test_session = test_sessions_list[ii]
        test_session_alias = test_session_alias_list[ii]
        m = best_split_id[ii]

        site = 'hcp_model_' + model_session + '_index_' + str(m) + '_test_' + test_session_alias
        data_path = 'PROJECT_DIR/results/restfmri/dnn/attributions/'
        data_file = data_path + 'hcp_model_' + model_session + '_index_' + str(m) + '_test_' + test_session + '.npz'

        data = np.load(data_file)
        labels = data['labels']
        data = data['features'] # num_sub x num_roi x num_tp

        percentile = 95 # show top 5% features
        for ss in range(len(subj_idcs)):
            subj_idx = subj_idcs[ss]
            print('subj_list {}, subj_idx {}'.format(subj_idcs, subj_idx))
            if sample_gender[ss] == 0:
                group='male'
            elif sample_gender[ss] == 1:
                group='female'

            subj_data = data[subj_idx] # num_roi x num_tp
            print('subj_data shape {}'.format(subj_data.shape))
            features = np.abs(np.median(subj_data, axis=1))# num_roi
            print('features shape {}'.format(features.shape))
            percentiles = np.where(features >= np.percentile(features, percentile))
            features_idcs = percentiles[0]  # includes all indices (rois) at which position the values are above the cutoff
            save_indiv_nifti(sample_subjids[ss], features_idcs, features, output_path, group, site, percentile)

