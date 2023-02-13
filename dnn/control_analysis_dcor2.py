import warnings
warnings.filterwarnings("ignore")
from utilityFunctions import *
import pickle
from scipy import stats


if __name__ == "__main__":

    # which sessions models were trained on
    model_sessions_list = ['LR_S1', 'LR_S1', 'RL_S1', 'RL_S1']
    # which normz non-windowed entire dataset were used for generating feature attribution
    test_sessions_list = ['REST1_LR', 'REST2_LR', 'REST1_RL', 'REST2_RL']
    test_session_alias_list =['LR_S1', 'LR_S2', 'RL_S1', 'RL_S2']

    subjid_file = 'PROJECT_DIR/results/restfmri/dnn/basic_info/subjid_HCP_sessions.pkl'
    # get subjid for all hcp sessions
    with open(subjid_file, 'rb') as handle:
        mydict = pickle.load(handle)

    headmov_file = 'PROJECT_DIR/results/restfmri/dnn/basic_info/HeadMov_HCP_sessions.pkl'
    # get headmov for all hcp sessions
    with open(headmov_file, 'rb') as handle:
        mydict_headmov = pickle.load(handle)

    # output
    output_path = 'PROJECT_DIR/results/restfmri/dnn/control_analysis/'
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    data_path = 'PROJECT_DIR/results/restfmri/dnn/attributions/'

    for ii in range(4): # 4 sessions
        model_session = model_sessions_list[ii]
        test_session = test_sessions_list[ii]

        dcor2M_list = []
        dcor2F_list = []
        for m in range(5): # 5folds

            site = 'hcp_model_' + model_session + '_index_' + str(m) + '_test_' + test_session
            data_file = data_path + site + '.npz'
            data = np.load(data_file)  # load feature attributions (#sub x #roi x #tp), labels etc.
            fingerprint = data['features']
            labels = data['labels']
            # get median across tp (#sub x #roi) - this is individual-level fingerprint
            fingerprint = np.squeeze(np.median(fingerprint, axis=2))
            print("fingerprint shape {}".format(fingerprint.shape))

            idxMale = np.argwhere(labels == 0)  # get male subjects' indices
            idxFemale = np.argwhere(labels == 1)  # get female subjects' indices

            subjlist = mydict[test_session_alias_list[ii]].to_numpy()
            headmov = mydict_headmov[test_session_alias_list[ii]].to_numpy()
            # print(len(subjlist), len(headmov)) # e.g. S1: 1073, 1073
            # print(fingerprint.shape) # e.g. S1: 1073 x 246

            # dcor2
            idxMale = np.squeeze(idxMale)
            idxFemale = np.squeeze(idxFemale)
            # print("fingerprint shape {}".format(fingerprint[idxMale].shape))
            # print("headmov shape {}".format(np.array(headmov[idxMale]).reshape(-1, 1).shape))
            # print("fingerprint shape {}".format(fingerprint[idxFemale].shape))
            # print("headmov shape {}".format(np.array(headmov[idxFemale]).reshape(-1, 1).shape))

            print("\nmodel_session {}, split {}, test_session {}\n".format(model_session, m, test_session))
            dcor2M = distcorr2(fingerprint[idxMale], np.array(headmov[idxMale]).reshape(-1, 1))
            print("Male: dcor2 between FD and features {}".format(dcor2M))
            dcor2F = distcorr2(fingerprint[idxFemale], np.array(headmov[idxFemale]).reshape(-1,1))
            print("Female: dcor2 between FD and features {}\n\n".format(dcor2F))

            dcor2M_list.append(dcor2M)
            dcor2F_list.append(dcor2F)

        print("Male: dcor2 between FD and features {} ({})".format(np.mean(np.array(dcor2M_list)), np.std(np.array(dcor2M_list))))
        print("Female: dcor2 between FD and features {} ({})".format(np.mean(np.array(dcor2F_list)), np.std(np.array(dcor2F_list))))
