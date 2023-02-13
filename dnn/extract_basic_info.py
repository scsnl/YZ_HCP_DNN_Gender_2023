import numpy as np
import pandas as pd
import os
import pickle

if __name__ == "__main__":

    hcp_sessions_list = ['REST1_LR', 'REST2_LR', 'REST1_RL', 'REST2_RL']
    cv_sessions_list = ['LR_S1', 'LR_S2', 'RL_S1', 'RL_S2']

    # to save subject ids and head motions across sessions to files
    output_path = 'PROJECT_DIR/results/restfmri/dnn/basic_info/'
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    subjid_file = output_path + 'subjid_HCP_sessions.pkl'
    subjectID = {}  # a dictionary of subjectID for all HCP sessions
    fd_file = output_path + 'HeadMov_HCP_sessions.pkl'
    HeadMov = {} # a dictionary of head motions for all HCP sessions

    for ii in range(4): # four HCP sessions
        session = hcp_sessions_list[ii]
        ts_path = 'PUBLIC_DATA_DIR/hcp/restfmri/timeseries/group_level/brainnetome/normz/'
        path_to_dataset = ts_path + 'hcp_run-rfMRI_' + session + '_brainnetome_mean_regMov-6param_wmcsf_dt1_bpf008-09_normz_246ROIs.pklz'
        print(ts_path)

        cv_session = cv_sessions_list[ii]
        cv_path = 'PROJECT_DIR/data/imaging/roi/cv_datasets/cv_dataset_' + cv_session
        cv_train_list_index = cv_path + '/train_list_' + cv_session + '_index.npy'
        cv_test_list_index = cv_path + '/test_list_' + cv_session + '_index.npy'

        print("HCP session: {}, {}\n".format(session, cv_session))
        datao = pd.read_pickle(path_to_dataset)
        print("data loaded.")
        datao.drop(datao[datao['percentofvolsrepaired']>10].index, inplace=True)
        datao.drop(datao[datao['mean_fd']>0.5].index, inplace=True)
        datao = datao.reset_index()
        data = np.asarray([np.asarray(lst)[:, :] for lst in datao.data])
        labels_gender = datao['gender']
        subjid = datao['subject_id']

        subjectID[cv_sessions_list[ii]] = subjid
        HeadMov[cv_sessions_list[ii]] = datao['mean_fd']

        labels = []
        for i in labels_gender:
          if i == 'male':
            labels.append(0)
          else:
            labels.append(1)
        labels = np.asarray(labels)

        print("total number of subjects for hcp session {}: {}\n".format(session, len(labels)))
        print("female/male for hcp session {}: {}/{}\n".format(session, sum(labels), len(labels)-sum(labels)))

        # get # of train and test samples
        train = np.load(cv_train_list_index,allow_pickle=True)
        test = np.load(cv_test_list_index, allow_pickle=True)
        print("train (HCP session {}):\n fold 1: {}, fold 2: {}, fold 3: {}, fold 4: {}, fold 5: {}"\
              .format(session, train[0].shape[0],train[1].shape[0],train[2].shape[0],train[3].shape[0],train[4].shape[0]))
        print("test (HCP session {}):\n fold 1: {}, fold 2: {}, fold 3: {}, fold 4: {}, fold 5: {}\n" \
              .format(session, test[0].shape[0], test[1].shape[0], test[2].shape[0], test[3].shape[0], test[4].shape[0]))
        print("test (HCP session {}) F/M:\n fold 1: {}/{}, fold 2: {}/{}, fold 3: {}/{}, fold 4: {}/{}, fold 5: {}/{}\n" \
              .format(session, sum(labels[test[0]]), len(labels[test[0]])-sum(labels[test[0]]),
                      sum(labels[test[1]]), len(labels[test[1]])-sum(labels[test[1]]),
                      sum(labels[test[2]]), len(labels[test[2]])-sum(labels[test[2]]),
                      sum(labels[test[3]]), len(labels[test[3]])-sum(labels[test[3]]),
                      sum(labels[test[4]]), len(labels[test[4]])-sum(labels[test[4]])))

# save subject ids and head motion across sessions to files
with open(subjid_file, 'wb') as handle:
    pickle.dump(subjectID, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open(fd_file, 'wb') as handle:
    pickle.dump(HeadMov, handle, protocol=pickle.HIGHEST_PROTOCOL)
