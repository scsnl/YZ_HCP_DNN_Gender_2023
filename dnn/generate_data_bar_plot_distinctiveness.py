import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE


if __name__=="__main__":
    best_split_id = [2, 2, 2, 2, 2, 2]  # correspoding to split index for S1, S1, S3, S3, S1, S3
    hcp_sessions_list = ['LR_S1', 'LR_S1', 'RL_S1', 'RL_S1', 'LR_S1', 'RL_S1']  # models S1, S1, S3, S3, S1, S3
    test_session_list = ['LR_S1', 'LR_S2', 'RL_S1', 'RL_S2', 'nki_645', 'nki_645']  # test S1, S2, S3, S4, NKI, NKI

    data_path = '/Users/zhangyuan/Google Drive/2021_HCP_Gender_DNN/data/npz/'
    # the npz fingerprints files were same as that saved on oak at
    # PROJECT_DIR/results/restfmri/dnn/tsne_plots/

    for ss in range(6):

        model_session = hcp_sessions_list[ss]
        m = best_split_id[ss]
        test_session = test_session_list[ss]
        data_file = data_path + 'fingerprints_Subj_by_ROIs_HCP_model_' + model_session + '_index_' + str(m) + '_test_' + test_session + '.npz'
        print(data_file)

        data = np.load(data_file)
        fingerprint = data['features']
        labels = data['labels']
        print(fingerprint.shape)
        print(labels.shape)

        idxMale = np.argwhere(labels == 0)  # get male subjects' indices
        idxFemale = np.argwhere(labels == 1)  # get female subjects' indices
        # get group-level fingerprint
        grpFingerprintMale = np.median(fingerprint[idxMale, :], axis=0)
        grpFingerprintFemale = np.median(fingerprint[idxFemale, :], axis=0)

        # calculate distance of fingerprints between a male individual and group
        distFromMaletoGrpMale = np.zeros(np.shape(idxMale)[0])
        distFromMaletoGrpFemale = np.zeros(np.shape(idxMale)[0])
        for ii in range(np.shape(idxMale)[0]):
            distFromMaletoGrpMale[ii] = np.corrcoef(fingerprint[idxMale[ii]], grpFingerprintMale)[0, 1]
            distFromMaletoGrpFemale[ii] = np.corrcoef(fingerprint[idxMale[ii]], grpFingerprintFemale)[0, 1]

        Mdist = np.concatenate((np.reshape(distFromMaletoGrpMale,(-1,1)), np.reshape(distFromMaletoGrpFemale,(-1,1))),axis=1)
        print(distFromMaletoGrpMale.shape)
        print(distFromMaletoGrpFemale.shape)
        print(Mdist.shape) # #males x 2
        # print(distFromMaletoGrpMale[0:4])
        # print(distFromMaletoGrpFemale[0:4])
        # print(Mdist[0:4,])
        outputf = data_path + 'distanceM_HCP_model_' + model_session + '_index_' + str(m) + '_test_' + test_session + '.csv'
        np.savetxt(outputf, Mdist, delimiter=",")

        # calculate distance of fingerprints between a female individual and group
        distFromFemaletoGrpFemale = np.zeros(np.shape(idxFemale)[0])
        distFromFemaletoGrpMale = np.zeros(np.shape(idxFemale)[0])
        for ii in range(np.shape(idxFemale)[0]):
            distFromFemaletoGrpFemale[ii] = np.corrcoef(fingerprint[idxFemale[ii]], grpFingerprintFemale)[0, 1]
            distFromFemaletoGrpMale[ii] = np.corrcoef(fingerprint[idxFemale[ii]], grpFingerprintMale)[0, 1]

        Fdist = np.concatenate((np.reshape(distFromFemaletoGrpFemale, (-1, 1)), np.reshape(distFromFemaletoGrpMale, (-1, 1))), axis=1)
        print(distFromFemaletoGrpFemale.shape)
        print(distFromFemaletoGrpMale.shape)
        print(Fdist.shape)  # #females x 2
        outputf = data_path + 'distanceF_HCP_model_' + model_session + '_index_' + str(
            m) + '_test_' + test_session + '.csv'
        np.savetxt(outputf, Fdist, delimiter=",")
