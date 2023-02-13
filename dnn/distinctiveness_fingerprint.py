import numpy as np
from scipy import stats


if __name__ == "__main__":

	best_split_id = [2, 2, 2, 2]  # correspoding to split index for S1, S1, S3, and S3
	hcp_sessions_list = ['LR_S1', 'LR_S1', 'RL_S1', 'RL_S1']  # models S1, S1, S3, S3
	test_session_list = ['REST1_LR', 'REST2_LR', 'REST1_RL', 'REST2_RL']  # test S1, S2, S3, S4
	test_session_alias_list = ['LR_S1', 'LR_S2', 'RL_S1', 'RL_S2']
	# we want the combination of hcp_session and test_session to get the features file

	dnn_results_path = 'PROJECT_DIR/results/restfmri/dnn/tsne_plots/'

	data_path = 'PROJECT_DIR/results/restfmri/dnn/attributions/'
	for ss in range(4):
		model_session = hcp_sessions_list[ss]
		m = best_split_id[ss]
		test_session = test_session_list[ss]
		data_file = data_path + 'hcp_model_' + model_session + '_index_' + str(m) + '_test_' + test_session + '.npz'
		print("\n\ndata file {}".format(data_file))
		output_fname = dnn_results_path + 'fingerprints_Subj_by_ROIs_HCP_model_' + model_session + '_index_' + str(m) + '_test_' + test_session_alias_list[ss] + '.npz'
		print("output file {}".format(output_fname))

		data = np.load(data_file) # load feature attributions (#sub x #roi x #tp), labels etc.
		fingerprint = data['features']
		labels = data['labels']
		print(len(labels))

		# get median across tp (#sub x #roi) - this is individual-level fingerprint
		fingerprint = np.median(fingerprint, axis=2)
		np.savez(output_fname, features=fingerprint, labels=labels)

		idxMale = np.argwhere(labels == 0) # get male subjects' indices
		idxFemale = np.argwhere(labels == 1) # get female subjects' indices
		# get group-level fingerprint
		grpFingerprintMale = np.median(fingerprint[idxMale, :], axis=0)
		grpFingerprintFemale = np.median(fingerprint[idxFemale, :], axis=0)

		# calculate distance of fingerprints between a male individual and group
		distFromMaletoGrpMale = np.zeros(np.shape(idxMale)[0])
		distFromMaletoGrpFemale = np.zeros(np.shape(idxMale)[0])
		for ii in range(np.shape(idxMale)[0]):
			distFromMaletoGrpMale[ii] = np.corrcoef(fingerprint[idxMale[ii]], grpFingerprintMale)[0, 1]
			distFromMaletoGrpFemale[ii] = np.corrcoef(fingerprint[idxMale[ii]], grpFingerprintFemale)[0, 1]

		print("\nmodel_session {}, split {}, test_session {}\n".format(model_session, m, test_session))
		print('Comparing between individual male to group male with between individual male to group female distances')
		print("distFromMaletoGrpMale: mean {}, std {}, n {}".format(np.mean(distFromMaletoGrpMale),np.std(distFromMaletoGrpMale), len(distFromMaletoGrpMale)))
		print("distFromMaletoGrpFemale: mean {}, std {}, n {}".format(np.mean(distFromMaletoGrpFemale),np.std(distFromMaletoGrpFemale),len(distFromMaletoGrpFemale)))
		print(stats.ttest_rel(distFromMaletoGrpMale, distFromMaletoGrpFemale))

		# calculate distance of fingerprints between a female individual and group
		distFromFemaletoGrpFemale = np.zeros(np.shape(idxFemale)[0])
		distFromFemaletoGrpMale = np.zeros(np.shape(idxFemale)[0])
		for ii in range(np.shape(idxFemale)[0]):
			distFromFemaletoGrpFemale[ii] = np.corrcoef(fingerprint[idxFemale[ii]], grpFingerprintFemale)[0, 1]
			distFromFemaletoGrpMale[ii] = np.corrcoef(fingerprint[idxFemale[ii]], grpFingerprintMale)[0, 1]

		print("\nmodel_session {}, split {}, test_session {}\n".format(model_session, m, test_session))
		print('Comparing between individual female to group female with between individual female to group male distances')
		print("distFromFemaletoGrpFemale: mean {}, std {}, n {}".format(np.mean(distFromFemaletoGrpFemale),np.std(distFromFemaletoGrpFemale),len(distFromFemaletoGrpFemale)))
		print("distFromFemaletoGrpMale: mean {}, std {}, n {}".format(np.mean(distFromFemaletoGrpMale),np.std(distFromFemaletoGrpMale),len(distFromFemaletoGrpMale)))
		print(stats.ttest_rel(distFromFemaletoGrpFemale, distFromFemaletoGrpMale))

		# distance between each pair of subjects
		dist = np.corrcoef(fingerprint)
		# distance between male subjects and between male and female subjects
		distFromMaletoMale = np.zeros(np.shape(idxMale)[0])
		distFromMaletoFemale = np.zeros(np.shape(idxMale)[0])
		for ii in range(np.shape(idxMale)[0]):
			distFromMaletoMale[ii] = np.mean(dist[idxMale[ii], idxMale])
			distFromMaletoFemale[ii] = np.mean(dist[idxMale[ii], idxFemale])

		print("\nmodel_session {}, split {}, test_session {}\n".format(model_session, m, test_session))
		print('Comparing within male to between male and female distances')
		print(np.mean(distFromMaletoMale))
		print(np.mean(distFromMaletoFemale))
		print(stats.ttest_rel(distFromMaletoMale, distFromMaletoFemale))

		# distance between female subjects and between female and male subjects
		distFromFemaletoFemale = np.zeros(np.shape(idxFemale)[0])
		distFromFemaletoMale = np.zeros(np.shape(idxFemale)[0])
		for ii in range(np.shape(idxFemale)[0]):
			distFromFemaletoFemale[ii] = np.mean(dist[idxFemale[ii], idxFemale])
			distFromFemaletoMale[ii] = np.mean(dist[idxFemale[ii], idxMale])

		print("\nmodel_session {}, split {}, test_session {}\n".format(model_session, m, test_session))
		print('Comparing within female to between female and male distances')
		print(np.mean(distFromFemaletoFemale))
		print(np.mean(distFromFemaletoMale))
		print(stats.ttest_rel(distFromFemaletoFemale, distFromFemaletoMale))