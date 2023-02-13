import numpy as np
import pickle
from scipy import stats


if __name__=="__main__":

	best_split_id = [2, 2, 2, 2]  # correspoding to split index for S1, S1, S3, and S3
	hcp_sessions_list = ['LR_S1', 'LR_S1', 'RL_S1', 'RL_S1']  # models S1, S1, S3, S3
	test_session_list = ['REST1_LR', 'REST2_LR', 'REST1_RL', 'REST2_RL']  # test S1, S2, S3, S4
	test_session_alias_list = ['LR_S1', 'LR_S2', 'RL_S1', 'RL_S2']

	dnn_results_path = 'PROJECT_DIR/results/restfmri/dnn/'
	# dnn_results_path = '/Users/zhangyuan/Desktop/Sherlock/projects/menon/2021_HCP_Gender/results/restfmri/dnn/basic_info/'
	subjid_file = dnn_results_path + 'basic_info/subjid_HCP_sessions.pkl'

	for ii in [0, 2]: # corresponds to HCP S1, S3

		model_session1 = hcp_sessions_list[ii]
		m_session1 = best_split_id[ii] # get the best split id
		test_session1 = test_session_list[ii] # to get feature attribution of this session
		test_session1_alias = test_session_alias_list[ii] # for getting subjid of this session

		model_session2 = hcp_sessions_list[ii+1]
		m_session2 = best_split_id[ii+1] # get the best split id
		test_session2 = test_session_list[ii+1] # to get feature attribution of this session
		test_session2_alias = test_session_alias_list[ii+1] # for getting subjid of this session

		# get subjid for all hcp sessions
		with open(subjid_file, 'rb') as handle:
			dict = pickle.load(handle)
		# print(dict['LR_S1'][0])
		# print(dict['LR_S2'][0])
		# print(dict['RL_S1'][0])
		# print(dict['RL_S2'][0])

		if model_session1 == model_session2 and m_session1 == m_session2:
			output_fname = dnn_results_path + 'tsne_plots/fingerprints_commonSubj_by_ROIs_HCP_model_' + model_session1 + '_index_' + str(m_session1) \
						   + '_test1_' + test_session1_alias + '_test2_' + test_session2_alias + '.npz'
		else:
			print("session 1 and session 2 should be same")

		# load fingerprints of the two test sessions
		fingerprints_path = dnn_results_path + 'attributions/'
		session1data = np.load(fingerprints_path + 'hcp_model_' + model_session1 + '_index_' + str(m_session1) + '_test_' + test_session1 + '.npz')
		session1fingerprint = session1data['features']
		print("session 1 fingerprint {}".format(session1fingerprint.shape))
		session1labels = session1data['labels']
		print("session 1 label {}".format(len(session1labels)))
		session1subjids = dict[test_session1_alias].to_numpy()
		print("session 1 subj {}".format(session1subjids.shape))

		session2data = np.load(fingerprints_path + 'hcp_model_' + model_session2 + '_index_' + str(m_session2) + '_test_' + test_session2 + '.npz')
		session2fingerprint = session2data['features']
		print("session 2 fingerprint {}".format(session2fingerprint.shape))
		session2labels = session2data['labels']
		print("session 2 label {}".format(len(session2labels)))
		session2subjids = dict[test_session2_alias].to_numpy()
		print("session 2 subj {}".format(session2subjids.shape))


		session1features = np.median(session1fingerprint, axis=2)
		session2features = np.median(session2fingerprint, axis=2)

		commonsubjids, session1_ix, session2_ix = np.intersect1d(session1subjids, session2subjids, return_indices=True)
		print(type(session1_ix), len(session1_ix))
		print(type(session2_ix), len(session2_ix))
		session1subjids = session1subjids[session1_ix]
		session2subjids = session2subjids[session2_ix]
		print("# common subjs for both test dataset: s1: {} s2: {}".format(len(session1subjids), len(session1subjids)))

		# vertically stack two sessions' data
		features = np.vstack((session1features[session1_ix,], session2features[session2_ix,]))
		# calculate within and between session distances/correlations between subjects
		d = np.corrcoef(features)
		# get cross-session distances/correlations
		dd = d[0:np.shape(session1_ix)[0], np.shape(session1_ix)[0]:np.shape(d)[0]]
		# find the index of max value for each row
		ixmatched = dd.argmax(axis=1)
		# check if the highest correlation is within the same subject
		# print("type session1subjids[0] {}".format(type(session1subjids[0])))
		diff = session1subjids.astype(np.int) - session2subjids[ixmatched].astype(np.int)
		# if diff = 0, yes/matched; diff !=0, no/not matched
		print('percent of matched ids across session')
		print('{}% of subjects showing the highest correlation with self across sessions'.format(np.shape(np.argwhere(diff == 0))[0]*100/np.shape(diff)[0]))

		withinSubj = np.zeros(np.shape(session1subjids)[0])
		betweenSubj = np.zeros(np.shape(session1subjids)[0])

		for ii in range(np.shape(session1subjids)[0]):
			ix = np.argwhere(session2subjids == session1subjids[ii])
			# get intra-individual distances/correlations for the ii subject
			withinSubj[ii] = dd[ii, ix]
			ixAll = np.array(range(0, np.shape(session1subjids)[0]))
			# get mean of cross-individual distances/correlations for the ii subject
			betweenSubj[ii] = np.mean(dd[ii, np.argwhere(ixAll != ix)])

		print('Comparing between subject and across subjects corr')
		print('mean within: {}, std {}, n {}'.format(np.mean(withinSubj),np.std(withinSubj),len(withinSubj)))
		print('mean between: {}, std {}, n {}'.format(np.mean(betweenSubj),np.std(betweenSubj),len(betweenSubj)))
		print(stats.ttest_rel(withinSubj, betweenSubj))

		np.savez(output_fname, features=features, diff=diff, commonsubjids=commonsubjids)