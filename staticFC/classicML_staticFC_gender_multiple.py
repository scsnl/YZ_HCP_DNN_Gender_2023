import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, classification_report
from sklearn import preprocessing, linear_model, model_selection, svm, tree, ensemble, neighbors
import pickle


def data_cleaning_hcp(path_to_dataset):
    # load & clean data
    datao = pd.read_pickle(path_to_dataset)
    datao.drop(datao[datao['percentofvolsrepaired'] > 10].index, inplace=True)
    datao.drop(datao[datao['mean_fd'] > 0.5].index, inplace=True)
    datao = datao.reset_index()
    data = np.asarray([np.asarray(lst)[:, :] for lst in datao.data])
    labels_gender = datao['gender']
    # subjid = datao['subject_id']

    labels = []
    for i in labels_gender:
        if i == 'male':
            labels.append(0)
        else:
            labels.append(1)
    labels = np.asarray(labels)
    # print("data dimension: {}".format(data.shape)) # no_subj, no_ts, no_roi
    return data, labels


def data_cleaning_nkirs(path_to_dataset):
    # Load and clean
    datao = pd.read_pickle(path_to_dataset)
    datao.drop(datao[datao['percentofvolsrepaired'] > 10].index, inplace=True)
    datao.drop(datao[datao['mean_fd'] > 0.5].index, inplace=True)
    datao.drop(datao[datao['age'] < 22].index, inplace=True)
    datao.drop(datao[datao['age'] > 35].index, inplace=True)
    datao.reset_index(inplace=True)

    data_sel = [idx for idx in range(datao.shape[0]) if len(datao['data'][idx]) == 900]
    datao = datao.iloc[data_sel]
    datao.reset_index(inplace=True)

    data = np.asarray([np.asarray(lst)[:, :] for lst in datao.data])
    labels_gender = datao['gender']
    subjid = datao['subject_id']

    labels = []
    for i in labels_gender:
        if i == 'male':
            labels.append(0)
        else:
            labels.append(1)
    labels = np.asarray(labels)
    print("data dimension: {}".format(data.shape))

    return data, labels

def get_features_labels(path_to_dataset, site):
    if site == 'hcp':
        data, labels = data_cleaning_hcp(path_to_dataset)
    elif site == 'nkirs':
        data, labels = data_cleaning_nkirs(path_to_dataset)

    print("data dimension: {}".format(data.shape))  # no_subj, no_ts, no_roi
    print("labels dimension: {}".format(labels.shape))

    # generate static FC features
    no_subjs, no_ts, no_rois = data.shape
    data_fcz = np.empty((no_subjs, int(no_rois * (no_rois - 1) / 2)))
    print('data_fcz dimension {}'.format(data_fcz.shape))

    for subj in range(no_subjs):
        # print(subj)
        x_subj = data[subj, :, :]
        df_subj = pd.DataFrame(x_subj)
        fc_subj = df_subj.corr('pearson')  # get correlation matrix
        fc_subj = fc_subj.to_numpy()
        # get upper tri elements of the FC matrix and apply fisher z transformation
        data_fcz[subj, :] = np.arctanh(fc_subj[np.triu_indices(fc_subj.shape[0], k=1)])

    return data_fcz, labels

if __name__ == '__main__':

    hcpData = 'PUBLIC_DATA_DIR/hcp/restfmri/timeseries/group_level/brainnetome/' \
                      'normz/hcp_run-rfMRI_REST1_RL_brainnetome_mean_regMov-6param_wmcsf_dt1_bpf008-09_normz_246ROIs.pklz'
    testData1 = 'PUBLIC_DATA_DIR/hcp/restfmri/timeseries/group_level/brainnetome/' \
                      'normz/hcp_run-rfMRI_REST2_RL_brainnetome_mean_regMov-6param_wmcsf_dt1_bpf008-09_normz_246ROIs.pklz'
    testData2 = 'PUBLIC_DATA_DIR/nkirs/restfmri/timeseries/group_level/brainnetome/normz/nkirs_site-nkirs_run-rest_645_brainnetome_mean_regMov-6param_wmcsf_dt1_bpf008-09_normz_246ROIs.pklz'

    print("models trained on HCP Session 3, tested on HCP Session 3, Session 4, and NKI-RS")
    model_ss = 'HCP_Session3'

    output_path = 'PROJECT_DIR/results/restfmri/dnn/sFC/'
    f1 = output_path + 'sFC_HCP_Session3.npz'
    f2 = output_path + 'sFC_HCP_Session4.npz'
    f3 = output_path + 'sFC_NKI-RS.npz'

    # data1 = np.load(f1)
    # data2 = np.load(f2)
    # data3 = np.load(f3)
    #
    # data_features = data1['features']
    # labels = data1['labels']
    # testData1_features = data2['features']
    # testData1_labels = data2['labels']
    # testData2_features = data3['features']
    # testData2_labels = data3['labels']


    #####################################
    K = 5
    # Perform classification and compute classification performance metrics
    accuracy = np.zeros(K)
    precision = np.zeros(K)
    recall = np.zeros(K)
    f1score = np.zeros(K)
    test1_accuracy = np.zeros(K)
    test1_precision = np.zeros(K)
    test1_recall = np.zeros(K)
    test1_f1score = np.zeros(K)
    test2_accuracy = np.zeros(K)
    test2_precision = np.zeros(K)
    test2_recall = np.zeros(K)
    test2_f1score = np.zeros(K)

    # prepare models
    models = []
    models.append(('linSVM', svm.SVC(kernel='linear')))
    models.append(('KNN', (neighbors.KNeighborsClassifier())))
    models.append(('DT', tree.DecisionTreeClassifier()))
    models.append(('LR', linear_model.LogisticRegression()))
    # models.append(('rbfSVM', svm.SVC(kernel='rbf')))
    models.append(('RC', linear_model.RidgeClassifier(alpha=0.5)))
    models.append(('LASSO', linear_model.LogisticRegression(penalty='l1', solver='liblinear')))
    # models.append(('ELNet', linear_model.LogisticRegression(penalty='elasticnet', l1_ratio=0.5, solver='saga')))
    models.append(('RF', ensemble.RandomForestClassifier()))

    # get features and labels (0: male, 1: female)
    data_features, labels = get_features_labels(hcpData, 'hcp')
    testData1_features, testData1_labels = get_features_labels(testData1, 'hcp')
    testData2_features, testData2_labels = get_features_labels(testData2, 'nkirs')
    np.savez(f1, features=data_features, labels=labels)
    np.savez(f2, features=testData1_features, labels=testData1_labels)
    np.savez(f3, features=testData2_features, labels=testData2_labels)

    # 5-fold
    kf = StratifiedKFold(n_splits=5, random_state=6666, shuffle=True)
    train_index_list = []
    val_index_list = []
    for train_index, val_index in kf.split(data_features, labels):
        train_index_list.append(train_index)
        val_index_list.append(val_index)

    for name, model in models:
        print('*** Classifier: ', name)
        for foldid in range(K):
            print('** Evaluating: Fold {}'.format(foldid))
            
            fname_model = output_path + 'model_' + name + '_' + model_ss + '_' + str(foldid) + '.sav'
            print('model name {}'.format(fname_model))

            idx = foldid
            train_index = train_index_list[idx]
            val_index = val_index_list[idx]
            x_train, x_val = data_features[train_index], data_features[val_index]
            y_train, y_val = labels[train_index], labels[val_index]

            x_test1 = testData1_features
            y_test1 = testData1_labels

            x_test2 = testData2_features
            y_test2 = testData2_labels

            modelfit = model.fit(x_train, y_train)
            y_val_predicted = modelfit.predict(x_val)

            # save the model
            pickle.dump(modelfit, open(fname_model, 'wb'))

            classifreport_dict = classification_report(y_val, y_val_predicted,
                                                       output_dict=True)
            accuracy[foldid] = classifreport_dict['accuracy'] * 100
            precision[foldid] = classifreport_dict['0']['precision'] * 100
            recall[foldid] = classifreport_dict['0']['recall'] * 100
            f1score[foldid] = classifreport_dict['0']['f1-score'] * 100

            y_test1_predicted = modelfit.predict(x_test1)

            test1_classifreport_dict = classification_report(y_test1, y_test1_predicted,
                                                            output_dict=True)
            test1_accuracy[foldid] = test1_classifreport_dict['accuracy'] * 100
            test1_precision[foldid] = test1_classifreport_dict['0']['precision'] * 100
            test1_recall[foldid] = test1_classifreport_dict['0']['recall'] * 100
            test1_f1score[foldid] = test1_classifreport_dict['0']['f1-score'] * 100

            y_test2_predicted = modelfit.predict(x_test2)

            test2_classifreport_dict = classification_report(y_test2, y_test2_predicted,
                                                             output_dict=True)
            test2_accuracy[foldid] = test2_classifreport_dict['accuracy'] * 100
            test2_precision[foldid] = test2_classifreport_dict['0']['precision'] * 100
            test2_recall[foldid] = test2_classifreport_dict['0']['recall'] * 100
            test2_f1score[foldid] = test2_classifreport_dict['0']['f1-score'] * 100

        # performance on training dataset
        print('** Mean HCP (training session) accuracy across 5 folds {0:.2f} +/- {1:.2f} %'.format(
            np.mean(accuracy), np.std(accuracy)))
        print('** Mean HCP (training session) precision across 5 folds {0:.2f} +/- {1:.2f} %'.format(
            np.mean(precision), np.std(precision)))
        print('** Mean HCP (training session) recall across 5 folds {0:.2f} +/- {1:.2f} %'.format(
            np.mean(recall), np.std(recall)))
        print('** Mean HCP (training session) f1-score across 5 folds {0:.2f} +/- {1:.2f} %'.format(
            np.mean(f1score),np.std(f1score)))

        # performance on test1 dataset
        print('** Mean HCP (testing session) accuracy across 5 folds {0:.2f} +/- {1:.2f} %'.format(np.mean(test1_accuracy),np.std(test1_accuracy)))
        print('** Mean HCP (testing session) precision across 5 folds {0:.2f} +/- {1:.2f} %'.format(np.mean(test1_precision),np.std(test1_precision)))
        print('** Mean HCP (testing session) recall across 5 folds {0:.2f} +/- {1:.2f} %'.format(np.mean(test1_recall), np.std(test1_recall)))
        print('** Mean HCP (testing session) f1-score across 5 folds {0:.2f} +/- {1:.2f} %'.format(np.mean(test1_f1score),np.std(test1_f1score)))

        # performance on test2 dataset
        print('** Mean NKI-RS accuracy across 5 folds {0:.2f} +/- {1:.2f} %'.format(np.mean(test2_accuracy),np.std(test2_accuracy)))
        print('** Mean NKI-RS precision across 5 folds {0:.2f} +/- {1:.2f} %'.format(np.mean(test2_precision),np.std(test2_precision)))
        print('** Mean NKI-RS recall across 5 folds {0:.2f} +/- {1:.2f} %'.format(np.mean(test2_recall),np.std(test2_recall)))
        print('** Mean NKI-RS f1 score across 5 folds {0:.2f} +/- {1:.2f} %'.format(np.mean(test2_f1score),np.std(test2_f1score)))

