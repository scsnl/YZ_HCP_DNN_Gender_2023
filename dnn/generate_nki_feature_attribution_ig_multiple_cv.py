import torch
from torch.utils.data import DataLoader,TensorDataset
import pandas as pd
from modelClasses import *
from utilityFunctions import *
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


if __name__ == "__main__":

    # which sessions models were trained on
    model_sessions_list = ['LR_S1','RL_S1']
    # which normz non-windowed entire dataset were used for generating feature attribution
    test_sessions_list = ['645', '645']
    num_5folds = 100

    # output
    output_path = 'PROJECT_DIR/results/restfmri/dnn/attributions/multiple_cv/'
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    for ii in range(2):
        model_session = model_sessions_list[ii]
        test_session = test_sessions_list[ii]
        # path_to_testdata = 'PUBLIC_DATA_DIR/hcp/restfmri/timeseries/group_level/brainnetome/normz/hcp_run-rfMRI_' + test_session + '_brainnetome_mean_regMov-6param_wmcsf_dt1_bpf008-09_normz_246ROIs.pklz'
        path_to_testdata = 'PUBLIC_DATA_DIR/nkirs/restfmri/timeseries/group_level/brainnetome/normz/nkirs_site-nkirs_run-rest_' + test_session + '_brainnetome_mean_regMov-6param_wmcsf_dt1_bpf008-09_normz_246ROIs.pklz'
        print('model session: {}; test session: {}'.format(model_session, test_session))

        # load & clean test data
        datao = pd.read_pickle(path_to_testdata)
        datao.drop(datao[datao['percentofvolsrepaired'] > 10].index, inplace=True)
        datao.drop(datao[datao['mean_fd'] > 0.5].index, inplace=True)
        print("data shape after head motion exclusion: {}".format(datao.shape))
        datao.drop(datao[datao['age'] < 22].index, inplace=True)  # yz added
        datao.drop(datao[datao['age'] > 35].index, inplace=True)  # yz added
        print("data shape after age exclusion: {}".format(datao.shape))

        # add the following lines to check length of data across subjects
        datao.reset_index(inplace=True)
        dlen = [len(datao['data'][idx]) for idx in range(datao.shape[0])]
        df = pd.DataFrame(dlen, columns=['length'])
        # print("unique length of data: {}".format(df['length'].value_counts()))
        
        # print("data shape after rest_index: {}".format(datao.shape))
        data_sel = [idx for idx in range(datao.shape[0]) if len(datao['data'][idx]) == 900]
        # print('length of data_sel is {}'.format(len(data_sel)))
        datao = datao.iloc[data_sel]
        # print(datao.shape)
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
        del datao

        print("test data dimension before reshape: {}".format(data.shape))
        data = reshapeData(data)
        print("test data dimension after reshape: {}".format(data.shape))

        # prepare test data for data loader
        input_tensor_test = torch.from_numpy(data).type(torch.FloatTensor)
        label_tensor_test = torch.from_numpy(labels)
        dataset_test = TensorDataset(input_tensor_test, label_tensor_test)

        # load test data into the loader
        test_loader = DataLoader(dataset=dataset_test, batch_size=data.shape[0], shuffle=False)

        for jj in range(num_5folds):
            print("the {} 5folds".format(jj))

            for m in range(5): # 5 splits/folds (5 models); get features attributions for each model
                path_to_models = 'PROJECT_DIR/data/imaging/roi/outputs/saved_models_multiple_cv/models_cv_' + model_session + '/'
                fname_model = path_to_models + 'ff_' + str(jj) + '_model_hcp_brainnetome_mean_' + model_session + '_normz_window_CV_train_valid_test_window_index_%s.pt' % str(m)
                print("model name: {}".format(fname_model))

                model = ConvNet()
                model.load_state_dict(torch.load(fname_model))

                use_cuda = True #False
                if use_cuda and torch.cuda.is_available():
                    model.cuda()

                # # making prediction on test data
                # model.eval()
                # predictions = []
                # with torch.no_grad():
                #     correct = 0
                #     total = 0
                #     for images, labels in test_loader:
                #         if use_cuda:  # yz added to avoid gpu runtime error (guided by Carlo)
                #             images = images.cuda()
                #             labels = labels.cuda()
                #         outputs = model(images)
                #         _, predicted = torch.max(outputs.data, 1)
                #         total += labels.size(0)
                #         correct += (predicted == labels).sum().item()
                #         predictions.append(predicted.cpu().detach().numpy())
                #     predictions = np.concatenate(predictions)
                #     print('Split/Fold {}: Test Accuracy of the model on the  test data: {} %'.format(str(m), (correct / total) * 100))

                '''Feature Attributions (on test data)'''
                input_tensor_test = input_tensor_test.cuda() # do not use cuda as it leads to runtime error
                features = []

                print(label_tensor_test.size())
                for i in range(len(input_tensor_test)):
                    # print(label_tensor_test[i], type(label_tensor_test[i]))
                    attr = getInputAttributions(model, input_tensor_test[i].unsqueeze_(-1).permute(2,0,1), label_tensor_test[i]) # female = 1; male = 0
                    features.append(attr)

                features = np.concatenate(features)

                feature_attribution_fname = output_path + 'ff' + str(jj) + '_hcp_model_' + model_session + '_index_' + str(m) + '_test_nki_' + test_session + '.npz'
                # np.savez(feature_attribution_fname, features=features, predictions=predictions, labels=label_tensor_test.numpy())
                np.savez(feature_attribution_fname, features=features, labels=label_tensor_test.numpy())

