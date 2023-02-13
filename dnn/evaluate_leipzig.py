import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_fscore_support
from scipy.interpolate import interp1d
import math
from modelClasses import *
from utilityFunctions import *
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

models_sessions_list = ['LR_S1','RL_S1']
# test_sessions_list = ['run-AP_run-01','run-AP_run-02','run-PA_run-01','run-PA_run-02']
# test_sessions_list = ['run-AP_run-01','run-PA_run-01']
test_sessions_list = ['run-AP_run-01']
model_ss_list = ['HCP_S1','HCP_S3']
# test_ss_list = ['leipzig_AP01','leipzig_AP02','leipzig_PA01','leipzig_AP02']
# test_ss_list = ['leipzig_AP01','leipzig_PA01']
test_ss_list = ['leipzig_AP01']

output_path = 'PROJECT_DIR/results/restfmri/dnn/classification_excel_files/'
if not os.path.exists(output_path):
    os.makedirs(output_path)
excel_file = output_path + 'classification_normz_leipzig_noduplicates.xlsx'

for ii in range(2): # model sessions HCP S1, S3
    for jj in range(1): #test session

        models_session = models_sessions_list[ii] # specify HCP session which models were trained on
        path_to_models = 'PROJECT_DIR/data/imaging/roi/outputs/saved_models/models_cv_' + models_session + '/'

        test_session = test_sessions_list[jj] # specify test data
        testdata = 'PUBLIC_DATA_DIR/mpi_leipzig/restfmri/timeseries/group_level/brainnetome/normz/mpi_leipzig_'+ test_session + '_brainnetome_mean_regMov-6param_wmcsf_dt1_bpf008-09_normz_246ROIs.pklz'
        print("apply models (from {}) to test data {}\n".format(models_session, test_session))

        model_ss = model_ss_list[ii] # specify HCP session that models were trained on
        test_ss = test_ss_list[jj] # specify data that models were tested on

        test_acc = np.empty((0,1),float)
        precision = np.empty((0,1),float)
        recall = np.empty((0,1),float)
        f1_score = np.empty((0,1),float)

        # Load test data and data cleaning etc.
        datao = pd.read_pickle(testdata)
        # print(datao.shape)
        # print(datao.columns)
        datao.drop(datao[datao['percentofvolsrepaired']>10].index, inplace=True)
        datao.drop(datao[datao['mean_fd']>0.5].index, inplace=True)
        print("data shape after head motion exclusion: {}".format(datao.shape))

        # here age corresponds to the lower bound of the age bin
        # (e.g., 20 refers to 20-25, 25 refers to 25-30, 30 refers to 30-35, 35 refers to 35-40)
        # keep subjects aged between 22-35
        datao.drop(datao[datao['age'] < 20].index, inplace=True)
        datao.drop(datao[datao['age'] > 30].index, inplace=True)
        print("data shape after age exclusion: {}".format(datao.shape))
        datao.reset_index(inplace=True)

        # remove visit 2 (215 subjects have visit1, 81 of them have visit2)
        datao.drop(datao[datao['visit'].to_numpy(dtype="int32") > 1].index, inplace=True)
        datao.reset_index(inplace=True)

        print(datao.shape)

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

        print("total number of subjects: {}\n".format(len(labels)))
        print("female/male: {}/{}\n".format(sum(labels), len(labels)-sum(labels)))

        del datao

        x_test = data

        # leipzig TR = 1.4 and HCP TR = 0.72
        # # padding one 0 between time points to match HCP TR
        # [N, T, C] = np.shape(x_test)
        # print(x_test.shape)
        # x_test_extend = np.zeros((N, 2 * T, C))
        # x_test_extend[:, 0:2 * T:2, :] = x_test

        # interpolation
        print('before interpolation')
        print(x_test.shape)
        interp = interp1d(np.linspace(0, 1, x_test.shape[1]), x_test, axis=1)
        x_test_extend = interp(np.linspace(0, 1, math.floor(x_test.shape[1] * 1.4 / 0.72)))
        print('after interpolation')
        print(x_test_extend.shape)

        y_test = labels
        # x_test = reshapeData(x_test)
        x_test = reshapeData(x_test_extend) # this is for padding 0 or interpolation
        print(x_test.shape)
        print('Data loading completed')


        input_tensor_test = torch.from_numpy(x_test).type(torch.FloatTensor)
        label_tensor_test = torch.from_numpy(y_test)
        dataset_test1 = TensorDataset(input_tensor_test, label_tensor_test)

        # Load Test data into the loader
        test_loader = DataLoader(dataset=dataset_test1, batch_size=x_test.shape[0], shuffle=False)

        num_classes = 2
        for m in range(5):
            fname_model = path_to_models + 'model_hcp_brainnetome_mean_' + models_session + '_normz_window_CV_train_valid_test_window_index_%s.pt' % str(m)
            print("model name: {}".format(fname_model))

            # define model
            model = ConvNet()
            USE_PRETRAINED_MODEL = True
            use_cuda = False #True

            if USE_PRETRAINED_MODEL:
                print("Using the existing trained model")
                model.load_state_dict(torch.load(fname_model))

            if use_cuda and torch.cuda.is_available():
                model.cuda()

            model.eval()
            with torch.no_grad():
                correct = 0
                total = 0
                for images, labels in test_loader:
                    if use_cuda:
                        images = images.cuda()
                        labels = labels.cuda()
                    outputs = model(images)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

                print('Test Accuracy of the model on the  test data: {} %'.format((correct / total) * 100))
                test_acc = np.append(test_acc, 100 * correct / total)

            # print fold m results
            print(classification_report(labels.detach().cpu(),predicted.detach().cpu()))

        #     # print fold m labels and predicted labels
        #     act_labels = labels.detach().cpu()
        #     print(type(act_labels))
        #     print(act_labels)
        #     pred_labels = predicted.detach().cpu()
        #     print(type(pred_labels))
        #     print(pred_labels)
        #

            # print avg results for a fold
            report = precision_recall_fscore_support(labels.detach().cpu(), predicted.detach().cpu())
            print("precision: {},{}\n".format(np.mean(report[0].round(2)), np.std(report[0].round(2))))
            print("recall: {},{}\n".format(np.mean(report[1].round(2)), np.std(report[1].round(2))))
            print("f1-score: {},{}\n".format(np.mean(report[2].round(2)), np.std(report[2].round(2))))

            precision = np.append(precision, np.mean(report[0].round(2)))
            recall = np.append(recall, np.mean(report[1].round(2)))
            f1_score = np.append(f1_score, np.mean(report[2].round(2)))

            print("Confusion Matrix:")
            print(confusion_matrix(labels.detach().cpu(),predicted.detach().cpu()))

        # print results averaged across folds
        print("test accuracy (mean, std): {}, {}\n".format(np.mean(test_acc), np.std(test_acc)))
        print("precision (mean, std): {}, {}\n".format(100*np.mean(precision), 100*np.std(precision)))
        print("recall (mean, std): {}, {}\n".format(100*np.mean(recall), 100*np.std(recall)))
        print("f1_score (mean, std): {}, {}\n".format(100*np.mean(f1_score), 100*np.std(f1_score)))

        write_excel_file(test_acc, precision, recall, f1_score, excel_file, model_ss, test_ss)
