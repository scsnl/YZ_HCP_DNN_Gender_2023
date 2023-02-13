import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
import torch
from torch.utils.data import DataLoader,TensorDataset
from modelClasses import *
from utilityFunctions import *
from sklearn.metrics import confusion_matrix,classification_report,precision_recall_fscore_support
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# REST1_LR, LR_S1, S1 all refer to HCP Session 1
# REST1_RL, RL_S1, S3 all refer to HCP Session 3

ss_name = 'REST1_RL' # this is the session name used to get the original time-series file
session = 'RL_S1' # this is the session name used to locate/save the trained models
model_ss = 'S3' # this is the session used for model training
test_ss = 'S3' # this is the session used for testing

path_to_dataset = 'PUBLIC_DATA_DIR/hcp/restfmri/timeseries/group_level/brainnetome/\
normz/hcp_run-rfMRI_' + ss_name + '_brainnetome_mean_regMov-6param_wmcsf_dt1_bpf008-09_normz_246ROIs.pklz'
output_folder = 'PROJECT_DIR/data/imaging/roi/\
outputs/saved_models_multiple_cv/models_cv_' + session + '/'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

seeds = np.load('PROJECT_DIR/data/imaging/roi/\
multiple_cv_datasets/randomSeedsList.npz')
rlist1 = seeds['randomlist1'] # this is used in initial train-test split
rlist2 = seeds['randomlist2'] # this is used in further split of train into train-valid
# print(rlist1)
# print(rlist2)

# # classification results
# excel_path = 'PROJECT_DIR/results/restfmri/dnn/classification_excel_files/'
# if not os.path.exists(excel_path):
#     os.makedirs(excel_path)
# excel_file = excel_path + 'across_all_5folds_classification_normz_within_session.xlsx'

# load & clean data
datao = pd.read_pickle(path_to_dataset)
datao.drop(datao[datao['percentofvolsrepaired']>10].index, inplace=True)
datao.drop(datao[datao['mean_fd']>0.5].index, inplace=True)
datao = datao.reset_index()
data = np.asarray([np.asarray(lst)[:, :] for lst in datao.data])
labels_gender = datao['gender']
subjid = datao['subject_id']

labels_full = []
for i in labels_gender:
  if i == 'male':
    labels_full.append(0)
  else:
    labels_full.append(1)
labels_full = np.asarray(labels_full)

del datao
print('Data loading completed')


# Hyperparameters
num_epochs = 15
num_classes = 2
batch_size = 32
learning_rate = 0.0001

USE_PRETRAINED_MODEL = False
use_cuda = True

num_5folds = len(rlist1) # 100 x 5folds

avg_test_acc = np.empty((0, 1), float)
avg_precision = np.empty((0, 1), float)
avg_recall = np.empty((0, 1), float)
avg_f1_score = np.empty((0, 1), float)

for ii in range(num_5folds):
    print("the {} 5folds".format(ii))
    # split data
    kf = StratifiedKFold(n_splits=5,random_state=rlist1[ii],shuffle=True)

    train_index_list = []
    test_index_list = []
    for train_index, test_index in kf.split(subjid, labels_full):
        print("TRAIN:", len(train_index), "TEST:", len(test_index))
        train_index_list.append(train_index)
        test_index_list.append(test_index)

    test_acc = np.empty((0, 1), float)
    precision = np.empty((0, 1), float)
    recall = np.empty((0, 1), float)
    f1_score = np.empty((0, 1), float)

    for m in range(5):
        fname_model = output_folder + 'ff_' + str(ii) + '_model_hcp_brainnetome_mean_' + session + '_normz_window_CV_train_valid_test_window_index_%s.pt' % str(m)
        print(fname_model)

        # Split Data into Train, Test and Validation Sets
        data_split = data[train_index_list[m]]
        labels_split = labels_full[train_index_list[m]]
        print(data_split.shape)
        print(labels_split.shape)

        x_train_org, x_valid_org, y_train_org, y_valid_org = train_test_split(data_split, labels_split,
                              test_size = 0.1, random_state = rlist2[ii], stratify = labels_split)

        # Get windowed train and validate data
        window_size = 256
        step = 64
        x_train, y_train = prepare_data_sliding_window(x_train_org,y_train_org,window_size,step)
        x_valid, y_valid = prepare_data_sliding_window(x_valid_org,y_valid_org,window_size,step)

        # do not "window" test data
        x_test = data[test_index_list[m]]
        y_test = labels_full[test_index_list[m]]

        # prepare data
        x_train = reshapeData(x_train)
        # y_train = y_train.astype('int64')
        x_valid = reshapeData(x_valid)
        # y_valid = y_valid.astype('int64')
        x_test = reshapeData(x_test)
        # y_test = y_test.astype('int64')

        # Train Data
        batch_size = 32
        input_tensor = torch.from_numpy(x_train).type(torch.FloatTensor)
        label_tensor = torch.from_numpy(np.array(y_train))
        dataset_train1 = TensorDataset(input_tensor, label_tensor)

        # Validation Data
        input_tensor_valid = torch.from_numpy(x_valid).type(torch.FloatTensor)
        label_tensor_valid = torch.from_numpy(np.array(y_valid))
        dataset_valid1 = TensorDataset(input_tensor_valid, label_tensor_valid)

        # Test Data
        input_tensor_test = torch.from_numpy(x_test).type(torch.FloatTensor)
        label_tensor_test = torch.from_numpy(np.array(y_test))
        dataset_test1 = TensorDataset(input_tensor_test, label_tensor_test)

        # Load Train and Test data into the loader
        train_loader = DataLoader(dataset=dataset_train1, batch_size=batch_size, shuffle=True)
        valid_loader = DataLoader(dataset=dataset_valid1, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(dataset=dataset_test1, batch_size=x_test.shape[0], shuffle=False)

        # define model
        model = ConvNet()

        if use_cuda and torch.cuda.is_available():
            model.cuda()

        if USE_PRETRAINED_MODEL:
            print("Using the existing trained model")
            model.load_state_dict(torch.load(fname_model))
        else:
            print("Training the model")
            # Loss and optimizer
            criterion = nn.CrossEntropyLoss()
            # criterion = FocalLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
            # Train the model
            total_step = len(train_loader)
            loss_list = []
            acc_list = []
            val_acc_temp = 0.0
            for epoch in range(num_epochs):
                model.train()
                for i, (data_ts, labels) in enumerate(train_loader):
                    data_ts = data_ts.cuda()
                    labels = labels.cuda()
                    # Run the forward pas
                    # labels = labels.long()
                    outputs = model(data_ts)
                    loss = criterion(outputs, labels)
                    loss_list.append(loss.item())

                    # Backprop and perform Adam optimisation
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    # Track the accuracy
                    total = labels.size(0)
                    _, predicted = torch.max(outputs.data, 1)
                    correct = (predicted == labels).sum().item()
                    acc_list.append(correct / total)

                    if (i + 1) % 10 == 0:
                        print('Epoch [{}/{}], Step [{}/{}], Train Loss: {:.4f}, Accuracy: {:.2f}%'
                              .format(epoch + 1, num_epochs, i + 1, total_step, loss.item(),
                                      (correct / total) * 100))
                model.eval()
                with torch.no_grad():
                    correct = 0
                    total = 0
                    for images, labels in valid_loader:
                        if use_cuda and torch.cuda.is_available():
                            images = images.cuda()
                            labels = labels.cuda()
                        outputs = model(images)
                        _, predicted = torch.max(outputs.data, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum().item()
                        val_acc = (correct / total)
                    print('Validation Accuracy of the model on the Val data: {} %'.format((correct / total) * 100))
                if val_acc_temp < val_acc:
                    val_acc_temp = val_acc
                    print('**Saving Model on Drive**')
                    torch.save(model.state_dict(), fname_model)

        # evaluate
        model.load_state_dict(torch.load(fname_model))
        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for images, labels in test_loader:
                images = images.cuda()
                labels = labels.cuda()
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            print('Test Accuracy of the model on the  test data: {} %'.format((correct / total) * 100))
            test_acc = np.append(test_acc, 100 * correct / total)

        # print fold m results
        # print(classification_report(labels.detach().cpu(), predicted.detach().cpu()))

        report = precision_recall_fscore_support(labels.detach().cpu(), predicted.detach().cpu())
        # print("precision: {},{}\n".format(np.mean(report[0].round(2)), np.std(report[0].round(2))))
        # print("recall: {},{}\n".format(np.mean(report[1].round(2)), np.std(report[1].round(2))))
        # print("f1-score: {},{}\n".format(np.mean(report[2].round(2)), np.std(report[2].round(2))))

        precision = np.append(precision, np.mean(report[0].round(2)))
        recall = np.append(recall, np.mean(report[1].round(2)))
        f1_score = np.append(f1_score, np.mean(report[2].round(2)))

        # print("Confusion Matrix:")
        # print(confusion_matrix(labels.detach().cpu(), predicted.detach().cpu()))


    print("ff {} test accuracy (mean, std): {}, {}\n".format(ii, np.mean(test_acc), np.std(test_acc)))
    print("ff {} precision (mean, std): {}, {}\n".format(ii, 100 * np.mean(precision), 100 * np.std(precision)))
    print("ff {} recall (mean, std): {}, {}\n".format(ii, 100 * np.mean(recall), 100 * np.std(recall)))
    print("ff {} f1_score (mean, std): {}, {}\n".format(ii, 100 * np.mean(f1_score), 100 * np.std(f1_score)))

    avg_test_acc = np.append(avg_test_acc, np.mean(test_acc))
    avg_precision = np.append(avg_precision, np.mean(precision))
    avg_recall = np.append(avg_recall, np.mean(recall))
    avg_f1_score = np.append(avg_f1_score, np.mean(f1_score))

print("across all 5foldes")
print("test accuracy (mean, std): {}, {}\n".format(np.mean(avg_test_acc), np.std(avg_test_acc)))
print("precision (mean, std): {}, {}\n".format(100 * np.mean(avg_precision), 100 * np.std(avg_precision)))
print("recall (mean, std): {}, {}\n".format(100 * np.mean(avg_recall), 100 * np.std(avg_recall)))
print("f1_score (mean, std): {}, {}\n".format(100 * np.mean(avg_f1_score), 100 * np.std(avg_f1_score)))

# if USE_PRETRAINED_MODEL: # only write to excel if model has been well trained
#     write_excel_file(test_acc, precision, recall, f1_score, excel_file, model_ss, test_ss)




