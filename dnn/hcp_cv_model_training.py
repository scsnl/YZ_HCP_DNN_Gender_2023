import torch
from torch.utils.data import DataLoader,TensorDataset
from modelClasses import *
from utilityFunctions import *
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_fscore_support
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


models_sessions_list = ['LR_S1', 'LR_S2', 'RL_S1', 'RL_S2']
hcp_sessions_list = ['S1','S2','S3','S4']

output_path = 'PROJECT_DIR/results/restfmri/dnn/classification_excel_files/'
if not os.path.exists(output_path):
    os.makedirs(output_path)
excel_file = output_path + 'classification_normz_within_session.xlsx'
# best_split_file = output_path + 'best_performance_split.npz'

# Hyperparameters
num_epochs = 15
num_classes = 2
batch_size = 32
learning_rate = 0.0001

USE_PRETRAINED_MODEL = True
use_cuda = True

best_split_id = np.empty((0,1), int) # save the ID of best split for each session

for ii in range(4): # for each HCP session
    session = models_sessions_list[ii] # specify HCP session used for training and testing
    path_to_dataset = 'PROJECT_DIR/data/imaging/roi/cv_datasets/cv_dataset_' + session + '/'
    path_to_output = 'PROJECT_DIR/data/imaging/roi/outputs/saved_models/models_cv_' + session + '/'
    if not os.path.exists(path_to_output):
        os.makedirs(path_to_output)

    model_ss = hcp_sessions_list[ii] # specify HCP session which models were trained on (S1 to S4)
    test_ss = hcp_sessions_list[ii] # specify HCP session which models were tested on (S1 to S4)

    test_acc = np.empty((0,1),float)
    precision = np.empty((0,1),float)
    recall = np.empty((0,1),float)
    f1_score = np.empty((0,1),float)

    for m in range(5):

        fname_dataset = path_to_dataset + 'hcp_cross_validation_dataset_' + session + '_normz_index_%s.npz'%str(m)
        fname_model = path_to_output + 'model_hcp_brainnetome_mean_' + session + '_normz_window_CV_train_valid_test_window_index_%s.pt' % str(m)
        print("file names: \n dataset {}\n output model {}\n".format(fname_dataset, fname_model))
        datao = np.load(fname_dataset)

        x_train = datao['x_train']
        y_train = datao['y_train'].astype('int64')
        x_train = reshapeData(x_train)

        # Process Validation data
        x_valid = datao['x_valid']
        y_valid = datao['y_valid'].astype('int64')
        x_valid = reshapeData(x_valid)

        # Process Test data
        x_test = datao['x_test']
        y_test = datao['y_test'].astype('int64')
        x_test = reshapeData(x_test)

        # Train Data
        batch_size = 32
        input_tensor = torch.from_numpy(x_train).type(torch.FloatTensor)
        label_tensor = torch.from_numpy(y_train)
        dataset_train1 = TensorDataset(input_tensor, label_tensor)

        # Validation Data
        input_tensor_valid = torch.from_numpy(x_valid).type(torch.FloatTensor)
        label_tensor_valid = torch.from_numpy(y_valid)
        dataset_valid1 = TensorDataset(input_tensor_valid, label_tensor_valid)

        # Test Data
        input_tensor_test = torch.from_numpy(x_test).type(torch.FloatTensor)
        label_tensor_test = torch.from_numpy(y_test)
        dataset_test1 = TensorDataset( input_tensor_test, label_tensor_test )

        # Load Train and Test data into the loader
        train_loader = DataLoader(dataset=dataset_train1, batch_size=batch_size, shuffle=True)
        valid_loader = DataLoader(dataset=dataset_valid1, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(dataset=dataset_test1, batch_size=x_test.shape[0], shuffle=False)

        # define model
        model = ConvNet()
        # USE_PRETRAINED_MODEL = True
        # use_cuda = True

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
        print(classification_report(labels.detach().cpu(),predicted.detach().cpu()))

        report = precision_recall_fscore_support(labels.detach().cpu(),predicted.detach().cpu())
        print("precision: {},{}\n".format(np.mean(report[0].round(2)),np.std(report[0].round(2))))
        print("recall: {},{}\n".format(np.mean(report[1].round(2)), np.std(report[1].round(2))))
        print("f1-score: {},{}\n".format(np.mean(report[2].round(2)), np.std(report[2].round(2))))

        precision = np.append(precision, np.mean(report[0].round(2)))
        recall = np.append(recall, np.mean(report[1].round(2)))
        f1_score = np.append(f1_score, np.mean(report[2].round(2)))

        print("Confusion Matrix:")
        print (confusion_matrix(labels.detach().cpu(),predicted.detach().cpu()))

    print("test accuracy (mean, std): {}, {}\n".format(np.mean(test_acc), np.std(test_acc)))
    print("precision (mean, std): {}, {}\n".format(100*np.mean(precision), 100*np.std(precision)))
    print("recall (mean, std): {}, {}\n".format(100*np.mean(recall), 100*np.std(recall)))
    print("f1_score (mean, std): {}, {}\n".format(100*np.mean(f1_score), 100*np.std(f1_score)))

    best_split_id = np.append(best_split_id, np.argmax(test_acc))

    if USE_PRETRAINED_MODEL: # write to excel only if model has been well trained
        write_excel_file(test_acc, precision, recall, f1_score, excel_file, model_ss, test_ss)

print(hcp_sessions_list)
print(best_split_id)
# np.savez(best_split_file, sessions=hcp_sessions_list, best_split_id = best_split_id)
