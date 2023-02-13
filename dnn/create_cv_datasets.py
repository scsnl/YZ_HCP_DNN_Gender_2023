import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from utilityFunctions import *

# Session 1 = REST1_LR; Session 2 = REST2_LR
# Session 3 = REST1_RL; Session 4 = REST2_RL
ss_name = 'REST1_RL'
session = 'RL_S1'

path_to_dataset = 'PUBLIC_DATA_DIR/hcp/restfmri/timeseries/group_level/brainnetome/\
normz/hcp_run-rfMRI_' + ss_name + '_brainnetome_mean_regMov-6param_wmcsf_dt1_bpf008-09_normz_246ROIs.pklz'
output_folder = 'PROJECT_DIR/data/imaging/roi/\
cv_datasets/cv_dataset_' + session + '/'
output_f_trainlist_index = output_folder + 'train_list_' + session + '_index'
output_f_testlist_index = output_folder + 'test_list_' + session + '_index'

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# load & clean data
datao = pd.read_pickle(path_to_dataset)
# print(datao.shape)
datao.drop(datao[datao['percentofvolsrepaired']>10].index, inplace=True)
# print(datao.shape)
datao.drop(datao[datao['mean_fd']>0.5].index, inplace=True)
# print(datao.shape)
datao = datao.reset_index()
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
print('Data loading completed')

# split data
kf = StratifiedKFold(n_splits=5,random_state=None,shuffle=False)
train_index_list = []
test_index_list = []
for train_index, test_index in kf.split(subjid,labels):
    train_index_list.append(train_index)
    test_index_list.append(test_index)

# save train/test index
np.save(output_f_trainlist_index,train_index_list)
np.save(output_f_testlist_index,test_index_list)

np.random.seed(3655)
print('****Preparing windowed data****')
for i in range(5):
    '''Split Data into Train, Test and Validation Sets'''
    fname = output_folder + 'hcp_cross_validation_dataset_' + session + '_normz_index_%s'%str(i)
    data_split = data[train_index_list[i]]
    labels_split = labels[train_index_list[i]]

    x_train, x_valid, y_train, y_valid = train_test_split(data_split, labels_split,
                          test_size = 0.1, random_state = 67334, stratify = labels_split)

    '''Get windowed data'''
    window_size = 256
    step = 64
    data_train_window,labels_train_window = prepare_data_sliding_window(x_train,y_train,window_size,step)
    data_valid_window,labels_valid_window = prepare_data_sliding_window(x_valid,y_valid,window_size,step)

    print(data_train_window.shape)
    print(data_valid_window.shape)

    x_test = data[test_index_list[i]]
    y_test = labels[test_index_list[i]]
    # do not "window" test data

    np.savez(fname, x_train = data_train_window, y_train = labels_train_window, x_valid = data_valid_window, y_valid = labels_valid_window, x_test = x_test, y_test = y_test)
    print('**Dataset_%s_Saved**'%str(i))
    del data_train_window, data_valid_window, x_train, x_valid, x_test, y_train, y_valid, y_test