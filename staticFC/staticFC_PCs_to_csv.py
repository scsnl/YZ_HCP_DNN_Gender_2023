import numpy as np
from sklearn.decomposition import PCA

if __name__ == '__main__':

    data_path = 'PROJECT_DIR/results/restfmri/dnn/sFC/'
    fname = data_path + 'sFC_HCP_Session1.npz'

    data = np.load(fname)
    data_features = data['features']
    labels = data['labels']

    pca = PCA()
    feature_pcs = pca.fit_transform(data_features)
    print(feature_pcs.shape)
    feature_pcs_246 = feature_pcs[:,0:246]
    print(feature_pcs_246.shape, type(feature_pcs_246))

    # explained 80.4% variance in HCP Session1 sFC
    # explained 79.7% variance in HCP Session3 sFC
    print(np.sum(pca.explained_variance_ratio_[0:246]))

    fname = data_path + "sFC_PCs_HCP_Session1.csv"
    np.savetxt(fname, feature_pcs_246, delimiter=",")
    fname = data_path + "sFC_labels_HCP_Session1.csv"
    np.savetxt(fname, labels, delimiter=",")
