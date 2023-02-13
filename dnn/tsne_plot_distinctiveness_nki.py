import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE


if __name__=="__main__":
    best_split_id = [2, 2]
    hcp_sessions_list = ['LR_S1', 'RL_S1']
    test_session_list = ['645', '645']

    data_path = '/Users/zhangyuan/Desktop/Sherlock/projects/menon/2021_HCP_Gender/results/restfmri/dnn/tsne_plots/'
    fig_name = '/Users/zhangyuan/Google Drive/2021_HCP_Gender_DNN/results/tsne_plots/distinvtiveness_nki.tiff'
    fig_name2 = '/Users/zhangyuan/Google Drive/2021_HCP_Gender_DNN/results/tsne_plots/distinvtiveness_nki.eps'

    # fig setup
    fig = plt.figure(figsize=(10, 10), dpi=300)

    for ss in range(2):

        model_session = hcp_sessions_list[ss]
        m = best_split_id[ss]
        test_session = test_session_list[ss]
        data_file = data_path + 'fingerprints_Subj_by_ROIs_HCP_model_' + model_session + '_index_' + str(m) + '_test_nki_' + test_session + '.npz'
        print(data_file)

        data = np.load(data_file, allow_pickle=True)
        title = 'Session' + str(ss+1)

        features = data['features']
        labels = data['labels']
        print(len(labels))

        # compute tsne
        # perplexity: recommended 5-50 & should be smaller than #data points
        tsne = TSNE(n_components=2, verbose=0, perplexity=50, n_iter=500,
                    learning_rate=200, random_state=100)
        tsne_results = tsne.fit_transform(features)
        # visualize
        palette = sns.color_palette("hls", 2) # sns.color_palette("bright", 2)
        ax = fig.add_subplot(int('22'+str(ss+1)))
        sns.scatterplot(tsne_results[:,0], tsne_results[:,1], hue=labels, legend='full',
                        palette=palette, s=25, alpha=0.7)
        ax.set(xlabel='tSNE-1', ylabel='tSNE-2', title=title)
        # plt.show()
        #
        # df = pd.DataFrame(tsne_results)
        # mypal = sns.color_palette("hls", 2)
        # sns.scatterplot(data=df, x=0, y=1, hue=labels, palette=mypal, alpha=0.9,
        #                 s=20, linewidth=0, edgecolor='k', legend='full')
        # plt.gca().set(xlabel='tSNE-1', ylabel='tSNE-2', title=title)
        # plt.show()

        fig.savefig(fig_name, dpi=300, format="tiff", pil_kwargs={"compression": "tiff_lzw"}) # bbox_inches='tight'
        fig.savefig(fig_name2, dpi=300, format="eps")