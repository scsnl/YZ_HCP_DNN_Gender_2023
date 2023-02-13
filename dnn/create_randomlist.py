import random
import numpy as np

output_folder = 'PROJECT_DIR/data/imaging/roi/\
multiple_cv_datasets/'
fname = output_folder + 'randomSeedsList.npz'

randomlist1 = [] # this is for initial train-test split
randomlist2 = [] # this is for further split of train into train-valid
for i in range(100):
    randomlist1.append(random.randint(1,10000))
    randomlist2.append(random.randint(1,10000))
print(randomlist1)
print(randomlist2)
np.savez(fname, randomlist1 = randomlist1, randomlist2 = randomlist2)
