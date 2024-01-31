##**This work is license under [MIT License](https://choosealicense.com/licenses/mit/)** 

### Cite this work
#### Ryali, S., Zhang, Y., de los Angeles, C., Superkar, K., & Menon, V. (In Press). Deep learning models reveal replicable, generalizable, and behaviorally relevant sex differences in human functional brain organization. <em>The proceedings of the National Academy of Sciences. </em>
--


## Key steps and corresponding scripts used for the analyses are summarized below
### Time series extraction
 
 The extracted time series were used as inputs of DNN (scripts under timeseries_extraction/)

 1. config files: rest\_ts\_extraction\_config\_*_brainnetome.m 
 2. main script: read\_ts\_extraction\_dev.m

 
### DNN classification (scripts under dnn/)

1. utility functions: utilityFunctions.py 

2. model architecture: modelClasses.py

3. create cross-validation datasets on HCP and extract basic info (e.g. #F/#M, subject ids, head motions): create\_cv\_datasets.py and extract\_basic\_info.py

4. train models on HCP datasets: hcp\_cv\_model\_training.py
    
5. test replicability on HCP datasets: evaluate\_hcp\_session.py

6. test generalizability on NKI-RS dataset: evaluate\_nkirs.py
    
7. test generalizability on MPI Leipzig dataset: evaluate\_leipzig.py

   
   
    
### Consensus analysis (scripts under dnn/)

In consensus analysis, we split data into 5 folds for 100 times, repeated analysis for each CV dataset (training on HCP Session 1 or 3 and then extracting individual fingerprints using the trained models), and finally gathered all fingerprints from the 100 runs and identified consensus features (top features that consistently appeared across all folds)

1. create random seeds for each of the 100 split: create\_randomlist.py
    
2. train models on HCP session 1 or 3 for each of CV dataset: hcp\_training\_models\_multiple\_cv\_datasets.py
    

3. generate feature attributions from each of the trained models using IG or DeepLift
    - for IG: generate\_feature\_attribution\_ig\_multiple\_cv.py
    - for DeepLift: generate\_DL\_feature\_attribution\_multiple\_cv.py

4. get consensus features across all models for each session: get\_consensus\_features\_multiple\_cv.py (IG features of HCP cohort) and get\_DL\_consensus\_features\_multiple\_cv.py (DL features of HCP cohort)

    
5. generate common features across sexes: generate\_common\_features\_across\_sex.sh
    
6. generate brain plots of consensus features: generate\_consensus\_images.m    

7. generate NKI-RS feature attributions from each of the trained models using IG or DeepLift
    - for IG: generate\_nki\_feature\_attribution\_ig\_multiple\_cv.py
    - for DeepLift: generate\_nki\_DL\_feature\_attribution\_multiple\_cv.py

8. get NKI-RS consensus features across all HCP models of each session: get\_nki\_consensus\_features\_multiple\_cv.py (IG features of NKI-RS cohort) and get\_nki\_DL\_consensus\_features\_multiple\_cv.py (DL features of NKI-RS cohort)
      
9. generate NKI-RS common features across sexes: generate\_common\_features\_across\_sex.sh
    
10. generate brain plots of NKI-RS consensus features: generate\_consensus\_images.m
    
11. generate MPI Leipzig feature attributions from each of the trained models using IG: generate\_leipzig\_feature\_attribution\_multiple\_cv.py
    
12. get MPI Leipzig consensus features across all HCP models of each session: get\_leipzig\_consensus\_features\_multiple\_cv.py 

13. generate MPI Leipzig common features across sexes: generate\_common\_features\_across\_sex.sh

14. generate brain plots of Leipzig consensus features: generate\_consensus\_images.m


  
### Individual fingerprints (scripts under dnn/)

1. generate nifti files of top features for each individual: generate\_indiv\_niftis.py for HCP cohort and generate\_nki\_indiv\_niftis.py for NKI-RS cohort
    
2. generate brain images of top features for each individual: generate\_indiv\_images.m
    
3. radar plots of individual fingerprints (only show top 20% features identified at the group level with the same sex)
	- generate\_data\_radarplots.py and generate\_indiv\_radarplots.py for HCP cohort
	- generate\_data\_radarplots\_nki.py and generate\_indiv\_radarplots\_nki.py for NKI-RS cohort

4. distinctiveness of fingerprints: distinctiveness\_fingerprint.py for HCP cohort and distinctiveness\_nki\_fingerprint.py for NKI cohort
    
5. tsne plot of fingerprints' distinctiveness: tsne\_plot\_distinctiveness.py for HCP cohort and tsne\_plot\_distinctiveness\_nki.py for NKI cohort

6. bar plot of fingerprints' distinctiveness: generate\_data\_bar\_plot\_distinctiveness.py and distinctiveness\_barplots.R

7. stability of fingerprints: stability\_fingerprint.py
    
  
### Control analyses 

1. control analysis examining the relationship between head movement and brain features underlying sex classification: dnn/control\_analysis\_dcor2.py

2. control analyses with different artifact reduction methods and different brain atlases
    - config files: timeseries\_extraction/rest\_ts\_extraction\_config\_hcp\_rfMRI\_*_acompcor.m 
    - main script: timeseries\_extraction/read\_ts\_extraction\_dev.m
    - same DNN classification scripts were applied to time series extracted using the scripts here

    
  
### Brain-behavioral analysis (scripts under cca/)
1. brain-behavioral analysis using individual-level stDNN features attributions (fingerprints) and 3 PCs derived from NIH toolbox congnition measures
    - mycca\_brain\_behav\_cognition\_PCversion.R is used for brain-behavioral CCA
    - mycca\_brain\_behav\_cognition\_prediction\_MvsF\_HCP.R is used for prediction analysis
    - mycca\_plotting\_brain\_behav\_cognition\_PCversion.R is for plotting
    - myRFunc\_updated.R includes all supporting functions



### Conventional ML approaches (scripts under staticFC/)
1. classification using conventrional machine learning approaches
    - classicML\_staticFC\_gender\_multiple.py for HCP and NKI-RS cohorts 
    - classicML\_staticFC\_gender\_multiple\_leipzig.py for HCP and MPI Leipzig cohorts

2. brain-behavioral analysis using static functional connectivity and cognition measures
    - staticFC\_PCs\_to\_csv.py is used to get the first 246 PCs of static functional connectivity and save to csv file
    - mycca\_sFC\_behav\_cognition\_PCversion.R is used for brain-behavioral CCA with sFC PCs
    - mycca\_sFC\_behav\_cognition\_prediction\_MvsF\_HCP.R is used for prediction analysis
    - mycca\_plotting\_sFC\_behav\_cognition\_PCversion.R is for plotting
    - myRFunc\_updated.R includes all supporting functions
    
    
   
