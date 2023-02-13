#!/bin/bash

# cd PROJECT_DIR/results/restfmri/dnn/niis/group_niis_for_figures
cd PROJECT_DIR/results/restfmri/dnn/niis/consensus_multiple_cv
# cd PROJECT_DIR/results/restfmri/dnn/niis_DL/consensus_multiple_cv

pnt="80"

## declare an array variable
# declare -a arr=("REST1_LR" "REST2_LR" "REST1_RL" "REST2_RL"
#                "LR_S1_test_nki_645" "RL_S1_test_nki_645")

declare -a arr=("LR_S1_test_leipzig_AP_run-01" "RL_S1_test_leipzig_AP_run-01")

## now loop through the above array
for i in "${arr[@]}"
do
  fnii=`ls *_female_*${i}*_percentile_${pnt}.nii.gz`
  mnii=`ls *_male_*${i}*_percentile_${pnt}.nii.gz`
  mask="mask_${i}_${pnt}.nii.gz"
  outnii="fm_intersect_${i}_${pnt}.nii.gz"

  echo ${fnii}
  echo ${mnii}
  echo ${mask}
  echo ${outnii}

  fslmaths ${fnii} -bin ./tmp/${fnii/.nii.gz/_bin.nii.gz}
  fslmaths ${mnii} -bin ./tmp/${mnii/.nii.gz/_bin.nii.gz}
  fslmaths ./tmp/${fnii/.nii.gz/_bin.nii.gz} -add ./tmp/${mnii/.nii.gz/_bin.nii.gz} -thr 2 -bin ./tmp/${mask}
  fslmaths ${fnii} -add ${mnii} -mul 0.5 -mul ./tmp/${mask} ./fm_intersect_nii/${outnii}

done

