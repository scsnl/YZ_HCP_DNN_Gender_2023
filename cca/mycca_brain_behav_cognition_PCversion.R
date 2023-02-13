setwd("/Users/zhangyuan/Google Drive/2021_HCP_Gender_DNN")
rm(list=ls())
source("scripts/brainbeh/2022CCA/myRFunc_updated.R")

#library(vegan) # for cca
library(CCP) # for cca statistical test
library(reshape2)
library(ggplot2)
library(psych) # for pca

################################
########## LOAD DATA ###########
################################
# load behavior data file
beh = read.csv("data/beh_pc_cognition.csv",header=T)

# load pid for brain session 
pid_s1 = read.csv("data/pid_RL_S1.csv",header=F)

# load feature attribution weights (best fold) 
features_s1 = read.csv("data/fingerprints_Subj_by_ROIs_HCP_model_RL_S1_index_2_test_RL_S1.csv",header=F)
colnames(features_s1) = bn_atlas$Description

# ################################
# ######## DATA CLEANING #########
# ################################
beh_sel = beh[complete.cases(beh),]
# find subjects having both behavioral and neural data
pid = intersect(beh_sel$pid, pid_s1$V1)
# keep data for subject having both behavioral and brain data
beh_sel = beh_sel[which(beh_sel$pid %in% pid),]
features = features_s1[which(pid_s1$V1 %in% pid),]
# get idx for female and male
f_idx = which(beh_sel$gender == "F")
m_idx = which(beh_sel$gender == "M")

# heatmap(as.matrix(features[f_idx,]),Colv = NA, Rowv = NA, col = rainbow(256))
# heatmap(as.matrix(beh_sel[f_idx,-c(1:2)]),Colv = NA, Rowv = NA, scale="column")

################################
############# CCA ##############
################################
# set up X and Y
X = data.matrix(features[f_idx,])
Y = data.matrix(beh_sel[f_idx,-c(1:2)])

#
midfix = "HCP_RL_S1_CCA_fsubjs_cognition"
fname = paste("results/brain_behav/hcp_cca_246roi_3pcs_202209/",midfix,"_perm5000_PCversion.RData",sep="")

x_mtx <- data.matrix(X, rownames.force = NA)
y_mtx <- data.matrix(Y, rownames.force = NA)

# cca permutation test (shuffle rows and rerun cca for nperm times)
start_time <- Sys.time()
xy.cca <- mycca(x_mtx,y_mtx,5000)
end_time <- Sys.time()
end_time - start_time

# CCA model significance
xy.cca$Pillai
xy.cca$pillai.p
xy.cca$pillai.p.perm

# canonical correlations
xy.cca$cancor
xy.cca$cancor^2
# canonical correlations significance
xy.cca$cancor.p.perm
xy.cca$cancor.p.perm.adjust

# dimension test
p.asym(xy.cca$cancor, nrow(x_mtx), ncol(x_mtx), ncol(y_mtx), tstat = "Wilks")

save(x_mtx, y_mtx, xy.cca, file = fname)
