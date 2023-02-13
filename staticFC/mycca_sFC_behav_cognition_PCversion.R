setwd("/Users/zhangyuan/Google Drive/2021_HCP_Gender_DNN")
rm(list=ls())
source("scripts/brainbeh/2022CCA/myRFunc_updated.R")

library(reshape2)
library(ggplot2)
library(psych) # for pca

################################
########## LOAD DATA ###########
################################
# load behavior data file
beh = read.csv("data/beh_pc_cognition.csv",header=T)

# load pid for brain session 
pid = read.csv("data/pid_RL_S1.csv",header=F)

# load features (246 PCs from sFC) 
features = read.csv("sFC/sFC_PCs_HCP_Session3.csv",header=F)

# ################################
# ######## DATA CLEANING #########
# ################################
beh_sel = beh[complete.cases(beh),]
# find subjects having both behavioral and neural data
pid_comm = intersect(beh_sel$pid, pid$V1)
# keep data for subject having both behavioral and brain data
beh_sel = beh_sel[which(beh_sel$pid %in% pid_comm),]
features = features[which(pid$V1 %in% pid_comm),]
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
midfix = "HCP_RL_S1_CCA_fsubjs_sFC_cognition"
fname = paste("results/brain_behav/sFC_PCversion/",midfix,"_perm5000_PCversion.RData",sep="")

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

# 
# ###########################################
# ### Plot to better understand the modes ###
# ###########################################
# midfix = "HCP_LR_S1_CCA_fsubjs_sFC_cognition"
# fname = paste("sFC/results/",midfix,"_perm1000.RData",sep="")
# load(fname)
# 
# ## Correlation between behavioral variables (Y) and modes (i.e. loadings) ##
# Y.Cy.ld = melt(xy.cca$corr.Y.Cy)
# df = Y.Cy.ld 
# yy = max(abs(min(df$value)),abs(max(df$value)))
# 
# # heatmap color & value overlay
# ggplot() +
#   geom_tile(df, mapping = aes(Var2, Var1, fill = value)) +
#   scale_fill_gradient2(low = "blue", high = "red", mid = "white",
#                        midpoint = 0, limit = c(-1*yy,yy)) +
#   geom_text(df,mapping = aes(Var2, Var1,
#                              label = round(value, digit = 3)),color="black") +
#   labs(x="Modes", y="Behaviors") +
#   theme_classic() + theme(axis.text.x=element_text(size=9, angle=0, vjust=0.3),
#                           axis.text.y=element_text(size=9, angle=0, vjust=0.3),
#                           plot.title=element_text(size=11))
# # plt.name = paste(output_path,'/CCA_PCA', '_',condition, '_',postfix,'_YCy_heatmap.eps',sep="")
# # ggsave(plt.name,units="in", width=4, height=5)
# 
# ## Standardized weights for behavioral variables (Y) ## 
# rownames(xy.cca$ycoef.std) = rownames(xy.cca$corr.Y.Cy)
# ycoef.std = melt(xy.cca$ycoef.std)
# df = ycoef.std 
# yy = max(abs(min(df$value)),abs(max(df$value)))
# 
# # heatmap color & value overlay
# ggplot() +
#   geom_tile(df, mapping = aes(Var2, Var1, fill = value)) +
#   scale_fill_gradient2(low = "blue", high = "red", mid = "white",
#                        midpoint = 0, limit = c(-1*yy,yy)) +
#   geom_text(df,mapping = aes(Var2, Var1,
#                              label = round(value, digit = 3)),color="black") +
#   #labs(x="", y="PCs") +
#   theme_classic() + theme(axis.text.x=element_text(size=9, angle=0, vjust=0.3),
#                           axis.text.y=element_text(size=9, angle=0, vjust=0.3),
#                           plot.title=element_text(size=11))
# # plt.name = paste(output_path,'/CCA_PCA', '_',condition, '_',postfix,'_Ystdcoef_heatmap.eps',sep="")
# # ggsave(plt.name,units="in", width=4, height=5)

# ## plot scatter plot (canonical scores) ##
# # find the behavioral measure with the largest loading
# # df = Y.Cy.ld
# # type = "colorByLoading"
# 
# df = ycoef.std 
# type = "colorBystdcoef"
# 
# for(i in 1:dim(res$Cy)[2]){
#   col_id = which(df$Var2 == paste("CanAxis",i, sep=""))
#   idx = which(abs(df$value[col_id]) == max(abs(df$value[col_id])))
#   
#   print(cols[idx])
#   
#   Cy = res$Cy[,i] # behavior
#   Cx = res$Cx[,i] # brain
#   df_plot = data.frame(Cx = Cx, Cy=Cy, Cog=Y[,cols[idx]])
#   
#   ggplot(df_plot, aes(x=Cx, y=Cy, col=Cog)) +
#     geom_point() + 
#     scale_colour_gradientn(colours = c("blue", "cyan", "yellow", "red"),
#                            name = cols[idx]) +
#     labs(x="Mode 1 (Brain)", y="Mode 1 (Behavior)") +
#     theme_bw() + theme(axis.text.x=element_text(size=12, angle=0, vjust=0.3),
#                        axis.text.y=element_text(size=12, angle=0, vjust=0.3),
#                        plot.title=element_text(size=12))
#   plt.name = paste(output_path,'/CCA_PCA', '_',condition, '_',postfix,'_CxCy_scatter_mode', i, '_', type, '.eps',sep="")
#   ggsave(plt.name,units="in", width=5, height=4)
# }
# 
