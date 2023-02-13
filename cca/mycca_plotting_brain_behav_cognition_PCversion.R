setwd("/Users/zhangyuan/Google Drive/2021_HCP_Gender_DNN")
rm(list=ls())
source("scripts/brainbeh/2022CCA/myRFunc_updated.R")

library(ggplot2)
# library(ggrepel)
# library(reshape2)

#### load existing results for plotting ####
#*********************
midfix = "HCP_LR_S1_CCA_msubjs_cognition"
#*********************
fname = paste("results/brain_behav/hcp_cca_246roi_3pcs_202209/", midfix, "_perm5000_PCversion.RData", sep="")
load(fname)

# plt.name1 = paste("results/brain_behav/hcp_cca_246roi_3pcs_202209/", midfix, "_YCy_perm5000.eps", sep="")
# plt.name0 = paste("results/brain_behav/hcp_cca_246roi_3pcs_202209/", midfix, "_YCy_heatmap_perm5000.eps", sep="")
# plt.name0 = paste("results/brain_behav/hcp_cca_246roi_3pcs_202209/", midfix, "_stdcoef_heatmap_perm1000.eps", sep="")

plt.name2 = paste("results/brain_behav/hcp_cca_246roi_3pcs_202209/", midfix, "_CxCy_scatter_perm5000.eps", sep="")
plt.name3 = paste("results/brain_behav/hcp_cca_246roi_3pcs_202209/", midfix, "_modeR2_perm5000.eps", sep="")
# plt.name4 = paste("results/brain_behav/hcp_cca_246roi_3pcs_202209/", midfix, "_expvar_behavior_perm5000.eps", sep="")
# plt.name5 = paste("results/brain_behav/hcp_cca_246roi_3pcs_202209/", midfix, "_expvar_brain_perm5000.eps", sep="")

# xls.name1 = paste("results/brain_behav/hcp_cca_246roi_3pcs_202209/", midfix, "_mode1_top10_features_loadings.xlsx", sep="")
# xls.name2 = paste("results/brain_behav/hcp_cca_246roi_3pcs_202209/", midfix, "_mode1_top10_features_stdcoef.xlsx", sep="")

cogVar_names = c("IQ", "behavioral inhibition", "Reward-related self-regulation")

# ###### Correlation between 3PCs and mode 1 (i.e. loadings)#######
# rownames(xy.cca$corr.Y.Cy) = cogVar_names
# Y.Cy.ld = melt(xy.cca$corr.Y.Cy)
# # only plot mode 1
# df = subset(Y.Cy.ld, Var2 %in% c("CanAxis1"))
# yy = max(abs(min(df$value)),abs(max(df$value)))
# 
# ggplot(df, aes(x = Var2, y = value)) +
#   geom_text_repel(aes(label = Var1, size=abs(value), col=value)) +
#   scale_size(range=c(3,5)) + 
#   theme_classic()
# # ggsave(plt.name1,units="in", width=5, height=5)
# 
# # heatmap color & value overlay
# ggplot() +
#   geom_tile(df, mapping = aes(Var2, Var1, fill = value)) +
#   scale_fill_gradient2(low = "blue", high = "red", mid = "white",
#                        midpoint = 0, limit = c(-1,1)) +
#   geom_text(df,mapping = aes(Var2, Var1,
#                              label = round(value, digit = 3)),color="black") +
#   labs(x="Male", y="PCs") +
#   theme_classic() + theme(axis.text.x=element_text(size=9, angle=0, vjust=0.3),
#                      axis.text.y=element_text(size=9, angle=0, vjust=0.3),
#                      plot.title=element_text(size=11))
# ggsave(plt.name0,units="in", width=4, height=5)

# ###### Correlation between 3PCs and mode 1 (i.e. loadings)#######
# rownames(xy.cca$ycoef.std) = cogVar_names
# ycoef.std = melt(xy.cca$ycoef.std)
# # only plot mode 1
# df = subset(ycoef.std, Var2 %in% c("CanAxis1"))
# yy = max(abs(min(df$value)),abs(max(df$value)))
# 
# # heatmap color & value overlay
# ggplot() +
#   geom_tile(df, mapping = aes(Var2, Var1, fill = value)) +
#   scale_fill_gradient2(low = "blue", high = "red", mid = "white",
#                        midpoint = 0, limit = c(-1,1)) +
#   geom_text(df,mapping = aes(Var2, Var1,
#                              label = round(value, digit = 3)),color="black") +
#   labs(x="", y="PCs") +
#   theme_classic() + theme(axis.text.x=element_text(size=9, angle=0, vjust=0.3),
#                           axis.text.y=element_text(size=9, angle=0, vjust=0.3),
#                           plot.title=element_text(size=11))
# ggsave(plt.name0,units="in", width=4, height=5)


##### plot scatter plot (canonical scores of mode 1) ####
# find the cognitive measure with the largest loading
# idx = which(abs(df$value) == max(abs(df$value)))

Cy1 = xy.cca$Cy[,1] # behavior
Cx1 = xy.cca$Cx[,1] # brain
# df_plot = data.frame(Cx1 = Cx1, Cy1=Cy1, Cog=beh_sel[,idx])
df_plot = data.frame(Cx1 = Cx1, Cy1=Cy1)

ggplot(df_plot, aes(x=Cx1, y=Cy1)) +
  geom_point() + 
  geom_smooth(method=lm, se=FALSE, fullrange=TRUE) +
  scale_colour_gradientn(colours = c("blue", "cyan", "yellow", "red")) +
  # scale_colour_gradientn(colours = c("darkred", "orange", "yellow", "white")) +
  labs(x="Mode 1 (Brain)", y="Mode 1 (Behavior)") +
  theme_bw() + theme(axis.text.x=element_text(size=12, angle=0, vjust=0.3),
                     axis.text.y=element_text(size=12, angle=0, vjust=0.3),
                     plot.title=element_text(size=12))
ggsave(plt.name2,units="in", width=5, height=4)

##### plot canonical R2 ####
r2 = xy.cca$cancor ^2
r.595 = apply(xy.cca$cancor.rand,2,quantile,probs=c(0.05,0.95))
r2.595 = r.595 ^2

df_plot = data.frame(r2 = r2, mode = c(1: length(r2)),
                     lci = r2.595[1,], hci = r2.595[2,])

ggplot(df_plot, aes(x=mode, y=r2, group=1)) +
  geom_line()+
  geom_point() + 
  geom_ribbon(aes(ymin = lci, ymax= hci), alpha = 0.2) +
  theme_classic()
ggsave(plt.name3,units="in", width=5, height=4, device=cairo_ps)

# ##### plot total % variance explained (behavior)####
# expvar = colSums(xy.cca$corr.Y.Cy ^2)
# df_plot = data.frame(expvar = expvar, mode = c(1: length(r2)))
# 
# ggplot(df_plot, aes(x=mode, y=expvar, group=1)) +
#   geom_line()+
#   geom_point() + 
#   theme_classic()
# ggsave(plt.name4,units="in", width=5, height=4)
# 
# ##### plot total % variance explained (brain)####
# expvar = colSums(xy.cca$corr.X.Cx ^2)
# df_plot = data.frame(expvar = expvar, mode = c(1: length(r2)))
# 
# ggplot(df_plot, aes(x=mode, y=expvar, group=1)) +
#   geom_line()+
#   geom_point() + 
#   theme_classic()
# ggsave(plt.name5,units="in", width=5, height=4)
# 
# # top 10% brain features for mode 1 based on canonical loadings
# i = 1
# cutoff = quantile(abs(xy.cca$corr.X.Cx[,i]), 0.90)
# bn_ft_idx = which(abs(xy.cca$corr.X.Cx[,i]) > cutoff)
# df_ft = bn_atlas[bn_ft_idx, c(1:4)]
# df_ft$X.Cx = xy.cca$corr.X.Cx[bn_ft_idx,i]
# write.xlsx(df_ft, file=xls.name1)
# 
# # top 10% brain features for mode 1 based on standardized coefficients
# i = 1
# cutoff = quantile(abs(xy.cca$xcoef.std[,i]), 0.90)
# bn_ft_idx = which(abs(xy.cca$xcoef.std[,i]) > cutoff)
# df_ft = bn_atlas[bn_ft_idx, c(1:4)]
# df_ft$xcoef.std = xy.cca$xcoef.std[bn_ft_idx,i]
# write.xlsx(df_ft, file=xls.name2)
