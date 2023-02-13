setwd("/Users/zhangyuan/Google Drive/2021_HCP_Gender_DNN")
rm(list=ls())
source("scripts/brainbeh/2022CCA/myRFunc_updated.R")

library(ggplot2)
# library(ggrepel)
# library(reshape2)

#### load existing results for plotting ####
midfix = "HCP_LR_S1_CCA_msubjs_sFC_cognition"
fname = paste("results/brain_behav/sFC_PCversion/",midfix,"_perm5000_PCversion.RData",sep="")
load(fname)

plt.name1 = paste("results/brain_behav/sFC_PCversion/", midfix, "_CxCy_scatter_perm5000.eps", sep="")
plt.name2 = paste("results/brain_behav/sFC_PCversion/", midfix, "_modeR2_perm5000.eps", sep="")

##### plot scatter plot (canonical scores of mode 1) ####
Cy1 = xy.cca$Cy[,1] # behavior
Cx1 = xy.cca$Cx[,1] # brain
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
ggsave(plt.name1,units="in", width=5, height=4)

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
ggsave(plt.name2,units="in", width=5, height=4, device=cairo_ps)

