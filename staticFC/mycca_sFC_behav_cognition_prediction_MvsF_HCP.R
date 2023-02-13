setwd("/Users/zhangyuan/Google Drive/2021_HCP_Gender_DNN")
rm(list=ls())
source("scripts/brainbeh/2022CCA/myRFunc_updated.R")

library(ggplot2)

#################################################
### test if male models can predict in female ###
### and if female models can predict in male  ###
### for both HCP data and the pooled data    ###
#################################################

# load male CCA model
midfix = "HCP_RL_S1_CCA_msubjs_sFC_cognition" 
fname = paste("results/brain_behav/sFC_PCversion/",midfix,"_perm5000_PCversion.RData",sep="")
load(fname)
plt.name1 = paste("results/brain_behav/sFC_PCversion/", midfix, "_predictFemale_CxCy_scatter_perm5000.eps", sep="")

X.male = x_mtx
Y.male = y_mtx
xy.cca.male = xy.cca
rm(x_mtx, y_mtx, xy.cca)

# load female CCA model
midfix = "HCP_RL_S1_CCA_fsubjs_sFC_cognition" 
fname = paste("results/brain_behav/sFC_PCversion/",midfix,"_perm5000_PCversion.RData",sep="")
load(fname)
plt.name2 = paste("results/brain_behav/sFC_PCversion/", midfix, "_predictMale_CxCy_scatter_perm5000.eps", sep="")


X.female = x_mtx
Y.female = y_mtx
xy.cca.female = xy.cca
rm(x_mtx, y_mtx, xy.cca)

#####################################################################
############# CCA prediction (male model, female data) ##############
#####################################################################
# set up X and Y
X = data.matrix(X.female)
Y = data.matrix(Y.female)

x_mtx <- data.matrix(X, rownames.force = NA)
y_mtx <- data.matrix(Y, rownames.force = NA)
x_mtx = scale(x_mtx)
y_mtx = scale(y_mtx)

Cy <- (y_mtx %*% xy.cca.male$ycoef.std) #raw and std coefs are same as data were scaled before CCA
Cx <- (x_mtx %*% xy.cca.male$xcoef.std) 

diag(cor(Cx,Cy))
diag(cor(Cx,Cy))^2
cancor.perm.test(Cy, Cx, xy.cca.female$perm.idx)
cor.test(Cx[,1],Cy[,1])

# # dimension test
# rho = diag(cor(Cx,Cy))
# n = dim(Cx)[1]
# p = dim(x_mtx)[2]
# q = dim(y_mtx)[2]
# p.asym(rho, n, p, q, tstat = "Wilks")

# plotting
Cy1 = Cy[,1] # behavior
Cx1 = Cx[,1] # brain
df_plot = data.frame(Cx1 = Cx1, Cy1=Cy1)

ggplot(df_plot, aes(x=Cx1, y=Cy1)) +
  geom_point() + 
  geom_smooth(method=lm, se=FALSE, fullrange=TRUE) +
  labs(x="Mode 1 (Brain)", y="Mode 1 (Behavior)") +
  theme_bw() + theme(axis.text.x=element_text(size=12, angle=0, vjust=0.3),
                     axis.text.y=element_text(size=12, angle=0, vjust=0.3),
                     plot.title=element_text(size=12))
ggsave(plt.name1,units="in", width=5, height=4)


#####################################################################
############# CCA prediction (female model, male data) ##############
#####################################################################
# set up X and Y
X = data.matrix(X.male)
Y = data.matrix(Y.male)

x_mtx <- data.matrix(X, rownames.force = NA)
y_mtx <- data.matrix(Y, rownames.force = NA)
x_mtx = scale(x_mtx)
y_mtx = scale(y_mtx)

Cy <- (y_mtx %*% xy.cca.female$ycoef.std) 
Cx <- (x_mtx %*% xy.cca.female$xcoef.std)

diag(cor(Cx,Cy))
diag(cor(Cx,Cy))^2
cancor.perm.test(Cy, Cx, xy.cca.male$perm.idx)

# # dimension test
# rho = diag(cor(Cx,Cy))
# n = dim(Cx)[1]
# p = dim(x_mtx)[2]
# q = dim(y_mtx)[2]
# p.asym(rho, n, p, q, tstat = "Wilks")

# plotting
Cy1 = Cy[,1] # behavior
Cx1 = Cx[,1] # brain
df_plot = data.frame(Cx1 = Cx1, Cy1=Cy1)

ggplot(df_plot, aes(x=Cx1, y=Cy1)) +
  geom_point() + 
  geom_smooth(method=lm, se=FALSE, fullrange=TRUE) +
  labs(x="Mode 1 (Brain)", y="Mode 1 (Behavior)") +
  theme_bw() + theme(axis.text.x=element_text(size=12, angle=0, vjust=0.3),
                     axis.text.y=element_text(size=12, angle=0, vjust=0.3),
                     plot.title=element_text(size=12))
ggsave(plt.name2,units="in", width=5, height=4)
