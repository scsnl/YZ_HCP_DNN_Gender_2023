setwd("/Users/zhangyuan/Google Drive/2021_HCP_Gender_DNN")
rm(list=ls())

library(dplyr) 
library(ggplot2)
library(ggpubr)
library(ggforce)

hcp_sessions = c("LR_S1","LR_S1","RL_S1","RL_S1","LR_S1","RL_S1")
test_sessions = c("LR_S1","LR_S2","RL_S1","RL_S2","nki_645","nki_645")

#i = 1
for(i in 1:6){
   fname1 = paste("data/distance/distanceM_HCP_model_",hcp_sessions[i],"_index_2_test_",test_sessions[i],".csv",sep="")
   fname2 = paste("data/distance/distanceF_HCP_model_",hcp_sessions[i],"_index_2_test_",test_sessions[i],".csv",sep="")
   
   Mdist = read.csv(fname1,head=F)
   Fdist = read.csv(fname2,head=F)
   numM = dim(Mdist)[1]
   numF = dim(Fdist)[1]
   
   df = data.frame(Similarity = c(Mdist[,1],Mdist[,2],Fdist[,1],Fdist[,2]),
                   indG = c(rep("M",numM*2),rep("F",numF*2)), 
                   grpG = c(rep(c("M","F"),each=numM),rep(c("F","M"),each=numF)))
   df$indG = as.factor(df$indG)
   df$indG = factor(df$indG, levels=levels(df$indG)[c(2,1)])
   df$grpG = as.factor(df$grpG)
   df$grpG = factor(df$grpG, levels=levels(df$grpG)[c(2,1)])
   
   ggplot(df, aes(x = indG, y = Similarity)) + 
     geom_sina(aes(color = grpG), size = 0.3,
               position = position_dodge(0.4))+
     geom_boxplot(aes(fill=grpG, alpha=1),width = 0.2,
                  outlier.size = 0, outlier.stroke = 0,
                  position = position_dodge(0.4))+
     scale_fill_manual(values = c("#66C9D3","#DB5F57"))+
     scale_color_manual(values = c("#66C9D3","#DB5F57")) +
     theme_bw()
   fname = paste("data/distance/HCP_model_",hcp_sessions[i],"_index_2_test_",test_sessions[i],".eps",sep="")
   ggsave(fname,units="in", width=6, height=4, device=cairo_ps)
                   
}