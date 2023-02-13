library(CCA)
library(CCP) # for cca statistical test
library(permute)

############################################################
########## BASIC INFO (Same across HCP sessions) ###########
############################################################
# bn roi IDs and labels
bn_atlas = read.csv("/Users/zhangyuan/Google Drive/2021_HCP_Gender_DNN/data/bn_atlas.csv",header=T)


# cognition_unadj_refined = c("PicSeq_Unadj",
#                             "CardSort_Unadj",
#                             "Flanker_Unadj",
#                             "PMAT24_A_CR",
#                             "ReadEng_Unadj",
#                             "PicVocab_Unadj",
#                             "ProcSpeed_Unadj",
#                             "DDisc_AUC_200", "DDisc_AUC_40K",
#                             "VSPLOT_TC",
#                             "SCPT_TP", "SCPT_TN",
#                             "IWRD_TOT",
#                             "ListSort_Unadj")


################
##### CCA ######
################
mycca <- function(X,Y,nperm){
  
  epsilon = sqrt(.Machine$double.eps)
  
  # normalization before cca 
  x_mtx <- data.matrix(X, rownames.force = NA)
  y_mtx <- data.matrix(Y, rownames.force = NA)

  x_mtx = scale(x_mtx)
  y_mtx = scale(y_mtx)
  
  X = x_mtx
  Y = y_mtx
  Y.c <- scale(Y, center = TRUE, scale = FALSE)
  X.c <- scale(X, center = TRUE, scale = FALSE)
  S11 <- cov(Y)
  S22 <- cov(X)
  S12 <- cov(Y,X)
  S11.chol <- chol(S11)
  S11.chol.inv <- solve(S11.chol)
  S22.chol <- chol(S22)
  S22.chol.inv <- solve(S22.chol)
  # K summarizes the correlation structure between the two sets of variables
  K <- t(S11.chol.inv) %*% S12 %*% S22.chol.inv
  K.svd <- svd(K)
  Eigenvalues <- K.svd$d^2
  axenames <- paste("CanAxis",seq_along(K.svd$d),sep="")
  U <- K.svd$u
  V <- K.svd$v
  A <- S11.chol.inv %*% U # raw canonical coefficients
  B <- S22.chol.inv %*% V
  Cy <- (Y %*% A)
  Cx <- (X %*% B)
  ## Compute the 'Biplot scores of Y and X variables' a posteriori
  corr.Y.Cy <- cor(Y.c, Cy)  # To plot Y in biplot in space Y
  corr.Y.Cx <- cor(Y.c, Cx)  # Available for plotting Y in space of X
  corr.X.Cy <- cor(X.c, Cy)  # Available for plotting X in space of Y
  corr.X.Cx <- cor(X.c, Cx)  # To plot X in biplot in space X
  ## Add names
  colnames(corr.Y.Cy) <- colnames(corr.Y.Cx) <- axenames
  colnames(corr.X.Cy) <- colnames(corr.X.Cx) <- axenames
  colnames(A) <- colnames(B) <- axenames
  rownames(A) <- colnames(Y)
  rownames(B) <- colnames(X)
  # standardized canonical coefficient
  Astd <- diag(sqrt(diag(cov(Y)))) %*% A
  Bstd <- diag(sqrt(diag(cov(X)))) %*% B
  
  # Compute Pillai's trace = sum of the canonical eigenvalues
  #                        = sum of the squared canonical correlations
  S11.inv <- S11.chol.inv %*% t(S11.chol.inv)
  S22.inv <- S22.chol.inv %*% t(S22.chol.inv)
  gross.mat <- S12 %*% S22.inv %*% t(S12) %*% S11.inv
  PillaiTrace <- sum(diag(gross.mat))
  
  n = nrow(X)
  pp = ncol(y_mtx)
  qq = ncol(x_mtx)
  s = min(pp,qq)
  df1 = max(pp,qq)
  df2 = (n - max(pp,qq) - 1)
  
  Fval  <- (PillaiTrace*df2)/((s-PillaiTrace)*df1)
  Fref <- Fval
  p.Pillai <- pf(Fval, s*df1, s*df2, lower.tail=FALSE)
  
  # permutation tests 
  nperm = nperm
  perm.idx = shuffleSet(n,nperm) # a row = a permutation including n (number of observations) indices
  
  # permutation test for pillai (model fit)
  pillai.p.perm = c()
  for(perm in c(1:nperm)){
    idx = perm.idx[perm,]
    S12.per <- cov(Y[idx,], X)
    gross.mat.per <- S12.per %*% S22.inv %*% t(S12.per) %*% S11.inv
    Pillai.per <- sum(diag(gross.mat.per))
    Fper  <- (Pillai.per*df2)/((s-Pillai.per)*df1)
    pillai.p.perm = c(pillai.p.perm, Fper >= (Fref-epsilon))  
  }
  pillai.p.perm = (sum(pillai.p.perm)+1)/(nperm + 1)
  
  # permutation test for canonical correlation (with CCA rerun)
  r.actual = diag(cor(Cx,Cy))
  
  cancor.p.perm = c()
  df.rrand = c()
  for(perm in c(1:nperm)){
    idx = perm.idx[perm,]
    tmp = cc(X,Y[idx,])
    r.rand = tmp$cor
    df.rrand = rbind(df.rrand,r.rand)
  }
  
  df.ractual = do.call("rbind", replicate(nperm, t(r.actual), simplify = FALSE))
  r.perm = df.rrand > df.ractual
  cancor.p.perm = colSums(r.perm) / dim(r.perm)[1]
  cancor.p.perm.adjust = p.adjust(cancor.p.perm,method="fdr",n=length(r.actual))
  
  # output
  out = list(Pillai=PillaiTrace, Eigenvalues=Eigenvalues, cancor=K.svd$d,
             xcoef.raw = B, ycoef.raw = A,
             xcoef.std = Bstd, ycoef.std = Astd,
             pillai.p = p.Pillai, pillai.p.perm = pillai.p.perm, 
             Cy=Cy, Cx=Cx,
             corr.Y.Cy=corr.Y.Cy, corr.X.Cx=corr.X.Cx, 
             corr.Y.Cx=corr.Y.Cx, corr.X.Cy=corr.X.Cy,
             cancor.rand=df.rrand, 
             cancor.p.perm=cancor.p.perm, cancor.p.perm.adjust=cancor.p.perm.adjust,
             nperm=nperm, perm.idx=perm.idx)

  out
}

#######################################
### permutation test for canonical ####
### correlation without rerun CCA  ####
#######################################
cancor.perm.test <- function(Cy, Cx, perm.idx){
  n = dim(Cy)[1]
  naxis = dim(Cy)[2]
  nperm = dim(perm.idx)[1]
  cancor.ref = matrix(rep(diag(cor(Cy,Cx)),nperm),nperm,naxis,byrow=TRUE)
  
  df = matrix(0,nperm,naxis)
  for (i in c(1:nperm)){
    idx = perm.idx[i,]
    # df[i,] = diag(cor(Cy[idx,],Cx,method="spearman"))
    df[i,] = diag(cor(Cy[idx,],Cx,method="pearson"))
  }
  p.perm = (df > cancor.ref)
  p.perm = (colSums(p.perm) + 1)/(nperm + 1)
  #p.perm = colSums(p.perm)/nperm
  
  p.perm
}