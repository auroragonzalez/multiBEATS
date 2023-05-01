source("../../01.functions.R")
library("reticulate")

np <- import("numpy")
library(R.matlab)
df <- readMat("ArabicDigits.mat")

y_train <- df$mts[1]
X_train <- df$mts[2]
y_test <- df$mts[3]
X_test <- df$mts[4]
lista = list()


SL = 3
WD = 3

serie1= X_train[[1]][[1]]
serie1 = serie1[[1]]
NROW = nrow(serie1)
NCOL = ncol(serie1)

xx = serie1
#xx = as.numeric(serie1)
#xx = matrix(xx, nrow = NROW, ncol =  NCOL, byrow = T)
yy = f.multiBEATS(xx, slide = SL, window = WD, nfeat = 2)
lista[[1]] <- yy

for(i in 2:(ncol(X_train[[1]]))){
  #print(i)
  serie1= X_train[[1]][[i]]
  serie1 = serie1[[1]]
  NROW = nrow(serie1)
  NCOL = ncol(serie1)
  xx = serie1
  yy = f.multiBEATS(xx, slide = SL, window = WD, nfeat = 2)
  lista[[i]] <- yy
}

library("str2str")
ncolm <- max(unlist(lapply(lista, ncol)))


f.zeros = function(x){
  if(ncol(x) <ncolm){
    mzeros = matrix(0, nrow = NROW, ncol = ncolm-ncol(x))
    cbind(x, mzeros)
  }
  else{
    x
  }
}
x = lapply(lista, FUN = f.zeros)
k = lm2a(x , dim.order = c(3,1,2))
np$save("X_trainB.npy", k)

serie1= X_test[[1]][[1]]
serie1 = serie1[[1]]
NROW = nrow(serie1)
NCOL = ncol(serie1)

xx = serie1
#xx = as.numeric(serie1)
#xx = matrix(xx, nrow = NROW, ncol =  NCOL, byrow = T)
yy = f.multiBEATS(xx, slide = SL, window = WD, nfeat = 2)
lista = NULL 
lista[[1]] <- yy

#for(i in 2:(ncol(X_test[[1]]))){
for(i in 2:(ncol(X_test[[1]]))){
  serie1= X_test[[1]][[i]]
  serie1 = serie1[[1]]
  NROW = nrow(serie1)
  NCOL = ncol(serie1)
  xx = serie1
  yy = f.multiBEATS(xx, slide = SL, window = WD, nfeat = 2)
  lista[[i]] <- yy
}
#ncolm <- max(unlist(lapply(lista, ncol)))

f.zeros = function(x){
  if(ncol(x) <ncolm){
    mzeros = matrix(0, nrow = NROW, ncol = ncolm-ncol(x))
    cbind(x, mzeros)
  }
  else{
    x
  }
}
x = lapply(lista, FUN = f.zeros)
k = lm2a(x , dim.order = c(3,1,2))
np$save("X_testB.npy", k)


