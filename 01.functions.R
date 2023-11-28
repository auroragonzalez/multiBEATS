library("dtt")

#We create the U matrix for DCTs, based on this information: [http://www.whydomath.org/node/wavlets/dct.html](http://www.whydomath.org/node/wavlets/dct.html)

f.U <- function(n){
  res =NULL
  for (k in 0:(n - 1)) {
    res[[k + 1]] <- c(cos(pi/n * ((0:(n - 1)) + 0.5) * k))
  }
  res[[1]] =   res[[1]]*1/(sqrt(2))
  U = 1/2*matrix(unlist(res), nrow=length(res), byrow = T)
}

f.DCT <- function(A,U){
  B = U%*%A%*%t(U)
  return (B)  
}

# Sliding windows: We create a function that returns a list of indices of the data that we are going to consider for each matrix that we are going to compute the eigenvalues from (depending on the window size and the number of data we slide for each group)

f.sliwi <- function(slide=8, window=64,size_data = n){
  iM <- list()
  nM <- as.integer((size_data-window)/slide + 1) # number of matrices
  for(i in 0:(nM-1)){
    index = (1+i*slide):(window+i*slide)
    iM[[i+1]] <- index
  }
  return(iM)  
}


f.BEATS = function(sc,sl = 64,wd = 64, size =(ncol(sc)-1)){
  beatsData = NULL
  indices = f.sliwi(slide = sl, window = wd, size_data = size)
  for(i in 1:nrow(sc)){
    v = NULL
    a <- t(sc[i,-1])
    for(k in 1:length(indices)){
      matriz = matrix(a[indices[[k]]],sqrt(wd))
      v1 <- f.DCT(matriz,f.U(sqrt(wd)))
      x = Mod(eigen(round(v1/Z, f.roundingfactor(max(v1)))[1:4, 1:4])$values)
      sentence = !duplicated(x)
      if(sum(sentence)>= 3){
        x = x[sentence][1:3]
      }
      if(sum(sentence)<3){
        sentence[which(!sentence)[1]] = T
        x = x[sentence]
      }
      v = c(v, x)
    }
    beatsData <- rbind(beatsData, v)
  }
  rownames(beatsData) = 1:nrow(sc)
  return(beatsData)
}


f.DCT <- function(A,U){
  B = U%*%A%*%t(U)
  return (B)  
}



f.sliwi <- function(slide=8, window=64,size_data = n){
  iM <- list()
  nM <- as.integer((size_data-window)/slide + 1) # number of matrices
  for(i in 0:(nM-1)){
    index = (1+i*slide):(window+i*slide)
    iM[[i+1]] <- index
  }
  return(iM)  
}



## SIMPLE EXAMPLE
set.seed(1)
ts1 <- rnorm(100,3,10)
ts2 <- rnorm(100,3,10)
ts3 <- rnorm(100,3,10)
ts4 <- rnorm(100,3,10)

df = data.frame(ts1,ts2,ts3,ts4)
mat = as.matrix(t(df))

sl=8
wd=64 # should be 64
size_data = ncol(mat)
indices = f.sliwi(slide = sl, window = wd, size_data = ncol(mat))
k=1# k in 1:length(indices)
matriz = mat[,indices[[1]]]

# 2D DCT from https://cran.r-project.org/web/packages/dtt/dtt.pdf
v1 <- mvdtt(matriz, type = c("dct"), variant = 2, inverted = FALSE)

# how to compute eigenvectors for no-square matrices:
#https://sites.math.washington.edu/~greenbau/Math_554/Course_Notes/ch1.5.pdf


sv =svd(v1)
# projection ofa data v1 sover vectors v
v1%*%sv$v


## DEFINITION

f.multiBEATS = function(sc, slide = 8, window = 64, nfeat = 2){
  try(if(nfeat > window) stop("window size cannot be smaller than the number of features to be extracted"))
  multibeatsData = NULL
  indices = f.sliwi(slide = slide, window = window, size_data = ncol(sc))
  for(k in 1:length(indices)){
    matriz = as.matrix(sc[,indices[[k]]])
    # 2D DCT from https://cran.r-project.org/web/packages/dtt/dtt.pdf
    v1 <- mvdtt(matriz, type = c("dct"), variant = 2, inverted = FALSE)
    sv =svd(v1)
    S = sv$v
    V = sv$v
    # projection ofa data v1 sover vectors v
#    newb = v1%*%sv$v[,1:nfeat]
    newb = v1%*%V
    newb = newb[,1:nfeat]   # I DO NOT UNDERSTAND THIS ONE STEP
    multibeatsData <- cbind(multibeatsData, newb)
  }
  multibeatsData
}
  
f.multiBEATS(mat, slide = 8, window = 64)
