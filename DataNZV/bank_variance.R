library(caret)
train=read.csv("D:/russianVariance/train.csv/train.csv",header=T,sep=",")
test=read.csv("D:/russianVariance/test.csv/test.csv",header=T,sep=",")
nzv=nearZeroVar(train,saveMetrics = T)
nzv[nzv$nzv==TRUE,]
 
nzv_features=row.names(nzv[nzv$nzv==TRUE,])


train=train[,-which(names(train)%in%nzv_features)]
test=test[,-which(names(test)%in%nzv_features)]
write.csv(train,"D:/russianVariance/train.csv/trainNzv.csv",row.names = F)
write.csv(test,"D:/russianVariance/train.csv/testNzv.csv",row.names = F)