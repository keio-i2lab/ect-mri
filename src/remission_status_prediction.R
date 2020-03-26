library(xgboost)
library(randomForest)
library(e1071)
library(glmnet)

file_name <- "clinical_features.csv"
df<-read.csv(file=file_name)
df_x<-df[1:27, 2:9]
df_x<-cbind(df_x, as.factor(df$Remission_HAMD[1:27]))
colnames(df_x)[ncol(df_x)]<-'labels_data'


param_alpha<-c(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1)
param_ratio<-c(0.4, 0.5, 0.6, 0.7, 0.8)
paramSearch<-matrix(nrow=length(param_alpha), ncol=length(param_ratio))
for (u in 1:length(param_alpha)){
	for (v in 1:length(param_ratio)){
		y_p<-vector(mode="numeric", length=nrow(df_x))
		survived_features<-list()
		for (k in 1:nrow(df_x)){
			trainDF<-df_x[-k, ]
			testDF<-df_x[k, ]
			
			for (i in 1:(ncol(trainDF)-1)){
				col_mean<-mean(as.numeric(trainDF[, i]))
				col_sd<-sd(as.numeric(trainDF[, i]))
				trainDF[, i]<-(trainDF[, i] - col_mean)/col_sd
				testDF[, i]<-(testDF[, i] - col_mean)/col_sd
			}
			
			df_matrix<-as.matrix(trainDF[, -ncol(trainDF)])
			a<-glmnet(x=df_matrix, y=trainDF$labels_data, family="binomial", alpha=param_alpha[u])
			dev.ratio<-length(which(a$dev.ratio<param_ratio[v]))+1
			
			glmnet_idx<-as.vector(which(a$beta[,dev.ratio]!=0))
			survived_features[[k]]<-which(a$beta[,dev.ratio]!=0)
			
			trainDF<-trainDF[, c(glmnet_idx, ncol(trainDF))]
			testDF<-testDF[, c(glmnet_idx, ncol(testDF))]
			
			model<-randomForest(labels_data~., data=trainDF, ntree=5000, mtry=8)
			pred<-predict(model, testDF[, -ncol(testDF)])
			y_p[k]<- if (pred==0) 0 else 1
		}
		paramSearch[u, v]<-sum(df_x$labels_data == y_p)/length(y_p)
		print(paramSearch)
	}
}
temp<-vector(mode="integer", length=ncol(df_x))
for (i in 1:length(y_p)){
	temp[survived_features[[i]]]<-temp[survived_features[[i]]]+1
}


############ Naive Bayes ##############
param_alpha<-c(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1)
param_ratio<-c(0.4, 0.5, 0.6, 0.7, 0.8)
paramSearch<-matrix(nrow=length(param_alpha), ncol=length(param_ratio))
for (u in 1:length(param_alpha)){
	for (v in 1:length(param_ratio)){
		y_p<-vector(mode="numeric", length=nrow(df_x))
		survived_features<-list()
		for (k in 1:nrow(df_x)){
			trainDF<-df_x[-k, ]
			testDF<-df_x[k, ]
			
			for (i in 1:(ncol(trainDF)-1)){
				col_mean<-mean(as.numeric(trainDF[, i]))
				col_sd<-sd(as.numeric(trainDF[, i]))
				trainDF[, i]<-(trainDF[, i] - col_mean)/col_sd
				testDF[, i]<-(testDF[, i] - col_mean)/col_sd
			}
			
			df_matrix<-as.matrix(trainDF[, -ncol(trainDF)])
			a<-glmnet(x=df_matrix, y=trainDF$labels_data, family="binomial", alpha=param_alpha[u])
			dev.ratio<-length(which(a$dev.ratio<param_ratio[v]))+1
			
			glmnet_idx<-as.vector(which(a$beta[,dev.ratio]!=0))
			survived_features[[k]]<-which(a$beta[,dev.ratio]!=0)
			
			trainDF<-trainDF[, c(glmnet_idx, ncol(trainDF))]
			testDF<-testDF[, c(glmnet_idx, ncol(testDF))]
			
			model<-naiveBayes(labels_data~., data=trainDF, laplace=laplace)
			y_p[k]<-predict(model, testDF[, -ncol(testDF)], type="class")
			
			if (y_p[k] == 2){
				y_p[k]<-1
			} else {
				y_p[k]<-0
			}
			
			#y_p[k]<- if (pred==0) 0 else 1
		}
		paramSearch[u, v]<-sum(df_x$labels_data == y_p)/length(y_p)
		print(paramSearch)
	}
}
temp<-vector(mode="integer", length=ncol(df_x))
for (i in 1:27){
	temp[survived_features[[i]]]<-temp[survived_features[[i]]]+1
}


######### SVM ##########
param_alpha<-c(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1)
param_ratio<-c(0.4, 0.5, 0.6, 0.7, 0.8)
param_C<-c(2^-5, 2^-3, 2^-1, 1, 2, 2^3, 2^5, 2^7, 2^9, 2^11, 2^13, 2^15)
param_gamma<-c(2^-15, 2^-13, 2^-11, 2^-9, 2^-7, 2^-5, 2^-3, 2^-1, 1, 2, 4)
best_accuracy<-0
best_u<-0
best_v<-0
best_m<-0
best_n<-0
for (u in 1:length(param_C)){
	for (v in 1:length(param_gamma)){
		for (m in 1:length(param_alpha)){
			for (n in 1:length(param_ratio)){
				y_p<-vector(mode="numeric", length=nrow(df_x))
				survived_features<-list()
				for (k in 1:nrow(df_x)){
					trainDF<-df_x[-k, ]
					testDF<-df_x[k, ]
					
					for (i in 1:(ncol(trainDF)-1)){
						col_mean<-mean(as.numeric(trainDF[, i]))
						col_sd<-sd(as.numeric(trainDF[, i]))
						trainDF[, i]<-(trainDF[, i] - col_mean)/col_sd
						testDF[, i]<-(testDF[, i] - col_mean)/col_sd
					}
			
					df_matrix<-as.matrix(trainDF[, -ncol(trainDF)])
					a<-glmnet(x=df_matrix, y=trainDF$labels_data, family="binomial", alpha=param_alpha[m])
					dev.ratio<-length(which(a$dev.ratio<param_ratio[n]))+1
			
					glmnet_idx<-as.vector(which(a$beta[,dev.ratio]!=0))
					survived_features[[k]]<-which(a$beta[,dev.ratio]!=0)
			
					trainDF<-trainDF[, c(glmnet_idx, ncol(trainDF))]
					testDF<-testDF[, c(glmnet_idx, ncol(testDF))]
					
					model<-svm(labels_data~., data=trainDF, cost=param_C[u], gamma=param_gamma[v], kernel="radial")
					pred<-predict(model, testDF[, -ncol(testDF)])
					y_p[k]<- if (pred==0) 0 else 1
				}
				current_accuracy<-sum(df_x$labels_data == y_p)/length(y_p)
				if (current_accuracy > best_accuracy){
					best_accuracy<-current_accuracy
					best_u<-u
					best_v<-v
					best_m<-m
					best_n<-n
				} else if ((1 - current_accuracy) > best_accuracy){
					best_accuracy<- 1 - current_accuracy
					best_u<-u
					best_v<-v
					best_m<-m
					best_n<-n
				}
				print(paste("Best accuracy =", best_accuracy))
				print(paste("best m =", best_m, "best n =", best_n, "best u =", best_u, "best v =", best_v))
				print(paste("m =", m, "n =", n, "u =", u, "v =", v))
			}
		}
	}
}


######### SVM no lasso ##########
param_C<-c(2^-5, 2^-3, 2^-1, 1, 2, 2^3, 2^5, 2^7, 2^9, 2^11, 2^13, 2^15)
param_gamma<-c(2^-15, 2^-13, 2^-11, 2^-9, 2^-7, 2^-5, 2^-3, 2^-1, 1, 2, 4)
best_accuracy<-0
best_u<-0
best_v<-0
for (u in 1:length(param_C)){
	for (v in 1:length(param_gamma)){

		y_p<-vector(mode="numeric", length=nrow(df_x))
		survived_features<-list()
		for (k in 1:nrow(df_x)){
			trainDF<-df_x[-k, ]
			testDF<-df_x[k, ]
			
			model<-svm(labels_data~., data=trainDF, cost=param_C[u], gamma=param_gamma[v], kernel="radial")
			pred<-predict(model, testDF[, -ncol(testDF)])
			y_p[k]<- if (pred==0) 0 else 1
		}
		current_accuracy<-sum(df_x$labels_data == y_p)/length(y_p)
		if (current_accuracy > best_accuracy){
			best_accuracy<-current_accuracy
			best_u<-u
			best_v<-v
		} else if ((1 - current_accuracy) > best_accuracy){
			best_accuracy<- 1 - current_accuracy
			best_u<-u
			best_v<-v
		}
		print(paste("Best accuracy =", best_accuracy))
		print(paste("best u =", best_u, "best v =", best_v))
		print(paste("u =", u, "v =", v))

	}
}