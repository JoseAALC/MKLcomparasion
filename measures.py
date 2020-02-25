from __future__ import division

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

#Core
import numpy as np
import pandas as pd
import math
import time
import random

#utils
from sklearn import preprocessing
from sklearn.model_selection import KFold,RepeatedKFold,StratifiedKFold
from sklearn.metrics import make_scorer, accuracy_score,f1_score,roc_auc_score,precision_score,recall_score,confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Binarizer
from scipy.stats import wilcoxon
from imblearn.over_sampling import SMOTE 
from sklearn.neural_network import MLPClassifier
from imblearn.under_sampling import RandomUnderSampler

#models
from sklearn.tree import DecisionTreeClassifier as DT
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.naive_bayes import GaussianNB as nb
from sklearn.ensemble import RandomForestClassifier as rf

#Fearure selection
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2


from bio import genetic

#MKL
from mklaren.kernel.kinterface import Kinterface
from mklaren.kernel.kernel import linear_kernel, poly_kernel,rbf_kernel
from mklaren.mkl.alignf import Alignf



import scipy.stats as sttest


#consts
np.random.seed(800)
start=0
class_name ="class"
score = f1_score
scorer = make_scorer(f1_score)
test_rate =0.2


print("Dataset name without extension:")
dataset_name = raw_input()

results_file=open("Reports//"+dataset_name+"_results.csv","w")
report_file=open("Reports//"+dataset_name+"_report.txt","w")
pred_file=open("Reports//"+dataset_name+"_predictions.txt","w")
results_file.write("algorithm,accuracy,F1,rocauc,precision,recall\n")

data_rate =0.2
report_file.write("test portion: "+str(test_rate)+"\n")










#functions
def fix_time():
    global start
    start = time.time()

def elapsed():
    global start
    end = time.time()
    return end - start

def scores(y,evaluation,model_name,out):
   result =  str(model_name) +","+str(accuracy_score(y,evaluation))+","
   result+=  str(precision_score(y,evaluation))+","+str(recall_score(y,evaluation))+","
   result+=  str(f1_score(y,evaluation))+","+str(roc_auc_score(y,evaluation))+"\n"
   out.write(result)

def tunning_svm(samples,classes,rbf_par,poly_par,scorer,target_counts):
   
   
   svm_rbf =SVC(kernel="rbf")
   grid_obj = GridSearchCV(svm_rbf, rbf_par, scoring=scorer,cv=5)
   grid_obj = grid_obj.fit(samples,classes)
   
   svm_rbf = grid_obj.best_estimator_
   
   #print(grid_obj.best_score_ )


   gama = svm_rbf.get_params()["gamma"]
   
   svm_poly =SVC(kernel="poly")
   grid_obj = GridSearchCV(svm_poly, poly_par, scoring=scorer,cv=5)
   grid_obj = grid_obj.fit(samples, classes)
   svm_poly = grid_obj.best_estimator_
   degree = svm_poly.get_params()["degree"]
   coef0 = svm_poly.get_params()["coef0"]

   #print(grid_obj.best_score_ )

   par={"kernel":["rbf","poly","linear"],"C":[10**i for i in range(-5,3)]}
   svm =SVC(degree=degree,coef0=coef0,gamma=gama)
   grid_obj = GridSearchCV(svm, par, scoring=scorer,cv=5)
   grid_obj = grid_obj.fit(samples, classes)
   svm= grid_obj.best_estimator_

   #print(grid_obj.best_score_ )
   return svm

def createKernelCombination(kernel_indexs,samples,classes,rbf_par,poly_par,scorer):
   kernels= []
   for indexes in kernel_indexs:
      svm = tunning_svm(samples[:,indexes],classes,rbf_par,poly_par,scorer)
      kernel = svm.get_params()["kernel"]
      if kernel=="linear":
         kernels.append( Kinterface(data=x_train[:,indexes], kernel=linear_kernel))
      elif kernel =="rbf":
         gamma=svm.get_params()["gamma"]
         K = Kinterface(data=x_train[:,indexes], kernel=rbf_kernel,kernel_args={"gamma": gamma})
         kernels.append(K)
      else:
         degree=svm.get_params()["degree"]
         coef0 =degree=svm.get_params()["coef0"]
         K = Kinterface(data=x_train[:,indexes], kernel=poly_kernel,kernel_args={"degree": degree})
         kernels.append(K)

   model = Alignf(typ="convex")
   model.fit(kernels, classes.values)
   model.mu  # kernel weights (convex combination)
   mu = model.mu
   print(mu)

   combined_k = lambda x,y: \
      sum([mu[i]*kernels[i](x[:,kernel_indexs[i]],y[:,kernel_indexs[i]]) for i in range(len(kernels))])
   return combined_k

def ramdom_kernels_combination(kernel_indexs,samples,classes,rbf_par,poly_par,scorer):
   kernels= []
   for indexes in kernel_indexs:    
      choice = np.random.randint(3, size=1)[0]
      if choice ==0:
         kernels.append( Kinterface(data=x_train[:,indexes], kernel=linear_kernel))
      elif choice ==1:
         #print(rbf_par)
         length_of_param1 = len(rbf_par["gamma"])
         # print(rbf_par["gamma"])
         # print(np.random.randint(length_of_param1, size=1)[0])
         K = Kinterface(data=x_train[:,indexes], kernel=rbf_kernel,kernel_args={"gamma": rbf_par["gamma"][np.random.randint(length_of_param1, size=1)[0]]})
         kernels.append(K)
      else:
         length_of_param1 = len(poly_par["degree"])
         K = Kinterface(data=x_train[:,indexes], kernel=poly_kernel,kernel_args={"degree": poly_par["degree"][np.random.randint(length_of_param1, size=1)[0]]})
         kernels.append(K)
   #mu = [random.randrange(0,1) for i in range(40)]
   model = Alignf(typ="convex")
   model.fit(kernels, classes.values)
   model.mu  # kernel weights (convex combination)
   mu = model.mu
   print("numbers:" +str(mu))

   combined_k = lambda x,y: \
      sum([mu[i]*kernels[i](x[:,kernel_indexs[i]],y[:,kernel_indexs[i]]) for i in range(len(kernels))])
   return combined_k

def ramdom_kernels(kernel_indexs,samples,classes,rbf_par,poly_par):
    kernels= []
    for indexes in kernel_indexs:    
        choice = np.random.randint(3, size=1)[0]
        if choice ==0:
            kernels.append( Kinterface(data=x_train[:,indexes], kernel=linear_kernel))
        elif choice ==1:
            #print(rbf_par)
            length_of_param1 = len(rbf_par["gamma"])
            # print(rbf_par["gamma"])
            # print(np.random.randint(length_of_param1, size=1)[0])
            K = Kinterface(data=x_train[:,indexes], kernel=rbf_kernel,kernel_args={"gamma": rbf_par["gamma"][np.random.randint(length_of_param1, size=1)[0]]})
            kernels.append(K)
        else:
            length_of_param1 = len(poly_par["degree"])
            K = Kinterface(data=x_train[:,indexes], kernel=poly_kernel,kernel_args={"degree": poly_par["degree"][np.random.randint(length_of_param1, size=1)[0]]})
            kernels.append(K)
    return kernels


def tunningMKL(paramters,samples,classes,rbf_par,poly_par,scorer):
   kernels = []
   
   for feature_n in paramters["features"]:
      for kernel_n in paramters["kernels"]:
         kernels_indexes = [np.random.choice([i for i in range(df.shape[1]-1)],replace=False,size=(feature_n)) for j in range(kernel_n)]
         kernels.append(ramdom_kernels_combination(kernels_indexes,samples,classes,rbf_par,poly_par,scorer))
   par={"kernel":kernels,"C":[10**i for i in range(-5,3)]}
   svm =SVC()
   grid_obj = GridSearchCV(svm, par, scoring=scorer,cv=5)
   grid_obj = grid_obj.fit(samples, classes)
   svm= grid_obj.best_estimator_
   return svm

   
def kfolding2(samples,classes,model,model_name,folds = None):
   global results_file
   global report_file
   #print("Y: " +str(classes))
   
   metrics ={
      "accuracy":[],
      "F1":[],
      "precision":[],
      "recall":[],
      "rocauc":[]
   }



   k_folds = folds
   if folds == None:
      #k_folds =StratifiedKFold(n_splits=5 , random_state=0)
      k_folds = RepeatedKFold(n_splits=2, n_repeats=5, random_state=0)
      print("F: "+str(k_folds.split(samples)))
   
   predictionsTotal = np.array([])
   #for train_index, test_index in k_folds.split(samples,classes):
   for train_index, test_index in k_folds.split(samples):
      model.fit(samples[train_index,:],classes.iloc[train_index])
      predictions = model.predict(samples[test_index])
      predictionsTotal=np.concatenate([predictionsTotal,predictions])
      metrics["accuracy"].append(accuracy_score(classes.iloc[test_index], predictions))
      metrics["F1"].append(f1_score(classes.iloc[test_index], predictions))
      metrics["rocauc"].append(roc_auc_score(classes.iloc[test_index], predictions))
      metrics["precision"].append(precision_score(classes.iloc[test_index], predictions))
      metrics["recall"].append(recall_score(classes.iloc[test_index], predictions))

      tn, fp, fn, tp = confusion_matrix(classes.iloc[test_index], predictions).ravel()
      report_file.write(model_name +" MATRIX:\n")
      report_file.write("tn: " +str(tn)+"\n")
      report_file.write("fp: " +str(fp)+"\n")
      report_file.write("fn: " +str(fn)+"\n")
      report_file.write("tp: " +str(tp)+"\n")
      report_file.write("END MATRIX\n")
     # print("tn: " +str(tn))


   metrics["accuracy"]=np.array(metrics["accuracy"])
   metrics["F1"]=np.array(metrics["F1"])
   metrics["rocauc"]=np.array(metrics["rocauc"])
   metrics["precision"]=np.array(metrics["precision"])
   metrics["recall"]=np.array(metrics["recall"])
   results_string = str(np.mean(metrics["accuracy"])) +"+"+str(np.std(metrics["accuracy"]))+"," 
   results_string+= str(np.mean(metrics["F1"]))  +"+"+str(np.std(metrics["F1"])) +","
   results_string+=str(np.mean(metrics["rocauc"]))+ "+"+str(np.std(metrics["rocauc"])) +","
   results_string+=str(np.mean(metrics["precision"]))+ "+"+str(np.std(metrics["precision"])) +","
   results_string+=str(np.mean(metrics["recall"]))+ "+"+str(np.std(metrics["recall"])) +","
  
   results_file.write(model_name +","+results_string+"\n")
   
   return (k_folds,predictionsTotal)

def kfolding(samples,classes,model,model_name,folds = None):
   global results_file
   global report_file
   #print("Y: " +str(classes))
   


   predictions = model.predict(samples)


   tn, fp, fn, tp = confusion_matrix(classes, predictions).ravel()
   report_file.write(model_name +" MATRIX:\n")
   report_file.write("tn: " +str(tn)+"\n")
   report_file.write("fp: " +str(fp)+"\n")
   report_file.write("fn: " +str(fn)+"\n")
   report_file.write("tp: " +str(tp)+"\n")
   report_file.write("END MATRIX\n")



   results_string = str(accuracy_score(classes, predictions))+"," 
   results_string+= str(f1_score(classes, predictions)) +","
   results_string+=str(roc_auc_score(classes, predictions)) +","
   results_string+=str(precision_score(classes, predictions)) +","
   results_string+=str(recall_score(classes, predictions)) +","
  
   results_file.write(model_name +","+results_string+"\n")
   
   return ("k_folds",predictions)


def fitness1(genom,extra=None):
    datx = extra["X"]
    daty = extra["Y"]
    kernel_indexs = extra["Ki"]
    kernels= extra["K"]



    combined_k = lambda x,y: \
      sum([genom[i]*kernels[i](x[:,kernel_indexs[i]],y[:,kernel_indexs[i]]) for i in range(len(kernels))])
    
    score =0
   

    for i in range(10):
        x_train, x_test, y_train, y_test = train_test_split(datx,daty,test_size =test_rate,stratify=daty,random_state=0)
        cls =SVC(kernel=combined_k)
        cls.fit(x_train, y_train)
        predictions = cls.predict(x_test)
        score+=f1_score(y_test, predictions)


    return score/10

     
def counting(df):
   global class_name
   zeros =0
   ones =0
   for i in range(df.shape[0]):
      #print(df[class_name][i] )
      if df[class_name][i] == 0:
         zeros+=1
      else:
         ones+=1

   return [zeros,ones]   


def prepro(df):
   #print(df)
   print(counting(df))

   print(df.shape)
   data_rate=2400/df.shape[0]
   print("Rate",data_rate)
   if data_rate >1:
      data_rate=1
   report_file.write("data portion used: "+str(data_rate)+"\n")
  
   #sample data
   df = df.sample(frac=data_rate, replace=False,random_state=0)
   #print("Size: ",df.shape)
   #remove repeated rows
   df = df.drop_duplicates()
   #print("Size2: ",df.shape)
   #print(df.head())
   #dropnas
   #df.isna().sum()
   df=df.dropna(axis=0)
   #print("Size3: ",df.shape)
   #print(df["Attr1"].value_counts())
   #binarize categorical values
   features = list(df.head(0)) 
   colection = []
   names =[]
   for f in features:
      if df[f].dtype =='O' and f!=class_name :
         values =df[f].unique().tolist()
         maping = {}
         for i in range(len(values)):
            maping[i]=values[i]
         #ordvar = df[f].replace(maping)
         #df[f] =pd.factorize(ordvar)[0]
         
         colection.append(pd.get_dummies(df[f],prefix=f).iloc[:,1:])
         names.append(f)
   if(len(colection)>0):
      #print(colection)
      df =df.drop(names,axis=1)
      
      concatdf  =pd.concat(colection,axis =1)
      
      df = pd.concat([df,concatdf],axis=1)
      #print(len(df.head(0)))
      df.shape

   #print(df)
   #print("Size: ",df.shape)
   report_file.write("data size: "+str(df.shape)+"\n")

   #get class distribuition

   target_counts = df[class_name].value_counts()
   rate_of_maiority = max(target_counts)/sum(target_counts) 
   print(rate_of_maiority  )
   report_file.write("portion of class: "+str(max(target_counts)/sum(target_counts))+"\n")

   #reduce to featureset and class
   X_all = df.drop([class_name],axis=1)
   y_all = df[class_name]
   #print(y_all)
   #print(X_all)
   #rebalanced data
   if rate_of_maiority >= 0.6:
      print("Rebalancing data")
      sm = RandomUnderSampler(random_state=42)
      X_all, y_all = sm.fit_resample(X_all, y_all)
      #features.remove("class")
     # X_all = pd.DataFrame.from_records(X_all)
      #print("Y:",y_all)
     # print(y_all)
     # y_all = np.reshape(y_all, (-1, 1))
     # print(y_all)
     # y_all = pd.DataFrame.from_records(y_all)
     # print(y_all)
      #print ("X: ", X_all) 
   else:
      y_all=y_all.values
     # print("Y:",y_all)
      #print("X: ",X_all)



   #normalize
   #X_all = preprocessing.normalize(X_all)
   X_all = preprocessing.MinMaxScaler((0,1)).fit(X_all).transform(X_all)
   #print(X_all[0:5,:])

   #print head
   #print(df.head(0))


   #generate train and test_set
   x_train, x_test, y_train, y_test = train_test_split(X_all,y_all,test_size =test_rate,stratify=y_all,random_state=0)

   return x_train, x_test,y_train,y_test
#-----------------------------------------END HEADER-----------------------------------------------



#----------------------------------------PREPROCESSING----------------------------------------------


pd.set_option("display.max_columns",500)
#get data
df = pd.read_csv("Clean Datasets//"+dataset_name+".csv")
#print(df.shape)
#print(df.describe())
#df[class_name] =  df[class_name].map({'Ghoul':0,'Ghost':1,"Goblin":1})






x_train, x_test, y_train, y_test = prepro(df)

print("The shape: ",x_train.shape)


gammas = [i /df.shape[1] for i in range(1,7,1)]

parameters ={
             "C":[10**i for i in range(-5,3)],
          "gamma":gammas}
pparameters ={
             "C":[10**i for i in range(-5,3)],
          "degree":[1,2,3]}



predictionsTest = {}





target_counts = counting(df)

#decision tree
cls = DT()

fix_time()
cls.fit(x_train, y_train)
report_file.write("decision tree time: " + str(elapsed())+str("\n"))

predictions = cls.predict(x_test)

print("Tree:")
predictionsTest["Tree"] = kfolding(x_test,y_test,cls,"Tree")[1]

pred_file.write("Tree: " +str(predictionsTest["Tree"])+"\n")


#KNN
cls =KNN()

parameters ={
             "n_neighbors":[1,3,5,7,9,11]}
fix_time()
grid_obj = GridSearchCV(cls, parameters, scoring=scorer,cv=5)
grid_obj = grid_obj.fit(x_train, y_train)

cls = grid_obj.best_estimator_
report_file.write("KNN time: " + str(elapsed())+str("\n"))

print("KNN:")
predictionsTest["KNN"] =kfolding(x_test,y_test,cls,"KNN")[1]
pred_file.write("KNN: " +str(predictionsTest["KNN"])+"\n")
#Naive Bayes
cls =nb()

parameters ={
             "var_smoothing":[1e-09,1e-08,1e-07,1e-06,1e-05,1e-04,1e-03,1e-02,1e-01,1]}

fix_time()
grid_obj = GridSearchCV(cls, parameters, scoring=scorer,cv=5)
grid_obj = grid_obj.fit(x_train, y_train)

cls = grid_obj.best_estimator_
report_file.write("NB time: " + str(elapsed())+str("\n"))

print("Naive Bayes:")
predictionsTest["Naive Bayes"] = kfolding(x_test,y_test,cls,"Naive Bayes")[1]
pred_file.write("NB: " +str(predictionsTest["Naive Bayes"])+"\n")
#Random Forest
cls =rf()

parameters ={
             "n_estimators":[i*10 for i in range(10,50,100)]}

fix_time()
grid_obj = GridSearchCV(cls, parameters, scoring=scorer,cv=5)
grid_obj = grid_obj.fit(x_train, y_train)

cls = grid_obj.best_estimator_
report_file.write("Random Forest time: " + str(elapsed())+"\n")

print("Random Forest:")
predictionsTest["Random Forest"]=kfolding(x_test,y_test,cls,"Random Forest")[1]
pred_file.write("RF: " +str(predictionsTest["Random Forest"])+"\n")
#SVM

gammas = [i /df.shape[1] for i in range(1,7,1)]
#print(df.shape)
#print(gammas)
parameters ={
             "C":[10**i for i in range(-5,3)],
          "gamma":gammas}
pparameters ={
             "C":[10**i for i in range(-5,3)],
          "degree":[1,2,3]}

print("SVM:")
fix_time()
cls=tunning_svm(x_train,y_train,parameters,pparameters,scorer,target_counts)
report_file.write("SVM time: " + str(elapsed())+str("\n"))



predictionsTest["SVM"] =kfolding(x_test,y_test,cls,"SVM")[1]
pred_file.write("SVM: " +str(predictionsTest["SVM"])+"\n")


#Neural network
print("Neural Network:")
fix_time()
cls = MLPClassifier(hidden_layer_sizes=(100,), random_state=1)
cls.fit(x_train, y_train)
report_file.write("Neural Network time: " + str(elapsed())+str("\n"))
predictionsTest["Neural Network"]=kfolding(x_test,y_test,cls,"Neural Network")[1]
pred_file.write("NN: " +str(predictionsTest["Neural Network"])+"\n")

pred_file.close()
results_file.close()
report_file.close()
exit()


