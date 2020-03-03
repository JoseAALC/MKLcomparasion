# MKLcomparasion
Code use in MSc thesis where a MKL method based in genetic algorithm to learn the weights of kernels was compared with other ML methods.

# Files
bio.py - implementation of genetic algorithm used to get the weights of MKL
testbio.py - trains and MKL and exports many metrics requires a dataset as input
measures.py - trains every other model and exports many metrics requires a dataset as input
volcanos_pred.py - used only in volcanos2 dataset with the methodology explained in the theseis

other files where used to do other tests that were not included in the thesis.

# Datasets
The datasets where to big to include in this repository. If you need then contact me I can try to send then to you.

# Software and tools
The code was written in python 2.7 because we started by using the Mklaren tool, which was written in that version of python. We also used the tools available in the {\tt scikit-learn} package to create every model used in this work. We also used the imbalanced-learn package to rebalance the data in unbalanced datasets. Finally, we use part of the tool mklaren available in GitHub https://github.com/mstrazar/mklaren.

# Methodology
## Preprocessing
We first started by removing duplicate rows in the dataset to avoid having the same data multiple times across the samples. Leaving duplicates in the dataset could bias some models and, in some cases, the time complexity increases with the number of samples which could make our model training slower with duplicates. Also, all rows that have missing data present are eliminated.

Then we take each categorical variable and create N-1 variables to each one of the N possible values that indicate if that value is present for that variable(binarization). Converted all categorical variables into multiple boolean variables.

Then the data is balanced if the class with more frequency is at least 60\% of all data. In this work, that value is the minimum necessary to consider the data unbalanced. We performed under-sampling as a method to rebalance the data.

Finally, the data is normalized because some models are sensitive to the range of those values, for example, K-Nearest Neighbors.

## Exprimental methodology

Before training the models, we preprocess the data. We keep at most 1200 instances in each dataset in order to reduce execution time. The undersampling is performed randomly. A second preprocessing is performed in one of the datasets in order to artificially introduce multimodal characteristics (this is explained later). After preprocessing, each dataset is balanced, if needed, according to the class values using undersampling, as explained before. We then split the dataset in 20\% for validation and 80\% for training.

This work tests neural networks, decision trees, naive Bayes, random forest, and KNN against our approach of MKL. For tuning, we use 5 fold cross-validation, optimization by F1-score and a grid search method which makes the number of experiences for each model, dependent on the number of combinations of parameters. For KNN we use 1, 2, 3, 5, 7, 9 and 11 nearest neighbours. For the decision trees, we just use the default without tuning. For the neural network, we also do not make any tuning. We just use a hidden layer with 100 nodes as default. For the naive Bayes, we vary the smoothing parameter with the values: $e^{-i}\ \forall i \in [0,9]$. The random forest is tuned using 100, 500 and 1000 trees. For the SVM we create 3 models in the tuning process using three kernels: polynomial, RBF and linear. We choose the best from these three as our best SVM. For the linear kernel we vary C in the range $10^i\ \forall i \in [-5,2]$, for the RBF we also vary gamma in the range $i/\#rows\ \forall i \in [1,6]$ and for the polynomial kernel we use the same values of C  used in the RBF kernel and polynomials of degree 1, 2 and 3.  

Decisions trees (Tree) and Neural Network were not tuned (the reason why we did not tune parameters for the decision trees and neural networks was that they could consume too much time and our time to finish this work was limited).

## Used datasets
{\bf heart} was taken from the UCI repository~(\url{https://archive.ics.uci.edu/ml/datasets/heart+Disease}) the processed cleveland. Each row corresponds to a patient with 13 annotated features. The target variable indicates if the patient has a heart disease or not.  

{\bf pendigits} is a dataset that contains a representation of handwritten digits with 16 features that was used in the work of Gonen an Alpaydin and is in the UCI repository~(\url{https://archive.ics.uci.edu/ml/datasets/Pen-Based+Recognition+of+Handwritten+Digits}).

The {\bf adult} dataset was also taken from the UCI repository~(\url{https://archive.ics.uci.edu/ml/datasets/Adult}). This contains demographic data where each line is a person and the
classification task is to determine if that person gains over $50,000\$$ or not.

The {\bf mushrooms} dataset was taken from UCI~(\url{https://archive.ics.uci.edu/ml/datasets/mushroom}) and it has 22 features annotated from different types of mushrooms. The
target variable is if it is poisonous or not. This dataset is very easy to model and classification metrics are usually very high. We use it for comparison, but also to validate our MKL method.

The {\bf  fashion} dataset was created by using the training dataset from MNIST fashion on kaggle (\url{https://www.kaggle.com/zalando-research/fashionmnist}). This has images of clothes represented as 28 x 28 images. We created a binary classification problem by predicting if a piece of clothing covers the upper body (classes 0,2,3,4 and 6 in the original dataset) or not.

{\bf hiragana} is a dataset where we used the training set from Kuzushiji MNIST from kaggle~(\url{https://www.kaggle.com/anokas/kuzushiji}), which has
the objective of classifying the 49 Japanese handwritten hiragana characters. The characters are represented by 28 x 28 images. We transformed this in a binary classification problem and predict if the character belongs to the first 25 characters group or not.

{\bf gisette} is a dataset containing 28 x 28 images of handwritten numbers with highly confusable digits ’4’ and ’9’. New features were created as products of the images pixels to plunge the problem in a higher dimensional feature space (\url{https://archive.ics.uci.edu/ml/datasets/Gisette}).

The dataset {\bf volcanoes} contains a set of $110 \times 110$ images. The target variable indicates if the image is or is not a volcano. This dataset was taken from a kaggle challenge (\url{https://www.kaggle.com/fmena14/volcanoesvenus}). 

The {\bf volcanoes2} is the same as volcanos, but we added  $110 \times 110$ pixels resulting from the application of a sobel filter to the image,  which is a filter that has the objective of detecting edges in an image. That was done with the intention of showing the capacity of solving multimodal problems by MKL. Sobel was used just to make the experiment simple and because we did not have time to apply a sophisticated feature extraction or use actual multimodal data because we explored the datasets common in the literature what consumed too much time and we didn't have time to add additional datasets. 

Yet about \textbf{volcanoes2}, we performed an extra experiment with a different combination of kernels trying to explore multimodality capabilities of MKL. We used 20 kernels instead of 40 for the data from volcanoes and other 20 kernels using only the features we created. This method is called $MKL20\_20$ in the next chapter.

## Genetic algorithm 
 Aiming at reducing the combinatorial nature of MKL, we use a genetic algorithm to select weights for each kernel. The input to this algorithm is a set of kernels weights and the size of the population.
This approach was also used by Pinar~\cite{pinar2016explosive}. 

The fitness function we used was the mean of 5 holdouts of the F1 measure. For this work, the fitness calculation was performed for all individuals in parallel in order to reduce the execution time. Parents are randomly selected but individuals with higher fitness have a higher probability of producing descendants. Crossover is performed by choosing a random point on the genome and merging the first part of the first parent to the second part of the second parent to create a new individual that will become a member of the next generation. Mutation is performed by simply randomizing a weight from a number between 0 and 1. As we do not know anything about the effect of weights before doing this, we thought a random replacing of the value would be the best approach. For the mutation probability, we chose to have a mutation value that can change during the genetic algorithm. The mutation probability starts with value $0.01$ to each individual. If at any point, 3 consecutive generations do have the same best fitness then the mutation probability is increased to $0.05$. Otherwise, the probability returns to $0.01$. That approach was made to avoid our optimization process to be stuck at a local maximum by introducing diversity in the population. For all experiments, we used the same seed in order to have comparable results.
