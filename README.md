# 38-615-Machine-Learning
CMU 38-615 class repo

## HW1: Exploratory Data Analysis (EDA)

We have provided you with an obfuscated scientific dataset. Each row is an observation. The first column, titled “experimental_proprty” is your target property of interest. All other columns encode features. There are no observation IDs.


## HW2: Clustering

1. Load the data. (We provide you with CSV file in Canvas)
2. Perform k-means clustering. Determine the optimal number of clusters using e.g. elbow
method.
3. Perform clustering with different clustering method as implemented in Scikit Learn.
4. Now, try clustering with another distance metric (e.g. Cosine, Jaccard, etc). Hint: Think
whether the default distance metric is appropriate for your data or not.
5. Visualize results using the dimensionality reduction (UMAP or tSNE) technique with
respect to the cluster labels.
6. Compare clustering results. Try to rationalize observed commonalities or differences
with respect to clustering methods and distance metrics used.


## HW3: Wide data and linear models (https://www.kaggle.com/competitions/f24-38615-hw3)

You are provided with a dataset for 554 patients, 80% (444 patients) of the dataset was selected to be the training set, and 20% (110 patients) as the test set. Features and labels of the training set can be found in train_X.csv, train_y.csv respectively. Features of the test set can be found in test_X.csv while labels are hidden.
Your task is to predict the disease type (phenotype) from transcriptomics data. Disease: UCEC (uterine corpus endometrial carcinoma). Labels (0/1) are encoding tumor grade “II-” vs. “III+”


## HW4: Classification of Green Fluorescent Protein (https://www.kaggle.com/competitions/f24-38615-hw4)

In this work you will predict the brightness level binarized for classification between high brightness (class 1) and low brightness (class 0) for a set of mutants of Green Fluorescent Protein. Please explore descriptors/featurizations for amino acid sequences. Overfitting is prevented by using a public and private leaderboard.


## HW5: Band gap prediction of inorganic materials (https://www.kaggle.com/competitions/f24-38615-hw5)

You can start with basic models, and then try your best to optimize your predictions by using more sophisticated models, feature engineering, and fine-tuning the hyperparameters. Your grade for this homework will mainly depend on your Kaggle score(s). There is no specific requirement for a written summary for this homework, but please leave some necessary comments in your submission to help the grader understand your workflow.
