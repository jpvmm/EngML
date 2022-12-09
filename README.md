# MLAnalytics
This repository has the idea to build a first cut of a MLOps pipeline end to end in VertexAI.

The overview architecture of the project is this:

![alt text](images/arch.png "Overview")

## Problem Statement:

As toy a example, we are trying to solve the problem of default payment risk prediction.
The dataset for the problem can be found [here.](https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients)

To do that I will follow the these steps:

1. Data analysis to understand the problem and the dataset;
2. Baseline machine learning model for classification;
3. More complex machine learning model for classification;
4. Development of Vertex Pipeline for training and deployment;
5. BigQuery Table creation for model metada storage;
6. Devolopment of custom prediction image to deal with preprocessing in prediction time;
7. Setup of Vertex Model monitoring for data and concept drifts.