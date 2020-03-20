# complete-pipeline-of-ML-with-industry-standard

Allpication logging class is used to generate log file for each pipeline like, extracting data, transforming, model selection.

Exceptions are handled in all the pipelines.

read write file calss is used to read the input file and generate the log

data transformation class is used to select the features and transform them and generate the log

Tune model class uses imblearn to select the best model as input dataset is highly imbalanced, normal ML algorithms performace was very bad. This class used BalancedRandomForestClassifier and EasyEnsembleClassifier from imblearn. Hyper tune both the models using RandomizedSearchCV and get the best parameters. fit both the models using these parameters and return the best performing model

This pipline also generate log file with hyper parameters of both the models.

Performance mattrix i have use as F2 score as in this particular problem statement Recall is very important but at the same time i can't igonre precision completely, else Recall alone can be used.


Fbeta = ((1 + beta^2) * Precision * Recall) / (beta^2 * Precision + Recall)

F2-Measure (beta=2.0): Less weight on precision, more weight on recall

F2 =  ((1 + 2^2) * Precision * Recall) / (2^2 * Precision + Recall)

