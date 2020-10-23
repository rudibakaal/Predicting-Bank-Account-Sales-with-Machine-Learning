# Predicting Bank Account Sales Outcomes with Machine Learning

## Motivation

This dataset is related to direct marketing campaigns (phone calls) of a Portuguese banking institution. The classification goal is to predict if the client will subscribe a term deposit (variable y)[1].

On account of their multivariate nature[2], one hot encoding of categorical features and the binary label was implemented with pandas cat.codes and sklearn LabelEncoder classes, respectively. All features were also standardised via sklearn's StandardScaler class. 

## Neural Network Topology and Results Summary

The binary-crossentropy loss function was leveraged along with the Adagrad optimizer for this classification problem.


![model](https://user-images.githubusercontent.com/48378196/96961401-4be81500-1550-11eb-9cd2-4e0f682c3b56.png)

The binary classifier accurately predicts ~89% of outbound call sales outcomes 

![results](https://user-images.githubusercontent.com/48378196/96961083-aa60c380-154f-11eb-90d8-453a87595713.png)



## License
[MIT](https://choosealicense.com/licenses/mit/) 

## References
[1] [Moro et al., 2014] S. Moro, P. Cortez and P. Rita. A Data-Driven Approach to Predict the Success of Bank Telemarketing. Decision Support Systems, Elsevier, 62:22-31, June 2014
[2] https://archive.ics.uci.edu/ml/datasets/Bank+Marketing
 
