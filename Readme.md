## Indoor-Locating ( The Practice of UjiIndoorLoc Dataset )
[![Packagist](https://img.shields.io/badge/Tensorflow-1.2.1-blue.svg)]()
[![Packagist](https://img.shields.io/badge/Tensorlayer-1.5.4-blue.svg)]()
[![Packagist](https://img.shields.io/badge/Scikit_learn-0.17-blue.svg)]()</br>   

Abstract    
---
This repository try to build the model that can solve the indoor location problem. The dataset are downloaded from [Kaggle](https://www.kaggle.com/giantuji/UjiIndoorLoc). SVM, random forest, gradient boosting tree and DNN are adopted in this program. For the comparision, the **random forest** is the fastest and most powerful model for this problem. 

Result
---

|Model type|SVM|Random forest|Gradient boosting tree|DNN|
|---|---|---|---|---|
|Error value|58786.35|22390.67|45143.24|27990.10|

Train by Yourself
---
You should download the data from [Kaggle](https://www.kaggle.com/giantuji/UjiIndoorLoc). Next, depress the file and put them in the same folder.     

```
sunner@sunner-pc:~/loc/$ ls
abstract_model.py  data_helper.py  dl_model.py  main.py  ml_model.py  Readme.md  TrainingData.csv  ValidationData.csv
```

Notice
---
The code might not be accepted by Kaggle submission mechanism since it use some deep learning model. (For example, tensorlayer) What's more, the version of python is 2.7 while the 3+ version are accepted on the platform.    