# 🚑📝 Linear Regression Health Costs Calculator
A linear regression model built with TensorFlow, Keras, scikit-learn, and pandas that predicts healthcare expenses based on demographic and lifestyle features. This model is designed to predict healthcare costs accurately within $3500 (i.e. MAE of less than 3500).

> 🧠 **This challenge was provided by [freeCodeCamp’s Machine Learning with Python course](https://www.freecodecamp.org/learn/machine-learning-with-python/).**


## 🛠 What I Did

- Converted categorical data to numbers using `pd.Categorical().codes`
- Used 80% of the data as the `train_dataset` and 20% of the data as the `test_dataset` with `train_test_split` from scikit-learn
- Created a Sequential model with 3 dense layers

## 🤔 What I Learned

- **Applying a normalization layer**: Standardizing input features helped the model train more efficiently and with better accuracy
- **Mapping unique categories to numerical values**: There were several ways to go about this. I could've also used `pd.factorize()` to automatically assign a unique integer to each unique category in the column, or manually defined the conversion using `replace()`, etc. In the end, I just wanted to try out whatever I hadn't used before, which was `pd.Categorical()`.
- **Splitting the dataset**: Similarly, there were many different ways to split the data 80/20. My first thought was to shuffle the dataset and use `dataset.sample(frac=0.2)` to take 20% of the data for `test_dataset`,  followed by `dataset.drop()` to remove those lines for `train_dataset`, and then `pop` off the 'expenses' column. However, after some more research, I found the method `test_train_split` which automatically randomizes the 80/20 split and can get rid of 'expenses'. I went with this method in order to simplify the code.


## 🚀 Future Improvements

- **Hyperparameter Tuning**: Adjusting different hyperparameters such as the learning rate, batch size, etc., could lead to better performance.





