# Car-price-prediction-project

#### A project predicting car price. This project is tested over a lot of ml models like catboost, xgboost, random forest, support vector classifier, etc.. Out of these models catboost performed very well giving mean_squared_error score around 10.5650 and R2_score of 0.979478 which is far better score than other models.

## Tech Stack:
* Front-End:HTML, CSS
* Back-End:Flask
* IDE:Jupyter Notebook, Visual Studio Code

## How to run this app:
* First create a virtual environment by using this command: conda create -n myenv python=3.6
* Activate the environment using the below command: conda activate myenv
* Then install all the packages by using the following command: pip install -r requirements.txt
* Now for the final step. Run the app
* python app.py

## Data Preprocessing:
* Categorical columns like Fuel_Type, Owner, Seller_Type, Transmission has been imputed using the mapping method.
* Numerical columns such as Present_Price, Kms_Driven has high skewness and they has been transformed using numpy.log1p for better use of the column.
* Tree based models doesn't need transformation(min_max_scalar/standard_scalar/robust_scalar), so I avoided it completely.

## Model Creation:
* Different types of models were tried like catboost, random forest, logistic regression, xgboost, support vector machines, knn, naive bayes.
* Out of these catboost, xgboost and lgbm were top 3.
* The conclusion were made using regression metrics. Mean_squared_error and R2_score.

## Model Deployment:
The model is deployed using Flask at Heroku server.

## Screenshots:
### Front page:
![front_page](https://user-images.githubusercontent.com/15306703/146343860-0d386dfe-29bd-4cf7-9a04-fcabb590ed6f.png)

### Prediction page:
![predictor](https://user-images.githubusercontent.com/15306703/146344663-afa24205-29f1-4327-820a-8515e946df67.png)

## Final prediction:
![final_output](https://user-images.githubusercontent.com/15306703/146349845-857aec14-feb0-460e-9beb-3087a21bf229.png)

