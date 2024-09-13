# In this exercise we will try to predict the average housing value in suburbs of Boston given some characteristics of the suburb.

# These are the detailed attributes of the dataset:
#   1. CRIM      per capita crime rate by town
#   2. ZN        proportion of residential land zoned for lots over 25,000 sq.ft.
#   3. INDUS     proportion of non-retail business acres per town
#   4. CHAS      Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)
#   5. NOX       nitric oxides concentration (parts per 10 million)
#   6. RM        average number of rooms per dwelling
#   7. AGE       proportion of owner-occupied units built prior to 1940
#   8. DIS       weighted distances to five Boston employment centres
#   9. RAD       index of accessibility to radial highways
#   10. TAX      full-value property-tax rate per $10,000
#   11. PTRATIO  pupil-teacher ratio by town
#   12. B        1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town
#   13. LSTAT    % lower status of the population
#   14. MEDV     Median value of owner-occupied homes in $1000's

# Further information concerning this dataset can be found on [this file](https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.names)

# Our prediction concern the median value (column 14 of the dataset)

# 1) Start by setting the working directory to the directory of this file and activate it. If you have the provided `Manifest.toml` file in the directory, just run `Pkg.instantiate()`, otherwise manually add the packages Pipe, HTTP, CSV, DataFrames, Plots and BetaML.
# Also, seed the random seed with the integer `123`.

# [...] Write your code here

# 2) Load the packages/modules Pipe, HTTP, CSV, DataFrames, Plots, BetaML and import the functions `quantile` and the type `Normal` from the Distributions package

# [...] Write your code here

# 3) Load from internet or from local file the input data into a DataFrame or a Matrix.
# You will need the CSV options `header=false` and `ignorerepeated=true`

dataURL = "https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data"

# [...] Write your code here

# 4) Now create the X matrix of features (columns 1 to 13th). Make shure you have a 506×13 matrix (and not a DataFrame).

# [...] Write your code here

# 5) Similarly, define Y to be the 14th column of data

# [...] Write your code here

# 6) Partition the data in (`xtrain,`xtest`) and (`ytrain`,`ytest`) keeping 80% of the data for training and reserving 20% for testing. Keep the default option to shuffle the data, as the input data isn't.

# [...] Write your code here

# 7) Define a `NeuralNetworkEstimator` model with the following characteristics:
#   - 3 dense layers with respectively 13, 20 and 1 nodes and activation function relu
#   - cost function `squared_cost` 
#   - training options: 400 epochs and 6 records to be used on each batch

# [...] Write your code here

# 8) Train your model using `ytrain` and a scaled version of `xtrain` (where all columns have zero mean and 1 standard deviation)

# [...] Write your code here

# 9) Predict the training labels ŷtrain and the test labels ŷtest. Recall you did the training on the scaled features!

# [...] Write your code here

# 10) Compute the train and test relative mean error using the function `relative_mean_error`

# [...] Write your code here

# 11) Run the following commands to plots the average loss per epoch and the true vs estimated test values 
plot(info(mynn)["loss_per_epoch"])
scatter(ytest,ŷtest,xlabel="true values", ylabel="estimated values", legend=nothing)

# 12) Find the optimal model testing the following hyperparameters:
# - size of the middle layer
# - batch size
# - number of epochs

inner_layer_size_range = # [...] Write your code here
epoches_range          = # [...] Write your code here
bachsize_range         = # [...] Write your code here

# If you are using the BetaML autotune mechanism, use the following line to build the range of the `layers` parameters to be used in the `hpranges` dictionary starting for a range defined in terms of size of the inner layer:
layers_range = [[DenseLayer(13,i,f=relu), DenseLayer(i,i,f=relu), DenseLayer(i,1,f=relu)] for i in inner_layer_size_range]

# [...] Write your code here

# 13) Study the variable imporance of the neural network model.

# In the following vector you have the variable names of the features
var_names = [
  "CRIM",    # per capita crime rate by town
  "ZN",      # proportion of residential land zoned for lots over 25,000 sq.ft.
  "INDUS",   # proportion of non-retail business acres per town
  "CHAS",    # Charles River dummy variable  (= 1 if tract bounds river; 0 otherwise)
  "NOX",     # nitric oxides concentration (parts per 10 million)
  "RM",      # average number of rooms per dwelling
  "AGE",     # proportion of owner-occupied units built prior to 1940
  "DIS",     # weighted distances to five Boston employment centres
  "RAD",     # index of accessibility to radial highways
  "TAX",     # full-value property-tax rate per $10,000
  "PTRATIO", # pupil-teacher ratio by town
  "B",       # 1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town
  "LSTAT",   # % lower status of the population
]

# Which are the most important variables to correctly predict the house value ?

# [...] Write your code here