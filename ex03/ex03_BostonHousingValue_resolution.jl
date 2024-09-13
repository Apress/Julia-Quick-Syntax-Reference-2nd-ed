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

cd(@__DIR__)    
using Pkg             
Pkg.activate(".")   
# If using a Julia version different than 1.10 or 1.11 please uncomment and run the following line (reproductibility guarantee will hower be lost). If your version of Julia can't find a suitable combination, try running `Pkg.up();Pkg.resolve()`
# Pkg.resolve()   
Pkg.instantiate()
using Random
Random.seed!(123)

# 2) Load the packages/modules Pipe, HTTP, CSV, DataFrames, Plots, BetaML and import the functions `quantile` and the type `Normal` from the Distributions package
using Pipe, HTTP, CSV, DataFrames, Plots, BetaML
import Distributions: quantile, Normal

# 3) Load from internet or from local file the input data into a DataFrame or a Matrix.
# You will need the CSV options `header=false` and `ignorerepeated=true`
dataURL = "https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data"
data    = @pipe HTTP.get(dataURL).body |> CSV.File(_, delim=' ', header=false, ignorerepeated=true) |> DataFrame


# 4) Now create the X matrix of features (columns 1 to 13th). Make shure you have a 506×13 matrix (and not a DataFrame).
X = Matrix(data[:,1:13])

# 5) Similarly, define Y to be the 14th column of data
Y = data[:,14] # Median value of owner-occupied homes in $1000's


# 6) Partition the data in (`xtrain,`xtest`) and (`ytrain`,`ytest`) keeping 80% of the data for training and reserving 20% for testing. Keep the default option to shuffle the data, as the input data isn't.
((xtrain,xtest),(ytrain,ytest)) = partition([X,Y],[0.8,0.2])


# 7) Define a `NeuralNetworkEstimator` model with the following characteristics:
#   - 3 dense layers with respectively 13, 20 and 1 nodes and activation function relu
#   - cost function `squared_cost` 
#   - training options: 400 epochs and 6 records to be used on each batch
l1 = DenseLayer(13,20,f=relu)
l2 = DenseLayer(20,20,f=relu)
l3 = DenseLayer(20,1,f=relu)
mynn= NeuralNetworkEstimator(layers=[l1,l2,l3],loss=squared_cost,batch_size=6,epochs=400)

# 8) Train your model using `ytrain` and a scaled version of `xtrain` (where all columns have zero mean and 1 standard deviation)
fit!(mynn,fit!(Scaler(),xtrain),ytrain)

# 9) Predict the training labels ŷtrain and the test labels ŷtest. Recall you did the training on the scaled features!
ŷtrain   = predict(mynn, fit!(Scaler(),xtrain)) 
ŷtest    = predict(mynn, fit!(Scaler(),xtest))  


# 10) Compute the train and test relative mean error using the function `relative_mean_error`
trainRME = relative_mean_error(ytrain,ŷtrain) 
testRME  = relative_mean_error(ytest,ŷtest)


# 11) Run the following commands to plots the average loss per epoch and the true vs estimated test values 
plot(info(mynn)["loss_per_epoch"])
scatter(ytest,ŷtest,xlabel="true values", ylabel="estimated values", legend=nothing)

# 12) Find the optimal model testing the following hyperparameters:
# - size of the middle layer
# - batch size
# - number of epochs
inner_layer_size_range = 18:2:22 # 16:2:26
epoches_range          = 300:100:500 # 200:100:600
bachsize_range         = [4,6,8] # 4:2:10

# If you are using the BetaML autotune mechanism, use the following line to build the range of the `layers` parameters to be used in the `hpranges` dictionary starting for a range defined in terms of size of the inner layer:
layers_range = [[DenseLayer(13,i,f=relu), DenseLayer(i,i,f=relu), DenseLayer(i,1,f=relu)] for i in inner_layer_size_range]

res_shares             = [0.5, 0.7, 0.9]
tuning_method = SuccessiveHalvingSearch(
                   hpranges     = Dict("layers" => layers_range, "batch_size"=>bachsize_range, "epochs"=>epoches_range),
                   loss         = l2loss_by_cv,
                   res_shares   = res_shares,
                   multithreads = true
                )
m = NeuralNetworkEstimator(autotune=true, tunemethod=tuning_method)
fit!(m,fit!(Scaler(),xtrain),ytrain)
ŷtrain   = predict(m, fit!(Scaler(),xtrain)) 
ŷtest    = predict(m, fit!(Scaler(),xtest))  
trainRME = relative_mean_error(ytrain,ŷtrain) 
testRME  = relative_mean_error(ytest,ŷtest)

scatter(ytest,ŷtest,xlabel="true values", ylabel="estimated values", legend=nothing)

opt_epochs     = hyperparameters(m).epochs
opt_batch_size = hyperparameters(m).batch_size
opt_lsize      = size(hyperparameters(m).layers[1])[2][1]
opt_layers     = [DenseLayer(13,opt_lsize,f=relu), DenseLayer(opt_lsize,opt_lsize,f=relu),DenseLayer(opt_lsize,opt_lsize,f=relu), DenseLayer(opt_lsize,1,f=relu)] 

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
opt_layers_zero = [DenseLayer(13,opt_lsize,f=relu), DenseLayer(opt_lsize,opt_lsize,f=relu),DenseLayer(opt_lsize,opt_lsize,f=relu), DenseLayer(opt_lsize,1,f=relu)]
fr = FeatureRanker(model=NeuralNetworkEstimator(layers=opt_layers_zero,epochs=opt_epochs,batch_size=opt_batch_size, verbosity=NONE),nsplits=3,nrepeats=2,recursive=false)


fr = FeatureRanker(model=NeuralNetworkEstimator(verbosity=NONE),nsplits=3,nrepeats=2,recursive=false)


rank = fit!(fr,xtrain,ytrain)
loss_by_col        = info(fr)["loss_by_col"]
sobol_by_col       = info(fr)["sobol_by_col"]
loss_by_col_sd     = info(fr)["loss_by_col_sd"]
sobol_by_col_sd    = info(fr)["sobol_by_col_sd"]
loss_fullmodel     = info(fr)["loss_all_cols"]
loss_fullmodel_sd  = info(fr)["loss_all_cols_sd"]
ntrials_per_metric = info(fr)["ntrials_per_metric"]

bar(var_names[sortperm(loss_by_col)], loss_by_col[sortperm(loss_by_col)],label="Loss by var", permute=(:x,:y), yerror=quantile(Normal(1,0),0.975) .* (loss_by_col_sd[sortperm(loss_by_col)]./sqrt(ntrials_per_metric)), yrange=[0,0.9])
vline!([loss_fullmodel], label="Loss with all vars",linewidth=2)
vline!([loss_fullmodel-quantile(Normal(1,0),0.975) * loss_fullmodel_sd/sqrt(ntrials_per_metric),
        loss_fullmodel+quantile(Normal(1,0),0.975) * loss_fullmodel_sd/sqrt(ntrials_per_metric),
], label=nothing,linecolor=:black,linestyle=:dot,linewidth=1)
