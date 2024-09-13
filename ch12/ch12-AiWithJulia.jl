################################################################################
# Code snippets of the book:
# A. Lobianco (2024) Julia Quick Syntax Reference - A Pocket Guide for Data
# Science Programming, Apress, Ed 2
# https://doi.org/[TODO_DOI]
# Licence: Open Domain
################################################################################

# Chapter 12 : AI with Julia

using Pkg
cd(@__DIR__)
Pkg.activate(".")
# If using a Julia version different than 1.10 or 1.11 please uncomment and run the following line (reproductibility guarantee will hower be lost). If your version of Julia can't find a suitable combination, try running `Pkg.up();Pkg.resolve()`
#Pkg.resolve()
using Random
Random.seed!(123)

# ##############################################################################
# ------------------------------------------------------------------------------
# THE BETAML TOOLKIT
using BetaML
mod = DecisionTreeEstimator()
xtrain = [1 10; 20 1; 21 2; 2 11]
ytrain = ["a","b","b","a"]
ŷ  = fit!(mod,xtrain,ytrain)
ŷ1 = predict(mod)  
ŷ2 = predict(mod,xtrain) 

using StableRNGs
mod = DecisionTreeEstimator(;rng=FIXEDRNG)
mod = DecisionTreeEstimator(;rng=copy(FIXEDRNG))
mod = DecisionTreeEstimator(;rng=StableRNG(FIXEDSEED))

reset!(mod)
info(mod)
hyperparameters(mod)
parameters(mod)
options(mod)
inverse_predict(mod,ŷ)
model_save("fitted_models.jld2"; mod)
mod = model_load("fitted_models.jld2","mod")

# ##############################################################################
# ------------------------------------------------------------------------------
# DATA PROCESSING 

# ------------------------------------------------------------------------------
# Encoding
x = ["blue","red","blue","green","red"]
m1 = OneHotEncoder()
m2 = OrdinalEncoder()
x_oh  = fit!(m1,x)
x_ord = fit!(m2,x) 

x = ["blue" "apple"; "red" "apple"; "blue" "orange"; "green" "orange"]
enc_models = [OneHotEncoder() for i in axes(x,2)]
x_oh = hcat([fit!(enc_models[i],x[:,i])  for i in axes(x,2)]...)

inverse_predict(m1,x_oh) == inverse_predict(m2,x_ord) == x 

# ------------------------------------------------------------------------------
# Scaling

x  = [5000,1000,2000,3000] 
m1 = Scaler() # eq. to Scaler(method=StandardScaler(scale=true,center=true))
m2 = Scaler(method=MinMaxScaler())
x_ss = fit!(m1,x)
x_mm = fit!(m2,x)

m1c = Scaler(method=StandardScaler(scale=true,center=true))


using Statistics, Test
@test isapprox(mean(x_ss),0,atol=1E-15)
@test var(x_ss,corrected=false) ≈ 1

@test inverse_predict(m1,x_ss) == inverse_predict(m2,x_mm) == x

x_sm = softmax(x, β=0.001) # another way to scale data

x    = [[5000,1000,2000,3000] ["a", "b", "c", "d"] [5,1,2,3] [0.5, 0.1, 0.2, 0.3]]
m    = Scaler(skip=[2])
x_ss = fit!(m,x)

m1b = Scaler(method=StandardScaler(),skip=[2])

# ------------------------------------------------------------------------------
# Missing data Imputation

x = [2.0 missing 10; 20 40 100; ]
m1 = SimpleImputer()
m2 = SimpleImputer(norm=1)
x_full1 = fit!(m1,x)
x_full2 = fit!(m2,x)

x = [10 2.5; missing 20.5; 0.8 18; 0.4 22.8; 12 missing; 9 3.7];
m = GaussianMixtureImputer(n_classes=2,rng=StableRNG(FIXEDSEED))
x_full = fit!(m,x)

info(m)

x = [1.4 2.5 "a"; missing 20.5 "b"; 0.6 18 missing; 0.7 22.8 "b"; 0.4 missing "b"; 1.6 3.7 "a"]
m = RandomForestImputer(multiple_imputations=10,recursive_passages=3,rng=StableRNG(FIXEDSEED))
x_full_v = fit!(m,x) # returns a vector of imputed matrices
x_full   = [nonmissingtype(eltype(identity.(x[:,c]))) <: Number ? median([v[r,c] for v in x_full_v]) : mode([v[r,c] for v in x_full_v]) for r in axes(x,1), c in axes(x,2)]

x = [1.4 2.5 1; missing 20.5 2; 0.6 18 2; 0.7 22.8 2; 0.4 missing 2; 1.6 3.7 1; ]
m = GeneralImputer(estimator=NeuralNetworkEstimator(rng=StableRNG(FIXEDSEED)),fit_function = BetaML.fit!, predict_function=BetaML.predict,rng=StableRNG(FIXEDSEED))
x_full = fit!(m,x)

# --------------------------------------------------------------
# Dimensionality reduction

x = [0.12 0.31 0.29 3.21 0.21;
     0.22 0.61 0.58 6.43 0.42;
     0.51 1.47 1.46 16.12 0.99;
     0.35 0.93 0.91 10.04 0.71;
     0.44 1.21 1.18 13.54 0.85]

m_pca = PCAEncoder(encoded_size=1,rng=StableRNG(FIXEDSEED))
m_ae  = AutoEncoder(encoded_size=1,epochs=800,rng=StableRNG(FIXEDSEED))

xl_pca = fit!(m_pca,x)
xl_ae  = fit!(m_ae,x)

using StatsPlots
groupedbar([xl_pca xl_ae], label=["PCA" "AE"], title="1D Latent space")
savefig("12_pca_ae_latent_spaces.png")

info(m_pca)
info(m_ae)
x_reconstr  = inverse_predict(m_ae,xl_ae) 

# ------------------------------------------------------------------------------
# Data partitioning

x    = [0.1 0.1 3; 0.2 0.2 2; 0.3 0.3 1; 0.4 0.4 2]
xoh  = [0.1 0.1 0 0 1; 0.2 0.2 0 1 0; 0.3 0.3 1 0 0; 0.4 0.4 0 1 0]
y    = ["a", "b", "c", "d"] 
((x_train, x_test), (xoh_train, xoh_test), (y_train, y_test)) = partition([x,xoh,y], [0.8,0.2],rng=StableRNG(FIXEDSEED))

x_train
x_test
y_test


# ##############################################################################
# ------------------------------------------------------------------------------
# MODEL FITTING

Random.seed!(123)
# ------------------------------------------------------------------------------
# Perceptron like classifiers

x = [2.2 28.4; 3.5 15.4; 2.5 31.2; 4.3 23.1; 5.7 12.8; 0.8 18.9; 0.3 13.4]
ynum = x[:,1] .* 5 .- x[:,2] ./ 2 # linear relation
y = [(i .< -4) ? "a" : ( (i .< 4) ? "b" : "c") for i in ynum]
colmap     =  Dict("a" => :red, "b" => :green, "c" => :blue)
ycolvalues = [colmap[i] for i in y] 
scatter(x[:,1],x[:,2],color=ycolvalues, label=nothing)
plot!([([1, 2.2],[30, 15]); ([2.7, 4],[15,30])], colour=:grey, linestyle=:dashdotdot, label=nothing)

savefig("12_cat_data.png")
m1 = PerceptronClassifier(rng=StableRNG(FIXEDSEED))
m2 = KernelPerceptronClassifier(rng=StableRNG(FIXEDSEED))
m3 = PegasosClassifier(rng=StableRNG(FIXEDSEED))

ŷ1_prob = fit!(m1,x,y) 
ŷ2_prob = fit!(m2,x,y) 
ŷ3_prob = fit!(m3,x,y)

ŷ3 = mode(ŷ3_prob)
accuracy(y,ŷ1_prob)
accuracy(y,ŷ2_prob)
accuracy(y,ŷ3_prob)
accuracy(y,ŷ3)

hcat(y,ŷ3)
m3b   = PegasosClassifier(learning_rate_multiplicative=0.1, rng=StableRNG(FIXEDSEED))
acc3b = fit!(m3b,x,y) |> x -> accuracy(y,x)

m3c = PegasosClassifier(autotune=true, rng=StableRNG(FIXEDSEED))
acc3b = fit!(m3b,x,y) |> x -> accuracy(y,x)

parameters(m1)

# ------------------------------------------------------------------------------
# Tree-based models 

m1 = DecisionTreeEstimator(rng=StableRNG(FIXEDSEED))
m2 = RandomForestEstimator(rng=StableRNG(FIXEDSEED))
y1hat = fit!(m1,x,y) 
y2hat = fit!(m2,x,y)
m1err = accuracy(y,y1hat) # 1.0
m2err = accuracy(y,y2hat) # 1.0

# ------------------------------------------------------------------------------
# Neurl-networks models 
Random.seed!(123)
nn1  = NeuralNetworkEstimator(rng=StableRNG(FIXEDSEED)) # Default NN model, usable for both regression and classification
nn2  = NeuralNetworkEstimator(layers=[DenseLayer(6,15,f=relu,rng=StableRNG(FIXEDSEED)),DenseLayer(15,1,f=relu,rng=StableRNG(FIXEDSEED))],loss=squared_cost,rng=StableRNG(FIXEDSEED)) # regression in R+ from a 6 dims data
nn3  = NeuralNetworkEstimator(layers=[DenseLayer(6,10,rng=StableRNG(FIXEDSEED)),DenseLayer(10,3,rng=StableRNG(FIXEDSEED)),VectorFunctionLayer(3,rng=StableRNG(FIXEDSEED))],loss=squared_cost,rng=StableRNG(FIXEDSEED)) # classification in R+ from a 6 dims data

nn = NeuralNetworkEstimator(opt_alg=SGD(η = t -> 1/(1+t), λ=2),rng=StableRNG(FIXEDSEED))
nn = NeuralNetworkEstimator(batch_size=32,rng=StableRNG(FIXEDSEED))
nn = NeuralNetworkEstimator(epochs=100,rng=StableRNG(FIXEDSEED))

(N,D) = (1000,6)
x     = rand(StableRNG(FIXEDSEED),N,D)
y     = abs.([10*r[1]-r[2]+0.1*r[3]*r[1] + sqrt(r[6]*10) for r in eachrow(x) ])
ysort = sort(y);
ycat  = [(i < ysort[Int(round(N/3))]) ?  "c" :  ( (i < ysort[Int(round(2*N/3))]) ? "a" : "b")  for i in y]
ohm   = OneHotEncoder();
yoh   = fit!(ohm,ycat)

nn1a = NeuralNetworkEstimator(rng=StableRNG(FIXEDSEED))
nn1b = NeuralNetworkEstimator(rng=StableRNG(FIXEDSEED))

y1ahat = fit!(nn1a,x,y) 
y1bhat = fit!(nn1b,x,yoh) 
y2hat = fit!(nn2,x,y)
y3hat = fit!(nn3,x,yoh)

m1_rme    = relative_mean_error(y,y1ahat)
m1_accerr = accuracy(ycat,inverse_predict(ohm,y1bhat))
m2_rme    = relative_mean_error(y,y2hat)
m3_accerr = accuracy(ycat,inverse_predict(ohm,y3hat))

# ------------------------------------------------------------------------------
# Clustering

Random.seed!(123)
import Distributions: MixtureModel, MvNormal
# Mixture of 3 2D Normals... 
data_gen_model = MixtureModel(MvNormal[
   MvNormal([2.5,0.0],[2 -0.8; -0.8 3]),
   MvNormal([-5,-2],[0.8 0.4; 0.4 1.2]),
   MvNormal([-0.5,5.2],[2 2.5; 2.5 5])], [0.2, 0.5, 0.3])

data = rand(StableRNG(FIXEDSEED),data_gen_model,1000)'
scatter(data[:,1], data[:,2], legend=nothing, xlabel="Feature 1", ylabel="Feature 2")
savefig("12_clustered_data.png")
savefig("12_clustered_data.svg")

kmeans_mod   = KMeansClusterer(n_classes=3,initialisation_strategy="grid",rng=StableRNG(FIXEDSEED))
kmedoids_mod = KMedoidsClusterer(n_classes=3,initialisation_strategy="grid",rng=StableRNG(FIXEDSEED))
gmm_mod      = GaussianMixtureClusterer(n_classes=3,mixtures=FullGaussian,rng=StableRNG(FIXEDSEED))

classes = fit!(kmeans_mod,data)
colmap  = ["red", "green", "blue"]
colors  = [colmap[c] for c in classes]
scatter(data[:,1], data[:,2],color=colors, legend=nothing, title="KMeans assignments")
savefig("12_kmeans_assignments.png")

classes = fit!(kmedoids_mod,data)
colors  = [colmap[c] for c in classes]
scatter(data[:,1], data[:,2],color=colors, legend=nothing, title="KMedoids assignments")
savefig("12_kmedoids_assignments.png")

probs   = fit!(gmm_mod,data)
classes = BetaML.mode(probs,rng=StableRNG(FIXEDSEED))
colors  = [RGB(r...) for r in eachrow(probs)];
scatter(data[:,1], data[:,2],color=colors, legend=nothing, title="GMM assignments")
savefig("12_gmm_assignments.png")


# ##############################################################################
# ------------------------------------------------------------------------------
# MODEL EVALUATION, INTERPRETATION and HYPERPARAMETER TUNING


# ------------------------------------------------------------------------------
# Regression models

y  = [1.3, 16.8, 3.5]
ŷ  = [0.9, 17.0, 3.2]

l1_distance(y,ŷ)
l2_distance(y,ŷ)
cosine_distance(y,ŷ)
mse(y,ŷ)
relative_mean_error(y,ŷ)
sobol_index(y,ŷ)

yb  = 10 .* (y .- mean(y)) 
ŷb  = 10 .* (ŷ .- mean(y))
@test sobol_index(y,ŷ) ≈ sobol_index(yb,ŷb)

ŷ2 = [1.2, 18.9, 3.6]

relative_mean_error(y,ŷ)
relative_mean_error(y,ŷ2)
relative_mean_error(y,ŷ; normrec=true)
relative_mean_error(y,ŷ2; normrec=true)

y  = [1.3 20.1; 4.8 18.7; 3.5 23.2]
ŷ  = [0.9 20.8; 4.0 17.8; 3.2 22.8]
ŷ2 = [1.2 23.1; 4.9 16.8; 3.6 21.8]
relative_mean_error(y,ŷ)
relative_mean_error(y,ŷ2)
relative_mean_error(y,ŷ; normdim=true)
relative_mean_error(y,ŷ2; normdim=true)

# ------------------------------------------------------------------------------
# Classification models

y = ["green", "red", "red", "green", "blue", "green"]
ohm = OneHotEncoder()
oem = OrdinalEncoder()
yoh = fit!(ohm,y)
yoe = fit!(oem,y)
levels = parameters(ohm).categories
ŷprob = [0.8 0.2 0.0
         0.4 0.3 0.3
         0.2 0.5 0.3
         0.6 0.1 0.3
         0.1 0.1 0.8
         0.3 0.4 0.2]
ŷdict = [ Dict([levels[i] => r[i] for i in axes(ŷprob,2)])     for r in eachrow(ŷprob)]
ŷint = BetaML.mode(ŷprob,rng=StableRNG(FIXEDSEED))
ŷ    = BetaML.mode(ŷdict,rng=StableRNG(FIXEDSEED))

accuracy(y,ŷ)       # 0.667
accuracy(y,ŷdict)   # 0.667
accuracy(yoe,ŷprob) # 0.667
accuracy(yoe,ŷint)  # 0.667

# These combinations woudn't work:
# accuracy(yoh,ŷ)
# accuracy(yoh,ŷdict)
# accuracy(yoh,ŷprob)
# accuracy(yoh,ŷint)
# accuracy(y,ŷprob) 
# accuracy(y,ŷint)
# accuracy(yoe,ŷ)
# accuracy(yoe,ŷdict)  

avg_crossentropy = sum(crossentropy(yoh[r,:],ŷprob[r,:]) for r in axes(y,1)) / size(y,1)
avg_kl_div = sum(kl_divergence(yoh[r,:],ŷprob[r,:]) for r in axes(y,1)) / size(y,1)

cm = ConfusionMatrix(rng=StableRNG(FIXEDSEED))
fit!(cm,y,ŷ)
print(cm)

res = info(cm);
heatmap(string.(res["categories"]),string.(res["categories"]),res["normalised_scores"],seriescolor=cgrad([:white,:blue]),xlabel="Predicted",ylabel="Actual", title="Confusion Matrix (normalised scores)")

savefig("12_confusion_matrix.png")

fit!(ConfusionMatrix(rng=StableRNG(FIXEDSEED)),y,ŷdict)  
fit!(ConfusionMatrix(rng=StableRNG(FIXEDSEED)),yoe,ŷprob)
fit!(ConfusionMatrix(rng=StableRNG(FIXEDSEED)),yoe,ŷint) 

# ------------------------------------------------------------------------------
# Clustering models
Random.seed!(123)
classes_to_test = 2:5
sil_by_class = fill(-1.0,length(classes_to_test))
BIC_by_class = fill(Inf,length(classes_to_test))
AIC_by_class = fill(Inf,length(classes_to_test))
pd = pairwise(data,distance=l2_distance) # we compute the pairwise distances
for (i,cl) in enumerate(classes_to_test)
    m = GaussianMixtureClusterer(n_classes=cl,mixtures=FullGaussian, rng=StableRNG(FIXEDSEED))
    ŷ = fit!(m,data) |> mode
    s = mean(silhouette(pd,ŷ))
    bic = info(m)["BIC"]
    aic = info(m)["AIC"]
    sil_by_class[i] = s
    BIC_by_class[i] = bic
end

bar(classes_to_test, sil_by_class, c=:white, xlabel=" number of classes", title="Cluster quality metrics by number of classes",label="Sil score                   ")
plot!(twinx(), classes_to_test, BIC_by_class, label="BIC")
savefig("12_cluster_scores.png")


# ------------------------------------------------------------------------------
# K-folds cross validation
Random.seed!(123)
(N,D) = (1000,6)
x     = rand(StableRNG(FIXEDSEED),N,D)
y     = abs.([10*r[1]-r[2]+0.1*r[3]*r[1] + sqrt(r[6]*10) for r in eachrow(x) ])
((xtrain,xtest),(ytrain,ytest)) = partition([x,y],[0.8,0.2],rng=StableRNG(FIXEDSEED))
sampler = KFold(nsplits=3;nrepeats=2,rng=StableRNG(FIXEDSEED))
(μ,σ) = cross_validation([xtrain,ytrain],sampler) do train_data,val_data, rng
    (xtrain,ytrain) = train_data; (xval,yval) = val_data
    model           = RandomForestEstimator(n_trees=30, rng=rng)            
    fit!(model,xtrain,ytrain)
    ŷval            = predict(model,xval)
    ϵ               = relative_mean_error(yval,ŷval)
    return ϵ
end

model  = RandomForestEstimator(n_trees=30,rng=StableRNG(FIXEDSEED))            
ŷtrain =  fit!(model,xtrain,ytrain)
ŷtest  = predict(model,xtest)
ϵ      = relative_mean_error(ytest,ŷtest)

# ------------------------------------------------------------------------------
# Autotune

tuning_method = SuccessiveHalvingSearch(
                   hpranges     = Dict("max_depth" =>[5,10,nothing], "min_gain"=>[0.0, 0.1, 0.5], "min_records"=>[2,3,5],"max_features"=>[nothing,5,10,30], "n_trees"=>[10,20,30]),
                   loss         = l2loss_by_cv,
                   res_shares   = [0.05, 0.2, 0.3],
                   multithreads = true,
                )
m = RandomForestEstimator(autotune=true, tunemethod=tuning_method,rng=StableRNG(FIXEDSEED))            
ŷtrain = fit!(m,xtrain,ytrain)
ŷtest  = predict(model,xtest)
ϵ      = relative_mean_error(ytest,ŷtest)

# ------------------------------------------------------------------------------
# Feature importance

Random.seed!(123)
x     = rand(StableRNG(FIXEDSEED),1000,6)
y     = abs.([10*r[1]-r[2]+0.1*r[3]*r[1] + sqrt(r[6]*10) for r in eachrow(x) ])
fr   = FeatureRanker(model=NeuralNetworkEstimator(rng=StableRNG(FIXEDSEED),verbosity=NONE),nsplits=3,nrepeats=2,metric="mda",rng=StableRNG(FIXEDSEED))

rank = fit!(fr,x,y)
loss_by_col        = info(fr)["loss_by_col"]
sobol_by_col       = info(fr)["sobol_by_col"]
loss_by_col_sd     = info(fr)["loss_by_col_sd"]
sobol_by_col_sd    = info(fr)["sobol_by_col_sd"]
loss_fullmodel     = info(fr)["loss_all_cols"]
loss_fullmodel_sd  = info(fr)["loss_all_cols_sd"]
ntrials_per_metric = info(fr)["ntrials_per_metric"]


import Distributions: Normal
var_names=["x1","x2","x3","x4","x5","x6"]
bar(var_names[sortperm(loss_by_col)], loss_by_col[sortperm(loss_by_col)],label="Loss by varᶜ", permute=(:x,:y), yerror=quantile(Normal(0,1),0.975) .* (loss_by_col_sd[sortperm(loss_by_col)]./sqrt(ntrials_per_metric)), yrange=[0,0.25],legend=:bottomright, title="Feature importance")
vline!([loss_fullmodel], label="Loss with all vars",linewidth=2)
vline!([loss_fullmodel-quantile(Normal(0,1),0.975) * loss_fullmodel_sd/sqrt(ntrials_per_metric),
        loss_fullmodel+quantile(Normal(0,1),0.975) * loss_fullmodel_sd/sqrt(ntrials_per_metric),
], label=nothing,linecolor=:black,linestyle=:dot,linewidth=1)

savefig("12_variable_importance.png")
