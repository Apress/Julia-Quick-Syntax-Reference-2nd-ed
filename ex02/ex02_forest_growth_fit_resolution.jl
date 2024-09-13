################################################################################
# forestGrowthFitting Problem
#
# The objective of this problem is to use the so-called "raw data" that the National Forest Inventory of France make available at the level of the individual inventoried trees and plots to fit a generic growth model of the forest stands in terms of volumes with respect to the age of the trees.
#

# ------------------------------------------------------------------------------
# 1) Setting up the environment...

# Start by setting the working directory to the directory of this file and activate it. If you have the `Manifest.toml` and `Project.toml` files in the directory, run `instantiate()`, otherwise manually add the packages Pipe, HTTP, CSV, DataFrames, LsqFit, StatsPlots.
# Also, for reproducibility, fix the random seed.


cd(@__DIR__)         
using Pkg             
Pkg.activate(".")   
# If using a Julia version different than 1.10 or 1.11 please uncomment and run the following line (reproductibility guarantee will hower be lost). If your version of Julia can't find a suitable combination, try running `Pkg.up();Pkg.resolve()`
# Pkg.resolve()   
Pkg.instantiate() 
using Random
Random.seed!(123)

# ------------------------------------------------------------------------------
# 2) Load the packages 

# Load the packages Pipe, HTTP, CSV, DataFrames, LsqFit, StatsPlots

using Pipe, HTTP, CSV, DataFrames, LsqFit, StatsPlots

# ------------------------------------------------------------------------------
# 3) Load the data

# Import data from remote..
ltURL     = "https://github.com/sylvaticus/IntroSPMLJuliaCourse/blob/main/lessonsMaterial/02_JULIA2/forestExercise/data/arbres_foret_2012.csv?raw=true"
dtURL     = "https://github.com/sylvaticus/IntroSPMLJuliaCourse/blob/main/lessonsMaterial/02_JULIA2/forestExercise/data/arbres_morts_foret_2012.csv?raw=true"
pointsURL = "https://github.com/sylvaticus/IntroSPMLJuliaCourse/blob/main/lessonsMaterial/02_JULIA2/forestExercise/data/placettes_foret_2012.csv?raw=true"
docURL    = "https://github.com/sylvaticus/IntroSPMLJuliaCourse/blob/main/lessonsMaterial/02_JULIA2/forestExercise/data/documentation_2012.csv?raw=true"

lt     = @pipe HTTP.get(ltURL).body |> CSV.File(_) |> DataFrame
dt     = @pipe HTTP.get(dtURL).body |> CSV.File(_) |> DataFrame
points = @pipe HTTP.get(pointsURL).body |> CSV.File(_) |> DataFrame
doc    = @pipe HTTP.get(docURL).body |> CSV.File(_) |> DataFrame


# ------------------------------------------------------------------------------
# 4) Filter out unused information 
lt     = lt[:,["idp","c13","v"]]
dt     = dt[:,["idp","c13","v"]]
trees  = vcat(lt,dt)
points = points[:,["idp","esspre","cac"]]

# ------------------------------------------------------------------------------
# 5) Compute the timber volumes per hectare
"""
    vHaContribution(volume,circonference)

Return the contribution in terms of m³/ha of the tree.

The French inventory system is based on a concentric sample method: small trees are sampled on a small area (6 metres radius), intermediate trees on a concentric area of 9 metres and only large trees (with a circonference larger than 117.5 cm) are sampled on a concentric area of 15 metres of radius.
This function normalise the contribution of each tree to m³/ha.
"""
function vHaContribution(v,c13)
    if c13 < 70.5
        return v/(6^2*pi/(100*100))
    elseif c13 < 117.5
        return v/(9^2*pi/(100*100))
    else 
        return v/(15^2*pi/(100*100))
    end
end

trees.vHa  = vHaContribution.(trees.v,trees.c13)

# ------------------------------------------------------------------------------
# 6) Aggregate trees data
pointsVols = combine(groupby(trees,["idp"]) ,  "vHa" => sum => "vHa", nrow => "ntrees")

# ------------------------------------------------------------------------------
# 7) Join datasets
points     = innerjoin(points,pointsVols,on="idp")

# ------------------------------------------------------------------------------
# 8) Filter data
filter_nTrees           = points.ntrees .> 5 # we skip points with few trees 
filter_IHaveAgeClass    = .! in.(points.cac,Ref(["AA","NR"]))
filter_IHaveMainSpecies = .! ismissing.(points.esspre) 
filter_overall          = filter_nTrees .&& filter_IHaveAgeClass .&& filter_IHaveMainSpecies
points                  = points[filter_overall,:] 

# ------------------------------------------------------------------------------
# 9) Compute the age class
points.cac              = (parse.(Int64,points.cac) .- 1 ) .* 5 .+ 2.5

# ------------------------------------------------------------------------------
# 10) Define the model to fit
logisticModel(age,parameters) = parameters[1]/(1+exp(-parameters[2] * (age-parameters[3]) ))
logisticModelVec(age,parameters) = logisticModel.(age,Ref(parameters))

# ------------------------------------------------------------------------------
# 11) Set the initial values for the parameters to fit
initialParameters = [1000,0.05,50] #max growth; growth rate, mid age

# ------------------------------------------------------------------------------
# 12) Fit the model

fitobject         = curve_fit(logisticModelVec, points.cac, points.vHa, initialParameters)
fitparams         = fitobject.param

# ------------------------------------------------------------------------------
# 13) Compute the errors
fitobject.resid

sigma            = stderror(fitobject)
confidence_inter = confidence_interval(fitobject, 0.1) # 10% significance level

# ------------------------------------------------------------------------------
# 14) Plot fitted model
x = 0:maximum(points.cac)*1.5
plot(x->logisticModel(x,fitparams),0,maximum(x), label= "Fitted vols", legend=:topleft)

# ------------------------------------------------------------------------------
# 15) Add the observations to the plot
plot!(points.cac, points.vHa, seriestype=:scatter, label = "Obs vHa")


# ------------------------------------------------------------------------------
# 16) Differentiate the model per tree specie 
speciesCount = combine(groupby(points, :esspre), nrow => :count)
sort!(speciesCount,"count",rev=true)

spLabel        = doc[doc.donnee .== "ESSPRE",:]
spLabel.spCode = parse.(Int64,spLabel.code)
speciesCount = leftjoin(speciesCount,spLabel, on="esspre" => "spCode")


# plot the 5 main species separately
for (i,sp) in enumerate(speciesCount[1:5,"esspre"])
    local fitobject, fitparams, x
    spLabel    = speciesCount[i,"libelle"]
    pointsSp   = points[points.esspre .== sp, : ]
    fitobject  = curve_fit(logisticModelVec, pointsSp.cac, pointsSp.vHa, initialParameters)
    fitparams  = fitobject.param
    x = 0:maximum(points.cac)*1.5
    if i == 1
        myplot = plot(x->logisticModel(x,fitparams),0,maximum(x), label= spLabel, legend=:topleft)
    else
        myplot = plot!(x->logisticModel(x,fitparams),0,maximum(x), label= spLabel, legend=:topleft)
    end
    display(myplot)
end


