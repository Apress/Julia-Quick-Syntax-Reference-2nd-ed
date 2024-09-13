################################################################################
# forestGrowthFitting Problem
#
# The objective of this problem is to use the so-called "raw data" that the National Forest Inventory of France make available at the level of the individual inventoried trees and plots to fit a generic growth model of the forest stands in terms of volumes with respect to the age of the trees.


# ------------------------------------------------------------------------------
# 1) Setting up the environment...

# Start by setting the working directory to the directory of this file and activate it. If you have the `Manifest.toml` and `Project.toml` files in the directory, run `instantiate()`, otherwise manually add the packages Pipe, HTTP, CSV, DataFrames, LsqFit, StatsPlots.
# Also, for reproducibility, fix the random seed.


# [...] Write your code here

# ------------------------------------------------------------------------------
# 2) Load the packages 

# Load the packages Pipe, HTTP, CSV, DataFrames, LsqFit, StatsPlots

# [...] Write your code here

# ------------------------------------------------------------------------------
# 3) Load the data

# Load from internet the following datasets:

ltURL     = "https://bit.ly/apress_julia_alive_trees"
dtURL     = "https://bit.ly/apress_julia_dead_trees"
pointsURL = "https://bit.ly/apress_julia_inv_points"
docURL    = "https://bit.ly/apress_julia_inv_doc"
# You can make for each of the dataset a `@pipe` macro starting with `HTTP.get(URL).body`, continuing the pipe with `CSV.File(_)` and end the pipe with a DataFrame object.

# [...] Write your code here

# ------------------------------------------------------------------------------
# 4) Filter out unused information 

# These datasets have many variable we are not using in this exercise.
Out of all the variables, select only for the `lt` and `dt` dataframes the columns `idp` (pixel id), `c13` (circumference at 1.30 meters) and `v` (tree's volume). Then vertical concatenate the two dataset in an overall `trees` dataset.
For the `points` dataset, select only the variables `idp` (pixel id), `esspre` (code of the main forest species in the stand) and `cac` (age class).

# [...] Write your code here

# ------------------------------------------------------------------------------
# 5) Compute the timber volumes per hectare

# As the French inventory system is based on a concentric sample method (small trees are sampled on a small area (6 metres radius), intermediate trees on a concentric area of 9 metres and only large trees (with a circonference larger than 117.5 cm) are sampled on a concentric area of 15 metres of radius), define the following function to compute the contribution of each tree to the volume per hectare:

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

# Use the function above to compute `trees.vHa` based on `trees.v` and `trees.c13`.

# [...] Write your code here

# ------------------------------------------------------------------------------
# 6) Aggregate trees data

# Aggregate the `trees` dataframe by the `idp` column to retrieve the sum of `vHa` and the number of trees for each point, calling these two columns `vHa` and `ntrees`.

# [...] Write your code here


# ------------------------------------------------------------------------------
# 7) Join datasets

# Join the output of the previous step (the trees dataframe aggregated "by point") with the original points dataframe using the column `idp`.

# [...] Write your code here

# ------------------------------------------------------------------------------
# 8) Filter data

# Use boolean selection to apply the following filters:

filter_nTrees           = points.ntrees .> 5 # we skip points with few trees 
filter_IHaveAgeClass    = .! in.(points.cac,Ref(["AA","NR"]))
filter_IHaveMainSpecies = .! ismissing.(points.esspre) 
filter_overall          = filter_nTrees .&& filter_IHaveAgeClass .&& filter_IHaveMainSpecies

# [...] Write your code here

# ------------------------------------------------------------------------------
# 9) Compute the age class

# Run the following command to parse the age class (originally as a string indicating the 5-ages group) to an integer and compute the mid-range of the class in years. For example, class "02" will become 7.5 years.

points.cac              = (parse.(Int64,points.cac) .- 1 ) .* 5 .+ 2.5

# ------------------------------------------------------------------------------
# 10) Define the model to fit

# Define the following logistic model of the growth relation with respect to the age with 3 parameters and make its vectorised form:

logisticModel(age,parameters) = parameters[1]/(1+exp(-parameters[2] * (age-parameters[3]) ))
logisticModelVec(age,parameters) = # [...] Write your code here

# ------------------------------------------------------------------------------
# 11) Set the initial values for the parameters to fit

# Set `initialParameters` to 1000,0.05 and 50 respectively.

# [...] Write your code here

# ------------------------------------------------------------------------------
# 12) Fit the model

# Perform the fitting of the model using the function `curve_fit(model,X,Y,initial parameters)` and obtain the fitted parameter `fitobject.param`

# [...] Write your code here

# ------------------------------------------------------------------------------
# 13) Compute the errors

# Compute the standard error for each estimated parameter and the confidence interval at 10% significance level

# [...] Write your code here

# ------------------------------------------------------------------------------
# 14) Plot fitted model

# Plot a chart of fitted (y) by stand age (x) (i.e. the logisticModel with the given parameters)

# [...] Write your code here

# ------------------------------------------------------------------------------
# 15) Add the observations to the plot

# Add to the plot a scatter chart of the actual observed VHa

# [...] Write your code here


# ------------------------------------------------------------------------------
# 16) Differentiate the model per tree specie 

# Look at the growth curves of individual species. Try to perform the above analysis for individual species, for example plot the fitted curves for the 5 most common species

# [...] Write your code here

