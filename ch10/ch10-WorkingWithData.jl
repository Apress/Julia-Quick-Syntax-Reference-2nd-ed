################################################################################
# Code snippets of the book:
# A. Lobianco (2024) Julia Quick Syntax Reference - A Pocket Guide for Data
# Science Programming, Apress, Ed 2
# https://doi.org/[TODO_DOI]
# Licence: Open Domain
################################################################################

# Chapter 10 : Working with data
using Pkg
cd(@__DIR__)
Pkg.activate(".")
# If using a Julia version different than 1.10 or 1.11 please uncomment and run the following line (reproductibility guarantee will hower be lost). If your version of Julia can't find a suitable combination, try running `Pkg.up();Pkg.resolve()`
#Pkg.resolve()
Pkg.instantiate()
using Test


# ------------------------------------------------------------------------------
# Dataframes

# Create the example DataFrame
using DataFrames
dforig = DataFrame(region      = ["US","US","US","US","EU","EU","EU","EU"],
               product     = ["Hardwood","Hardwood","Softwood","Softwood","Hardwood","Hardwood","Softwood","Softwood"],
               year        = [2010,2020,2010,2020,2010,2020,2010,2020],
               production  = [3.3,3.2,2.3,2.1,2.7,2.8,1.5,1.3],
               consumption = [4.3,7.4,2.5,9.8,3.2,4.3,6.5,3.0]
)
df = deepcopy(dforig)
df2 = DataFrame(A = Int64[], B = Float64[])
mat = [1 2 3; 4 5 6]
headers = ["c1", "c2", "c3"]
df2 = DataFrame(mat,:auto)
df2 = DataFrame(mat,headers)

mat = ["c1" "c2" "c3"; 1 2 3; 4 5 6]
df2 = DataFrame(mat[2:end,:],mat[1,:])

using CSV
df2 = CSV.read(IOBuffer("""
region product  year production consumption
US     Hardwood 2010 3.3        4.3
US     Hardwood 2020 3.2        7.4
US     Softwood 2010 2.3        2.5
US     Softwood 2020 2.1        9.8
EU     Hardwood 2010 2.7        3.2
EU     Hardwood 2020 2.8        4.3
EU     Softwood 2010 1.5        6.5
EU     Softwood 2020 1.3        3.0
"""), DataFrame, delim=" ", ignorerepeated=true)

show(df)
n=3
first(df,n)
last(df,n)
describe(df)
names(df)
for r in eachrow(df) println(r) end
for c in eachcol(df) println(c) end
unique(df.year)
[unique(c) for c in eachcol(df)]
[eltype(c) for c in eachcol(df)]
size(df)
df[:,["product","year"] ]
df[!,["product","year"] ]
df.product
df[!,"product"]
df[:,2]
df[!,2]
df[2,:]
df.product[1:2:6]
df[2,"product"]

df[ [true,true,false,false,false,false,false,false],[true,true,true,false,true]]
df[ df.year .>= 2020, :] 
df[ (df.year .>= 2020) .& in.(df.region, Ref(["US","China"])), :]
using DataFramesMeta
reg_tofilter = :region; @subset(df, :production .> 3, cols(reg_tofilter) .== "US")
using Query
df2 = @from i in df begin
  @where i.region == "US"
  # Select a group of columns, eventually changing their name:
  @select {i.product, i.year, USProduction=i.production}
  @collect DataFrame
end

df2 = @from i in df begin
  @where i.production >= 3 && i.region in ["US","China"]
  @select i # Select the whole rows
  @collect DataFrame
end

df[[1,2],"production"] .= 4.2
df[(df.region .== "US") .& (df.product .== "Hardwood"), "production"] .= 5.2
reg_fullnames = Dict("US" => "United States", "EU" => "European Union"); df.region = map(r_shortname->reg_fullnames[r_shortname], df.region)

df2 = copy(df)
df2.net_export = df.production .- df.consumption
df2.net_import .=  df.consumption .- df.production 
df2.net_export = map((p,c) -> p - c, df.production, df.consumption)
push!(df,["EU" "Softwood" 2012 5.2 6.2]) # needs the original df
i=3; df2 = df[[1:(i-1);(i+1):end],:]
df2= similar(df,4)

sort!(df,["year","product"],rev=[false,true])

select!(df,Not(["product","year"]))
rename!(df, ["r","p","c"])
rename!(df, Dict("p"=>"prod","c"=>"cons"))

df = df[:,["prod", "cons", "r"] ]
df.id = 1:size(df, 1)
df.a = Array{Union{Missing,Float64},1}(missing,size(df,1))
insertcols!(df, 2, "something"=>zeros(9),"something_else"=>ones(9))
insertcols!(df, "b"=>fill(1,9))
df.b = convert(Array{Float64,1},df.b)
df.prod = map(string, df.prod)
string_to_float(str) = try parse(Float64, str) catch; return(missing) end
df.prod = map(string_to_float, df.prod)
df2 = similar(df, 0)

# Joining datasets
df1 = deepcopy(dforig); df2 = deepcopy(dforig);
vcat(df1,df2)
vcat([df1,df2]...)
hcat(df1,df2, makeunique=true)
push!(df1,["Japan","Hardwood",2020,5.0,1000.0])
push!(df2,["China","Softwood",2010,50,10000.0])
rename!(df2,Dict("production" => "prod"))
innerjoin(df1, df2, on = ["region","product","year"],makeunique=true)
leftjoin(df1, df2, on = ["region","product","year"],makeunique=true)
rightjoin(df1, df2, on = ["region","product","year"],makeunique=true)
outerjoin(df1, df2, on = ["region","product","year"],makeunique=true)
semijoin(df1, df2, on = ["region","product","year"])
antijoin(df1, df2, on = ["region","product","year"])
crossjoin(df1, df2,makeunique=true)
innerjoin(df1, df2, on = ["region","product"],makeunique=true)

# Categorical data
df = deepcopy(dforig)
using CategoricalArrays
df.region = categorical(df.region) # Transformation to a categorical array
levels(df.region)  # ["EU", "US"]
sort!(df,"region") # EU rows first
levels!(df.region,["US","EU"]) # Providing a presonalized order
levels(df.region)  # ["US", "EU"]
levels(df.region) .= ["United States", "European Union"] # Renaming
sort!(df,"region") # United States rows first
df.region = unwrap.(df.region)

# Missing data
df = deepcopy(dforig);
allowmissing!(df,"consumption")
allowmissing([1 2 3; 4 5 6])
disallowmissing(df)
disallowmissing(Matrix{Union{Missing,Int64}}([1 2 3; 4 5 6]))
disallowmissing!(df,"production")
df[2,"consumption"] = missing
dropmissing(df)
dropmissing!(df) # automatically disallow the missing
allowmissing!(df,["region","consumption"])
df[2,"consumption"] = missing
dropmissing(df,["consumption","production"])
completecases(df)
completecases(df,["consumption"])
nonmissingtype(Union{Float64,Missing})
df[4,"region"] = missing
ismissing.([2,missing])
isequal(missing,1)
[df[ismissing.(df[!,i]), i] .= 0 for i in names(df,Union{Missing,Number})]
[df[ismissing.(df[!,i]), i] .= "" for i in names(df,Union{Missing,AbstractString})]

length(findall(x -> ismissing(x), [1 missing 3; missing 4 5]))
length(findall(x -> ismissing(x), Matrix(df)))

using BetaML
# Build a dataframe with some missing data
df2 = CSV.read(IOBuffer("""
   col1    col2     col3
    1.4     2.5      "a"
missing    20.5      "b"
    0.6      18  missing
    0.7    22.8      "b"
    0.4 missing      "b"
    1.6     3.7      "a"
"""), DataFrame, delim=" ", missingstring="missing",ignorerepeated=true)
# Impute the missing values
data_imputed = fit!(RandomForestImputer(),Matrix(df2))
# Copy back the imputed value into the original dataframe
[df2[:,c] .= data_imputed[:,c] for c in axes(df2,2)]

# Pivoting
df = deepcopy(dforig)
long_df = stack(df,["production","consumption"])
long_df1 = stack(df)
long_df == long_df1 

wide_df = unstack(long_df,["region", "product", "year"],"variable","value")
wide_df1 = unstack(long_df,"variable","value")
wide_df == wide_df1 # true

wide_df_p = unstack(wide_df,["region","year"],"product","production")
wide_df_c = unstack(wide_df,["region","year"],"product","consumption")
rename!(wide_df_p, Dict("Hardwood" => "prod_Hardwood", "Softwood" => "prod_Softwood"))
rename!(wide_df_c, Dict("Hardwood" => "cons_Hardwood", "Softwood" => "cons_Softwood"))
widewide_df = innerjoin(wide_df_p,wide_df_c, on=["region","year"] )

# Split-apply-combine
df = deepcopy(dforig)
gdf = groupby(df, ["product"])

using Statistics # for the mean function
statistics = combine(groupby(df,["region","product"]),
    "production" => mean => "avg_prod",
    "consumption" => sum => "tot_cons",
    nrow
)
statistics2 = combine(groupby(df,["region","product"])) do subdf
    (avg_prod = mean(subdf.production),
    tot_cons = sum(subdf.consumption),
    nrow     = size(subdf,1))
end
@test statistics == statistics2
statistics3 = combine(groupby(df,["region","product"])) do subdf
  [mean(subdf.production) sum(subdf.consumption) size(subdf,1)]
end
statistics4 = combine(groupby(df,["region","product"])) do subdf
  DataFrame(avg_prod = mean(subdf.production),
  tot_cons = sum(subdf.consumption),
  nrow     = size(subdf,1))
end
@test statistics4 == statistics

cumprod = combine(groupby(df,["region","product"])) do subdf
    (year = subdf.year, prod = subdf.production, cumprod = cumsum(subdf.production))
end

using DataFramesMeta
cumprod = @linq df                                  |>
          groupby([:region,:product])               |>
          transform(:cumprod = cumsum(:production))

# DF export
Matrix(df)
Matrix{Union{Float64, Int64, String}}(df)
Matrix(df[:,["production","consumption"]])

function to_dict(df, dim_cols, value_col)
  ktypes = length(dim_cols) == 1 ? eltype(df[!,dim_cols[1]]) : Tuple{[eltype(df[!,dc]) for dc in dim_cols]...}
  to_return = Dict{ktypes,eltype(df[!,value_col])}()
  for r in eachrow(df)
      key_values = []
      [push!(key_values,r[d]) for d in dim_cols]
      to_return[(key_values...,)] = r[value_col]
  end
  return to_return
end


df_dict = to_dict(df,[:region,:product,:year],:production)

using HDF5
rm("out.h5",force=true)
h5write("out.h5", "mygroup/df", Matrix(df[:,["production","consumption"]]))
data = h5read("out.h5", "mygroup/df")

using JLD2
JLD2.jldopen("df.jld", "w") do f
  f["mydf"] = df
end
df2 = JLD2.load("df.jld","mydf")
# df3 = h5read("df.jld", "mydf") # This no longer works

# ------------------------------------------------------------------------------
# Indexed Tables 

# Create an IndexedTable

using IndexedTables
my_table = ndsparse((
            region      = ["US","US","US","US","EU","EU","EU","EU"],
            product     = ["Hardwood","Hardwood","Softwood","Softwood","Hardwood","Hardwood","Softwood","Softwood"],
            year        = [2020,2010,2020,2010,2020,2010,2020,2010]
          ),(
            production  = [3.3,3.2,2.3,2.1,2.7,2.8,1.5,1.3],
            consumption = [4.3,7.4,2.5,9.8,3.2,4.3,6.5,3.0]
         ))
my_table
my_table2 = ndsparse(df)

using Random, BenchmarkTools
n = 1000000
looked_value = 100
key1 = rand(1:1000,n); key2 = rand(1:1000,n); values = rand(n);
my_table2 = ndsparse((k1= key1, k2 = key2), (v=values,))
my_df2    = DataFrame(k1 = key1, k2 = key2, v = values)
@benchmark(my_table2[looked_value,:])            #   24 μs
@benchmark(my_df2[my_df2.k1 .== looked_value,:]) # 1004 μs


my_table["EU","Hardwood",2020] = (consumption = 3.4, production = 2.9)
my_table["EU","Hardwood",2012] = (production = 2.8, consumption = 3.3)

# ------------------------------------------------------------------------------
# Pipe chaining

using Pipe
add6(a) = a+6; div4(a) = a/4;
# Method #1, temporary variables:
a = 2;
b = add6(a);
c = div4(b);
println(c) # Output: 2
# Method 2, chained function calls:
println(div4(add6(a)))
# Method 3, using pipe
a |> add6 |> div4 |> println

mysum(x,y) = x+y; mydiv(x,y) = x/y
a = 2
# With temporary variables:
b = mysum(a,6)
c = mydiv(4,a)
d = b + c
println(d)
# With @pipe:
@pipe a |> mysum(_,6) + mydiv(4,_) |> println # Output: 10.0

data = (2,6,4)
# With temporary variables:
b = mysum(data[1],data[2]) # 8
c = mydiv(data[3],data[1]) # 2
d = b + c     # 10
println(d)
# With @pipe:
@pipe data |> mysum(_[1],_[2]) + mydiv(_[3],_[1]) |> println

# ------------------------------------------------------------------------------
# Plotting

# Choosing the backend
using Pkg
Pkg.add("Plots")
using Plots
plot(cos,-4pi,4pi, label="Cosine function (GR)") # Plot using the default GR backend
Pkg.add("PyPlot") # Install the PyPlot backend
pyplot() # Switch bakend
plot(cos,-4pi,4pi, label="Cosine function (PyPlot)") # Plot using the PyPlot backend
Pkg.add("PlotlyJS") # Install the PlotlyJS backend
plotlyjs() # Switch bakend
plot(cos,-4pi,4pi, label="Cosine function (PlotlyJS)") # Plot using the plotlyjs backend
gr()
backend()

# Plotting multiple series
using Plots
x = ["a","b","c","d","e",]
y = rand(5,3)
plot(x,y)

p = plot(x,y[:,1])  # create a new "current plot" and assign it to the variable p
plot!(x,y[:,2])     # edit the current plot
plot!(p, x,y[:,3])  # edit the plot assigned to variable p (that is also the current one)

plot(x,y[:,1]; seriestype=:scatter)

plot(x,y[:,1], seriestype=:bar)
plot!(x,y[:,2], seriestype=:line)
plot!(x,y[:,3], seriestype=:scatter)

plotattr() # require interactive choice on terminal

# Plotting from a dataframe

using DataFrames, StatsPlots
# Let's use a modified version of our example data with more years and just one region:
df = DataFrame(
  product       = ["Softwood","Softwood","Softwood","Softwood","Hardwood","Hardwood","Hardwood","Hardwood"],
  year        = [2010,2011,2012,2013,2010,2011,2012,2013],
  production  = [120,150,170,160,100,130,165,158],
  consumption = [70,90,100,95,   80,95,110,120]
)
mycolours = [:green :orange] # note it's a row vector and the colours of the series will be alphabetically ordered whatever order we give it here
timber_plot = @df df plot(:year, :production, group=:product, linestyle = :solid, linewidth=3, label=reshape(("Production of " .* sort(unique(:product))) ,(1,:)), color=mycolours)
@df df plot!(:year, :consumption, group=:product, linestyle = :dot, linewidth=3, label =reshape(("Consumption of " .* sort(unique(:product))) ,(1,:)), color=mycolours)

# Plotting distributions

using Distributions, StatsPlots
dist = Normal(10,10)
data = rand(dist,2000)
plot(dist, label="Exact distribution")
density!(data, label="Emp. dens, def bandwidth")
density!(data, bandwidth=0.5,label="Emp. dens with noises")

# Combine multiple plots in a single figure

l  = @layout [row1 ; r2c1 r2c2]; # create the layout obj
p1 = plot(x, y[:,1]);            # compose 1st plot
p2 = scatter(x, y[:,2]);         # compose 2nd plot
p3 = plot(x, y[:,3]);            # compose 3rd plot
plot(p1, p2, p3, layout = l)     # plot the final figure


savefig(timber_plot, "timber_markets.svg")
savefig("multiple_plots.pdf")
savefig("multiple_plots.png")

