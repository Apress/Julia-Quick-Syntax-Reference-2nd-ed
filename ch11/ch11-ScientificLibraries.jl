################################################################################
# Code snippets of the book:
# A. Lobianco (2024) Julia Quick Syntax Reference - A Pocket Guide for Data
# Science Programming, Apress, Ed 2
# https://doi.org/[TODO_DOI]
# Licence: Open Domain
################################################################################

# Chapter 11 : Scientific Libraries

using Pkg
cd(@__DIR__)
Pkg.activate(".")
# If using a Julia version different than 1.10 or 1.11 please uncomment and run the following line (reproductibility guarantee will hower be lost). If your version of Julia can't find a suitable combination, try running `Pkg.up();Pkg.resolve()`
#Pkg.resolve()
Pkg.instantiate()
using Test



# ------------------------------------------------------------------------------
# Solving optimization problems with JuMP

#Numerically solve a linear problem
using JuMP, HiGHS
using CSV, DataFrames # only for loading the parameters

# Define sets #
#  Sets
#      i   canning plants   / seattle, san-diego /
#      j   markets          / new-york, chicago, topeka / ;
plants  = ["seattle","san-diego"]          # canning plants
markets = ["new-york","chicago","topeka"]  # markets

# Define parameters #
#   Parameters
#       a(i)  capacity of plant i in cases
#         /    seattle     350
#              san-diego   600  /
a = Dict(              # capacity of plant i in cases
  "seattle"   => 350,
  "san-diego" => 600,
)

#       b(j)  demand at market j in cases
#         /    new-york    325
#              chicago     300
#              topeka      275  / ;
b = Dict(              # demand at market j in cases
  "new-york"  => 325,
  "chicago"   => 300,
  "topeka"    => 275,
)

# Table d(i,j)  distance in thousands of miles
#                    new-york       chicago      topeka
#      seattle          2.5           1.7          1.8
#      san-diego        2.5           1.8          1.4  ;
d_table = CSV.read(IOBuffer("""
plants     new-york  chicago  topeka
seattle    2.5       1.7      1.8
san-diego  2.5       1.8      1.4
"""),DataFrame, delim=" ", ignorerepeated=true)
d = Dict( (r[:plants],m) => r[Symbol(m)] for r in eachrow(d_table), m in markets)
# Here we are converting the table in a "(plant, market) => distance" dictionary
# r[:plants]:   the first key, using the cell at the given row and `plants` field
# m:            the second key
# r[Symbol(m)]: the value, using the cell at the given row and the `m` field

# Scalar f  freight in dollars per case per thousand miles  /90/ ;
f = 90 # freight in dollars per case per thousand miles

# Parameter c(i,j)  transport cost in thousands of dollars per case ;
#            c(i,j) = f * d(i,j) / 1000 ;
# We first declare an empty dictionary and then we fill it with the values
c = Dict() # transport cost in thousands of dollars per case ;
[ c[p,m] = f * d[p,m] / 1000 for p in plants, m in markets]

# Model declaration (transport model)
trmodel = Model(HiGHS.Optimizer) # we choose HiGHS as solver engine and add a few options
set_optimizer_attribute(trmodel, "parallel", "on")
set_optimizer_attribute(trmodel, "output_flag", true)
## Define variables ##
#  Variables
#       x(i,j)  shipment quantities in cases
#       z       total transportation costs in thousands of dollars ;
#  Positive Variable x ;
@variables trmodel begin
    x[p in plants, m in markets] >= 0 # shipment quantities in cases
end

## Define contraints ##
# supply(i)   observe supply limit at plant i
# supply(i) .. sum (j, x(i,j)) =l= a(i)
# demand(j)   satisfy demand at market j ;
# demand(j) .. sum(i, x(i,j)) =g= b(j);
@constraints trmodel begin
    supply[p in plants],   # observe supply limit at plant p
        sum(x[p,m] for m in markets)  <=  a[p]
    demand[m in markets],  # satisfy demand at market m
        sum(x[p,m] for p in plants)   >=  b[m]
end

# Objective
@objective trmodel Min begin
    sum(c[p,m]*x[p,m] for p in plants, m in markets)
end

print(trmodel) # The model in mathematical terms is printed
optimize!(trmodel)
status = termination_status(trmodel)
if (status == MOI.OPTIMAL || status == MOI.LOCALLY_SOLVED || status == MOI.TIME_LIMIT) && has_values(trmodel)
    if (status == MOI.OPTIMAL)
        println("** Problem solved correctly **")
    else
        println("** Problem returned a (possibly suboptimal) solution **")
    end
    println("- Objective value (total costs): ", objective_value(trmodel))
    println("- Optimal routes:")
    optRoutes = value.(x)
    [println("$p --> $m: $(optRoutes[p,m])") for m in markets, p in plants]
    println("- Dual of supply:")
    [println("$p = $(dual(supply[p]))") for p in plants]
    println("- Dual of demand:")
    [println("$m = $(dual(demand[m]))") for m in markets]
else
    println("The model was not solved correctly.")
    println(status)
end

# Alternative...
using Ipopt
set_optimizer(trmodel, Ipopt.Optimizer)
optimize!(trmodel)
amodel = Model()


# Numerically solve a non-linear problem

using JuMP, Ipopt

m = Model()
set_optimizer(m,Ipopt.Optimizer)
set_optimizer_attribute(m, "print_level", 0)

@variable(m, 0 <= p, start=1, base_name="Quantities of pizzas")
@variable(m, 0 <= s, start=1, base_name="Quantities of sandwiches")

@constraint(m, budget,     10p + 4*s <=  80 )

@objective(m, Max, 100*p - 2*p^2 + 70*s - 2*s^2 - 3*p*s)

optimize!(m)

status = termination_status(m)

if (status == MOI.OPTIMAL || status == MOI.LOCALLY_SOLVED || status == MOI.TIME_LIMIT) && has_values(m)
    if (status == MOI.OPTIMAL)
        println("** Problem solved correctly **")
    elseif (status == MOI.LOCALLY_SOLVED)
        println("** Problem returned a (possibly local) optimal solution **")
    else
        println("** Problem returned a (possibly suboptimal) solution **")
    end
    println("- Objective value : ", objective_value(m))
    println("- Optimal solutions:")
    println("  - pizzas: $(value.(p))")
    println("  - sandwitches: $(value.(s))")
    println("- Shadow price (budget): $(shadow_price.(budget))")
else
    println("The model was not solved correctly.")
    println(status)
end

# ------------------------------------------------------------------------------
# Symbolic computations

# Analytically solve the same non-linear problem

using SymPyPythonCall
@syms qₚ::positive qₛ::positive pₚ::positive pₛ::positive λ
α = symbols("α", integer=true, positive=true)
typeof(:(qₚ+qₛ))
typeof(:(qₚ))
typeof(qₚ+qₛ)
utility = 100*qₚ - 2*qₚ^2 + 70*qₛ - 2*qₛ^2 - 3*qₚ*qₛ
budget  = pₚ* qₚ + pₛ*qₛ
lagr    = utility + λ*(80 - budget)
dlqₚ    = diff(lagr,qₚ)
dlqₛ    = diff(lagr,qₛ)
dlλ     = diff(lagr,λ)
sol     = solve((Eq(dlqₚ,0), Eq(dlqₛ,0), Eq(dlλ,0)),(qₚ, qₛ, λ))
solve((dlqₚ, dlqₛ, dlλ),(qₚ, qₛ, λ))
qₚ_num  = sol[qₚ].evalf(subs=Dict(pₚ=>10,pₛ=>4)) # 4.64285714285714
qₛ_num  = sol[qₛ].evalf(subs=Dict(pₚ=>10,pₛ=>4)) # 8.39285714285714
λ_num   = sol[λ].evalf(subs=Dict(pₚ=>10,pₛ=>4))   # 5.625
z = utility.evalf(subs=Dict(qₚ=>qₚ_num, qₛ=>qₛ_num)) #750.892857142857

typeof(qₚ_num) # Sym{PythonCall.Core.Py}
N(qₚ_num)
typeof(N(qₚ_num)) # Float64

# ------------------------------------------------------------------------------
# Curve Fitting

# Fitting data to a logistic model

using LsqFit,CSV,DataFrames,Plots
# **** Fit volumes to logistic model ***
@. model(age, pars) = pars[1] / (1+exp(-pars[2] * (age - pars[3])) )
obsVols = CSV.read(IOBuffer("""
age  vol
20   56
35   140
60   301
90   420
145  502
"""),DataFrame,delim=" ", ignorerepeated=true)
par0   = [600, 0.02, 60]
par_lb = [200, 0.001, 5]
par_ub = [2000, 0.1, 1000]
fit = curve_fit(model, obsVols.age, obsVols.vol, par0, lower=par_lb, upper=par_ub)
fit.param # [497.07, 0.05, 53.5]
fitX = 0:maximum(obsVols.age)*1.2
fitY  = [fit.param[1] / (1+exp(-fit.param[2] * (y - fit.param[3]) ) ) for y in fitX]
plot(fitX,fitY, seriestype=:line, label="Fitted values")
plot!(obsVols.age, obsVols.vol, seriestype=:scatter, label="Obs values")
plot!(obsVols.age, fit.resid, seriestype=:bar, label="Residuals")

using Distributions, StatsPlots
d = Binomial(100,0.2) # number of trials, prob single trial 
params(d) # (100, 0.2)
mean(d)   # (20.0)
var(d)    # 16.0
median(d) # 20
rand(d)   # 13
using Random
rand(Xoshiro(123),d,(2,3)) # [31 18 23; 16 16 19]
pdf(d,20)        # 0.099
y = cdf(d,25)    # 0.91
quantile(d,0.91) # 25
bar(x -> pdf(d,x),1:40,label="Binomial(n=100,p=0.2)")

d = MvNormal([10,10,10],[10 5 7; 5 8 5; 7 5 9] )
params(d)
var(d) 
d = Normal(5,5)
var(d)