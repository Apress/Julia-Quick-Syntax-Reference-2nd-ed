################################################################################
# Code snippets of the book:
# A. Lobianco (2024) Julia Quick Syntax Reference - A Pocket Guide for Data
# Science Programming, Apress, Ed 2
# https://doi.org/[TODO_DOI]
# Licence: Open Domain
################################################################################

# Chapter 3 : Control flow and functions

using Pkg
cd(@__DIR__)
Pkg.activate(".")
# If using a Julia version different than 1.10 or 1.11 please uncomment and run the following line (reproductibility guarantee will hower be lost). If your version of Julia can't find a suitable combination, try running `Pkg.up();Pkg.resolve()`
#Pkg.resolve()
Pkg.instantiate()
using Test

# ------------------------------------------------------------------------------
# Code block structure and variable scope
for i in 1:5
    println(i)
end

a = 4
while a > 2
    a -= 1
end
println(a)

# ------------------------------------------------------------------------------
# Repeated iteration: for and while loops, list comprehension, maps

for i=1:2,j=2:5
    println("i: $i, j: $j")
end
x = [[1,2,3] [4,5,6]]; # 3x2 Matrix
for i in eachindex(x)
    println("$i: $(x[i])")
end
for c in axes(x,2), r in axes(x,1) # by col and row
    println("$r,$c: $(x[r,c])") 
end
for i in CartesianIndices(x)
    println("$i: $(x[i])")
end

x = [3,2];
function change_me!(x)
    for (i,e) in enumerate(x)
        x[i] = x[i] * 10
        e = e * 100
    end
end
change_me!(x)
x

myfunction = println
[myfunction(i) for i in [1,2,3]]
[x + 2y for x in [10,20,30], y in [1,2,3]]

mylist = ["a","b","c"]
mydict = Dict{Int64,String}()
students = Dict{String,String}()
names = ["Marc","Anne"]; genders = ["M","F"]
[mydict[i] = value  for (i, value) in enumerate(mylist)]
[students[name] = gender for (name,gender) in zip(names,genders)]
[println("i: $i - j: $j") for i in 1:5, j in 2:5 if i > j]
map((n,g) -> students[n] = g, names, genders)
f = println
a = map(f, [1,2,3])
a = map(x->f(x), [1,2,3])

# ------------------------------------------------------------------------------
# Conditional statements: if blocks, ternary operator

i = 5
if i == 1
    println("i is 1")
elseif i == 2
    println("i is 2")
else
    println("is in neither 1 or 2")
end

# ------------------------------------------------------------------------------
# Functions

f2(x,y) = 2x+y

function f2(x)
    x+2
end
# A nested function:
function f1(x)
    function f2(x,y)
        x+y
    end
    f2(x,2)
end
# A recursive function:
function fib(n)
    if n == 0  return 0
    elseif n == 1 return 1
    else
     return fib(n-1) + fib(n-2)
    end
end

myfunction2(a,b=1;c=2) = (a+b+c)
myfunction2(1,c=3)
myfunction2(a::Int64,b::Int64=1;c::Int64=2) = (a+b+c)
function f2(par::Union{Float64, Vector{Float64}})  end

# Function arguments
myvalues = [1,2,3]
function additionalAverage(init, args...) # The parameter that uses the ellipsis must be the last one
  s = 0
  for arg in args
    s += arg
  end
  return init + s/length(args)
end
a = additionalAverage(10,1,2,3)        # 12.0
a = additionalAverage(10, myvalues ...)  # 12.0

myfunction3(a,b) = a*2,b+2
x,y = myfunction3(1,2)
methods(myfunction3)


myfunction5(x::T, y::T2, z::T2) where {T <: Number, T2} = x + y + z
myfunction5(1,2,3)
myfunction5(1,2.5,3.5)

myfunction6(x::T where {T <: Number}, y::T2 where {T2}, z::T2 where {T2})  = x + y + z
myfunction6(1,2.5,3.5)

f2(x) = 2x # define a function f inline
a = f2(2)  # call f and assign the return value to a. `a` is a value
a = f2     # bind f to a new variable name. `a` is now a function
a(5)       # call again the (same) function


# Call by reference, call by value
function f2(x,y)
    x = 10
    y[1] = 10
end
x = 1
y = [1,1]
f2(x,y) # x will not change, but y will now be [10,1]

x -> x^2 + 2x - 1
f = (x,y) -> x+y

f1(a::Int64,b::Int64) = a*b
f1(2,3)
@test broadcast(f1,[2,3],[3,4]) == f1.([2,3],[3,4])

f2(a::Int64,b::Int64,c::Array{Int64,1},d::Array{Int64,1}) =  a*b+sum(c)-sum(d)
f2(1,2,[1,2,3],[0,0,1]) # normal call without broadcast
f2.([1,1,1],[2,2,2],Ref([1,2,3]),Ref([0,0,1])) # broadcast over the first two arguments only

f1(f2,x,y) = f2(x+1,x+2)+y

f1(2,8) do i,j
    i*j
end

# Write this function on `~/.julia/config/startup.jl` or equivalent on other OSs...
# If you just use it here it goes on a infinite loop..
#=
function workspace()
    atexit() do
        run(`$(Base.julia_cmd())`)
    end
    exit()
end
=#
#workspace()
