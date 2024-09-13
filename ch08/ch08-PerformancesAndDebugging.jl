################################################################################
# Code snippets of the book:
# A. Lobianco (2024) Julia Quick Syntax Reference - A Pocket Guide for Data
# Science Programming, Apress, Ed 2
# https://doi.org/[TODO_DOI]
# Licence: Open Domain
################################################################################

# Chapter 8 : Efficiently write efficient code

using Pkg
cd(@__DIR__)
Pkg.activate(".")
# If using a Julia version different than 1.10 or 1.11 please uncomment and run the following line (reproductibility guarantee will hower be lost). If your version of Julia can't find a suitable combination, try running `Pkg.up();Pkg.resolve()`
#Pkg.resolve()
Pkg.instantiate()
using Test


# ------------------------------------------------------------------------------
# Benchmarking
function fib(n)
    if n == 0 return 0 end
    if n == 1 return 1 end
    return fib(n-1) + fib(n-2)
end

println("Done fib")

@time @eval fib(10)
@time @eval fib(10)
@time fib(10) # @eval does put an hoverhead

using BenchmarkTools
@benchmark fib(10)

# ------------------------------------------------------------------------------
# Profiling example
using Profile
function lAlgb(n)
 a = rand(n,n) # matrix initialisation with random numbers
 b = a + a     # matrix sum
 c = b * b     # matrix multiplication
end
@profile (for i = 1:100; lAlgb(1000); end)
Profile.print()
Profile.clear()

lAlgb(10)
@profview  (for i = 1:100; lAlgb(1000); end) # Only in VSCode Julia extension


# ------------------------------------------------------------------------------
# Debugging
@run fib(5) # If you don't set any breackpoints it may crash

# ------------------------------------------------------------------------------
# Type stable/unstable functions

using BenchmarkTools

function f_unstable(x)    # Type unstable
    out_vector = [1,2.0,"2"]
    if x < 0
        return out_vector[1]
    elseif x == 0
        return out_vector[2]
    else
        return out_vector[3]
    end
end

function f_stable(x)   # Type stable
    out_vector = [1,convert(Int64,2.0),parse(Int64,"2")]
    if x < 0
        return out_vector[1]
    elseif x == 0
        return out_vector[2]
    else
        return out_vector[3]
    end
end

@code_warntype f_unstable(2)
@code_warntype f_stable(2)
@benchmark f_unstable(2) # median time: 234 ns
@benchmark f_stable(2)   # median time: 118 ns

slow_function() = begin a = []; [push!(a,i) for i in 1:1000]; return sum(a) end
fast_function() = begin a = Int64[]; [push!(a,i) for i in 1:1000]; return sum(a) end
@btime slow_function()
@btime fast_function()

@btime begin x = []; map(i->push!(x,i), 1:10000) end
@btime begin x = Int64[]; map(i->push!(x,i),1:10000) end
@btime begin x = Vector{Int64}(undef,10000); map(i->x[i] = i, 1:10000) end

# ------------------------------------------------------------------------------
# Matrix loops

using BenchmarkTools
M = rand(2000,2000)
function sum_row_col(M) # slower
    s = 0.0
    for r in 1:size(M)[1]
        for c in 1:size(M)[2]
            s += M[r,c]
        end
    end
    return s
end
function sum_col_row(M) # faster
    s = 0.0
    for c in 1:size(M)[2]
        for r in 1:size(M)[1]
            s += M[r,c]
        end
    end
    return s
end
@benchmark sum_row_col(M) # median time: 24.3 ms
@benchmark sum_col_row(M) # median time:  5.8 ms

# ------------------------------------------------------------------------------
# Introspection tools

myfunction = sum
methods(myfunction)
myargs = (2,3)
@which myfunction(myargs)
@test typeof(ones(Int64,10)) == Array{Int64,1}
@test eltype(ones(Int64,10)) == Int64
struct AType foo; boo; end
myobj = AType(1,2)
fieldnames(AType)
dump(myobj)
 
@less myfunction(myargs)
@edit myfunction(myargs)
@code_native f_unstable(2)
@code_llvm f_unstable(2)
@code_typed f_unstable(2)
@code_lowered f_unstable(2)
names(Main,all=false)
sizeof(1.5)
bitstring(8)
# ------------------------------------------------------------------------------
# Exceptions
data =Dict(("volume","Germany",2020) => 3683,
           ("volume","France",2020)  => 3055)
function volume(region, year)
    try
        return data["volume",region,year]
    catch  e
        if isa(e, KeyError)
          return missing
        end
        rethrow(e)
    end
end
volume("Germany",2020) # 3683
volume("Germany",2025) # missing
get(data,("Volume","Germany",2025),missing)


