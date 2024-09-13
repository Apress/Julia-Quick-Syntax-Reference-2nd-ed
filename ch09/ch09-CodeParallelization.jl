################################################################################
# Code snippets of the book:
# A. Lobianco (2024) Julia Quick Syntax Reference - A Pocket Guide for Data
# Science Programming, Apress, Ed 2
# https://doi.org/[TODO_DOI]
# Licence: Open Domain
################################################################################

# Chapter 8 :  Code parallelization
using Pkg
cd(@__DIR__)
Pkg.activate(".")
# If using a Julia version different than 1.10 or 1.11 please uncomment and run the following line (reproductibility guarantee will hower be lost). If your version of Julia can't find a suitable combination, try running `Pkg.up();Pkg.resolve()`
#Pkg.resolve()
Pkg.instantiate()
using Test

using BenchmarkTools

# ------------------------------------------------------------------------------
# GPU Programming

# Function definitions
relu(x) = max(0,x)
forward_layer(x,w,w0,f) = f.(w*x .+ w0)
function forward_network!(y,x,w1,w2,w3,w01,w02,w03,f=relu)
    x1 = forward_layer(x,w1,w01,f)
    x2 = forward_layer(x1,w2,w02,f)
    y  .= forward_layer(x2,w3,w03,identity)
    return nothing
end

# CPU data
(nd0,nd1,nd2,ndy) = (20000,30000,30000,1)
x   = rand(Float32,nd0);      y = Vector{Float32}(undef,ndy)
w1  = rand(Float32,nd1,nd0); w2 = rand(Float32,nd2,nd1); w3 = rand(Float32,ndy,nd2)
w01 = rand(Float32,nd1);    w02 = rand(Float32,nd2);    w03 = rand(Float32,ndy);
# CPU call
forward_network!(y,x,w1,w2,w3,w01,w02,w03,relu)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# This section requires you to have NVIDIA GPU and CUDA drivers installed on the os
using CUDA

# GPU data 
y_g   = CuArray{Float32}(undef,ndy)
x_g   = CuArray(x)
w1_g  = CuArray(w1);  w2_g  = CuArray(w2);  w3_g  = CuArray(w3);
w01_g = CuArray(w01); w02_g = CuArray(w02); w03_g = CuArray(w03); 
# GPU call
forward_network!(y_g,x_g,w1_g,w2_g,w3_g,w01_g,w02_g,w03_g,relu)

# Correctness check
@test y ≈ Array(y_g)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# This section requires you to have INTEL GPU and drivers installed on the os
# Note that oneAPI support is still experimental

using oneAPI

# GPU data 
y_g   = oneArray{Float32}(undef,ndy)
x_g   = oneArray(x)
w1_g  = oneArray(w1);  w2_g  = oneArray(w2);  w3_g  = oneArray(w3);
w01_g = oneArray(w01); w02_g = oneArray(w02); w03_g = oneArray(w03); 
# GPU call
forward_network!(y_g,x_g,w1_g,w2_g,w3_g,w01_g,w02_g,w03_g,relu)

# Correctness check
@test y ≈ Array(y_g)

# ------------------------------------------------------------------------------
# Multi-threading

Threads.nthreads() # 4 in my case 
Threads.threadid()

function inner_function(x,cheap) # sum of square roots
    N = cheap ? 10 : 10000 
    return sum(sqrt(x) for i in 1:N) 
end
function rootsq_sums!(x;multithread=false,cheap=true)
    if multithread
        Threads.@threads for i in eachindex(x)
            x[i] = inner_function(x[i],cheap)
        end
    else
        for i in eachindex(x)
            x[i] = inner_function(x[i],cheap)
        end
    end
    return x
end

@assert rootsq_sums!(collect(1.0:100.0)) == rootsq_sums!(collect(1.0:100.0),multithread=true) 

x = collect(1.0:100.0)
@btime rootsq_sums!($x,multithread=false,cheap=true) 
@btime rootsq_sums!($x,multithread=true,cheap=true) 
@btime rootsq_sums!($x,multithread=false,cheap=false) 
@btime rootsq_sums!($x,multithread=true,cheap=false) 

x = collect(1.0:10000.0)
@btime rootsq_sums!($x,multithread=false,cheap=true) 
@btime rootsq_sums!($x,multithread=true,cheap=true) 
@btime rootsq_sums!($x,multithread=false,cheap=false) 
@btime rootsq_sums!($x,multithread=true,cheap=false) 

# GPU vs multithreading
# Note that on the GPU likely also the sum of the inner function is parallelised
using CUDA, BenchmarkTools

function inner_function_cheap(x) # sum of square roots
    return sum(sqrt(x) for i in 1:10) 
end
function inner_function_expensive(x) # sum of square roots
    return sum(sqrt(x) for i in 1:10000) 
end
function rootsq_sums_threads!(x;multithread=false,cheap=true)
    if multithread
        Threads.@threads for i in eachindex(x)
            x[i] = inner_function_cheap(x[i])
        end
    else
        for i in eachindex(x)
            x[i] = inner_function_expensive(x[i])
        end
    end
    return x
end
function rootsq_sums_cheap!(x)
    x .= inner_function_cheap.(x)
    return nothing
end
function rootsq_sums_expensive!(x)
    x .= inner_function_expensive.(x)
    return nothing
end

Threads.nthreads()
x = collect(1.0:100.0)
print("small x, no mt, cheap: "); @btime rootsq_sums_threads!($x,multithread=false,cheap=true) 
print("small x, mt, cheap: "); @btime rootsq_sums_threads!($x,multithread=true,cheap=true) 
print("small x, no mt, exp: "); @btime rootsq_sums_threads!($x,multithread=false,cheap=false) 
print("small x, mt, exp: "); @btime rootsq_sums_threads!($x,multithread=true,cheap=false) 
print("small x, base no return, cheap: "); @btime rootsq_sums_cheap!($x)
print("small x, base no return, exp: "); @btime rootsq_sums_expensive!($x)
x_g = CuArray(collect(1.0:100.0))
print("small x, gpu, cheap: "); @btime rootsq_sums_cheap!($x_g)
x_g = CuArray(collect(1.0:100.0))
print("small x, gpu, exp: "); @btime rootsq_sums_expensive!($x_g)

x = collect(1.0:10000.0)
print("large x, no mt, cheap: "); @btime rootsq_sums_threads!($x,multithread=false,cheap=true) 
print("large x, mt, cheap: "); @btime rootsq_sums_threads!($x,multithread=true,cheap=true) 
print("large x, no mt, exp: "); @btime rootsq_sums_threads!($x,multithread=false,cheap=false) 
print("large x, mt, exp: "); @btime rootsq_sums_threads!($x,multithread=true,cheap=false) 
print("large x, base no return, cheap: "); @btime rootsq_sums_cheap!($x)
print("large x, base no return, exp: "); @btime rootsq_sums_expensive!($x)
x_g = CuArray(collect(1.0:10000.0))
print("large x, gpu, cheap: "); @btime rootsq_sums_cheap!($x_g)
x_g = CuArray(collect(1.0:100.0))
print("large x, gpu, exp: "); @btime rootsq_sums_expensive!($x_g)

# ------------------------------------------------------------------------------
# Multi-processing

# Managing processes

using Distributed
wksIDs = addprocs(3) # 2,3,4
println("Worker pids: ")
for pid in workers()
    println(pid) # 2,3,4
end
rmprocs(wksIDs[2]) # or rmprocs(workers()[2]) remove process pid 3
println("Worker pids: ")
for pid in workers()
    println(pid) # 2,4 left
end
@everywhere println(myid()) # 2,4


# Run heavy tasks in parallel
using Distributed, BenchmarkTools
a = rand(1:35,100)
@everywhere function fib(n)
    if n == 0 return 0 end
    if n == 1 return 1 end
    return fib(n-1) + fib(n-2)
end
@benchmark results = map(fib,a)  # serialised: median time:   490.473 ms
@benchmark results = pmap(fib,a) # parallelised: median time: 249.295 ms


# Aggregate results
using Distributed, BenchmarkTools
function f(n)
  s = 0.0
  for i = 1:n
    s += i/2
  end
    return s
end
function pf(n)
  s = @distributed (+) for i = 1:n # aggregate using sum on variable s
        i/2                        # last element of for cycle is used by the aggregator
  end
  return s
end
@benchmark  f(10000000) # median time:      11.478 ms
@benchmark pf(10000000) # median time:      4.458 ms
