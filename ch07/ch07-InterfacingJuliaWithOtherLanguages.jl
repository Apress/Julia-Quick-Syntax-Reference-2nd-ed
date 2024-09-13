################################################################################
# Code snippets of the book:
# A. Lobianco (2024) Julia Quick Syntax Reference - A Pocket Guide for Data
# Science Programming, Apress, Ed 2
# https://doi.org/[TODO_DOI]
# Licence: Open Domain
################################################################################

# Chapter 7 : Interfacing Julia with other languages

using Pkg
cd(@__DIR__)
Pkg.activate(".")
# If using a Julia version different than 1.10 or 1.11 please uncomment and run the following line (reproductibility guarantee will hower be lost). If your version of Julia can't find a suitable combination, try running `Pkg.up();Pkg.resolve()`
#Pkg.resolve()
Pkg.instantiate()
using Test

# ------------------------------------------------------------------------------
# Using the C interface

write("myclib.h",
"""
extern int get2 ();
extern double sumMyArgs (float i, float j);
""")

write("myclib.c",
"""
int get2 (){
 return 2;
}
double sumMyArgs (float i, float j){
 return i+j;
}
""")

run(`gcc -o myclib.o -c myclib.c`)
run(`gcc -shared -o libmyclib.so myclib.o -lm -fPIC`)

const myclib = joinpath(@__DIR__, "libmyclib.so")
µa = ccall((:get2,myclib), Int32, ())
b = ccall((:sumMyArgs,myclib), Float64, (Float32,Float32), 2.5, 1.5)

# ------------------------------------------------------------------------------
# Embedding C++

using CxxWrap

write("libcpp.cpp",
"""
#include <string>
#include <iostream>
#include <vector>

#include "jlcxx/jlcxx.hpp"
#include "jlcxx/functions.hpp"
#include "jlcxx/stl.hpp"

using namespace std;

// Fist example: hello world without any data exchange between caller and called function.. 
void cpp_hello() {
  std::cout << "Hello world from a C++ function" << std::endl;
  return;
}

// Second example: primitive data passed and retrieved..
double cpp_average(int a, int b) {
  return (double) (a+b)/2;
}

// Third example: data relate to STD objects...
string cpp_sum (std::vector< double > data) {
  // Compute sum..
  double total = 0.0;
  double nelements = data.size();
  for (int i = 0; i< nelements; i++){
      total += data[i];
  }
  std::stringstream ss;
  ss << "The sum is " << total << endl;
  return ss.str();
}

// 4th example: complex, nested STD objects...
std::vector<double> cpp_multiple_averages (std::vector< std::vector<double> > data) {
  // Compute average of each element..
  std::vector <double> averages;
  for (int i = 0; i < data.size(); i++){
    double isum = 0.0;
    double ni= data[i].size();
    for (int j = 0; j< data[i].size(); j++){
      isum += data[i][j];
    }
    averages.push_back(isum/ni);
  }
  return averages;
}

JLCXX_MODULE define_julia_module(jlcxx::Module& mod) {
  mod.method("cpp_hello", &cpp_hello);
  mod.method("cpp_average", &cpp_average);
  mod.method("cpp_sum", &cpp_sum);
  mod.method("cpp_multiple_averages", &cpp_multiple_averages);
}
""")

cxx_include_path   = joinpath(CxxWrap.prefix_path(),"include")
julia_include_path = joinpath(Sys.BINDIR,"..","include","julia")

# Compile (in Linux, requires g++ installed) 
cmd = `g++ --std=c++20 -shared -fPIC -o libcpp.so -I $julia_include_path -I $cxx_include_path  libcpp.cpp`
run(cmd)
# Generate the functions for Julia
# Once the lib is wrappd you can't wrap it again nor modify the C++ code, you need to restart Julia
@wrapmodule(() -> joinpath(pwd(),"libcpp"))
# Call the functions
cpp_hello() # Prints "Hello world from a C++ function"
avg        = cpp_average(3,4) # 3.5
data_julia = [1.5,2.0,2.5]
data_sum   = cpp_sum(StdVector(data_julia)) # Returns "The sum is 6"
typeof(data_sum)
typeof(data_sum) <: AbstractString
data_julia = [[1.5,2.0,2.5],[3.5,4.0,4.5]]
data       = StdVector(StdVector.(data_julia))
data_avgs  = cpp_multiple_averages(data) # [2.0, 4.0]
typeof(data_avgs)
typeof(data_avgs) <: AbstractArray
data_avgs[1]

# ------------------------------------------------------------------------------
# Embedding Python

using PythonCall, Pipe

i = 3
@pyexec (i=i, j=4) => """
a=i+j
b=i/j
""" => (a::Int64,b::Float64)
typeof(a)

@pyexec """
def python_sum(i, j):
    return i+j
""" => python_sum
@pyexec """
def get_ith_element(n):
    a = [0,1,2,3,4,5,6,7,8,9]
    return a[n]
""" => get_ith_element

c = @pipe python_sum(3,4) |> pyconvert(Int64,_)                 # 7
d = @pipe python_sum([3,4],[5,6]) |> pyconvert(Vector{Int64},_) # [8,10]
typeof(d)                                                       # Vector{Int64}
e = @pipe get_ith_element(i) |> pyconvert(Int64,_)              # 3

pyexec(read("python_code.py", String),Main)
@pyexec (i=i, j=4) => "f = python_sum(i,j)" => (f::Float64)

# Using Python libraries

# Use the conda-based, project specific python...
using PythonCall
Pkg.add("CondaPkg")   # only once
import CondaPkg       # only once
CondaPkg.add("ezodf") # only once
const ez = pyimport("ezodf") 

## Use the system default Python
ENV["JULIA_CONDAPKG_BACKEND"] = "Null"
ENV["JULIA_PYTHONCALL_EXE"]   = "~.pyenv/shims/python" # this is ok in Linux Ubuntu
using PythonCall
const ez = pyimport("ezodf") # must be already available in your Python installation. Equiv. of Python `import ezodf as ez`
dest_doc = ez.newdoc(doctype="ods", filename="an_ods_sheet.ods")
typeof(dest_doc)
sheet = ez.Sheet("Sheet1", size=(10, 10))
dest_doc.sheets.append(sheet)
dcell1 = sheet[(2,3)] # This is cell "D3", not "B2"
dcell1.set_value("Hello")
sheet["A9"].set_value(10.5)
dest_doc.backup = false
dest_doc.save()

# ------------------------------------------------------------------------------
# Embedding R

ENV["R_HOME"]="*"
Pkg.build("RCall") # Only once or all the times you want to switch between private to Julia and shared R environment
using RCall

a = [1,2]; b= 3
@rput a
# $  c = a + $b # in R (type the dollar sign in the terminal)
@rget c

R"""
r_sum <- function(i,j) i+j
get_ith_element <- function(n) {
  a <- c(0,1,2,3,4,5,6,7,8,9)
  return(a[n])
}
"""
i = [3,4]
a = rcopy(R"r_sum"(3,4))         # 7
b = rcopy(R"r_sum"(i,[5,6]))     # [8,10]
b = rcopy(R"r_sum($i,c(5,6))")   # [8.0,10.0] (alternative)
c = rcopy(R"r_sum"([3,4],5))     # [8,9]
d = rcopy(R"get_ith_element"(1)) # 0.0

convert(Array{Float64}, R"r_sum"(i,[5,6])) # [8.0,10.0]
convert(Array{Int64}, R"r_sum($i,c(5,6))") # [8,10]

import Conda
Conda.update()
Conda.add("r-ggplot2") # This may require restart Julia if the update command above also updated R

# The following use the installed system version of R instead of the conda-based installed one and install on it ggplot2... 
# It may be incompatible (and may crash your Julia session) to use on the same session R from conda and system R
ENV["R_HOME"]="/usr/lib/R"
Pkg.build("RCall")
using RCall
R"options(download.file.method='wget')"
R"install.packages('ggplot2', repos='https://cloud.r-project.org/')"


using DataFrames

mydf = DataFrame(deposit = ["London","Paris","New-York","Hong-Kong"]; q =  [3,2,5,3] )  # Create a DataFrame ( a tabular structure with named cols) in Julia
R"""
  library(ggplot2)
  ggplot($mydf, aes(x = q)) +
  geom_histogram(binwidth=1)
"""
