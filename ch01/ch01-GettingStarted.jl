################################################################################
# Code snippets of the book:
# A. Lobianco (2024) Julia Quick Syntax Reference - A Pocket Guide for Data
# Science Programming, Apress, Ed 2
# https://doi.org/[TODO_DOI]
# Licence: Open Domain
################################################################################

# Chapter 1 : Getting Started

using Pkg
cd(@__DIR__)
Pkg.activate(".")
#Pkg.resolve()
Pkg.instantiate()


# ------------------------------------------------------------------------------
# Miscellaneous syntax elements

println("Some code..")
#=
  Multiline comment
  #= nested multiline comment =#
  Still a comment
=#
println(#= A comment in the middle of the line =# "This is a code") # Normal single-line comment

# ------------------------------------------------------------------------------
# Modules

module Foo
export plus2, an_exported_var
an_exported_var = 2
a_private_var = 5
plus10(x) = x + a_private_var + 10
plus2(x) = x + a_private_var + 2
println("I am in Foo, I can access $a_private_var")
end

Main.Foo.a_private_var
Foo.a_private_var
using Main.Foo
using .Foo
an_exported_var

# ------------------------------------------------------------------------------
# Packages

using Plots
plot(rand(4,4))

import Plots as pl
pl.plot(rand(4,4)) # `Equivalent to Plots.plot(rand(4,4))`

import Plots: plot # You can import multiple functions at once using commas
plot(rand(4,4))

using Pkg
Pkg.add(["CSV","DataFrames"])