################################################################################
# Code snippets of the book:
# A. Lobianco (2024) Julia Quick Syntax Reference - A Pocket Guide for Data
# Science Programming, Apress, Ed 2
# https://doi.org/[TODO_DOI]
# Licence: Open Domain
################################################################################

# Chapter 6 : Metaprogramming and macros

using Pkg
cd(@__DIR__)
Pkg.activate(".")
# If using a Julia version different than 1.10 or 1.11 please uncomment and run the following line (reproductibility guarantee will hower be lost). If your version of Julia can't find a suitable combination, try running `Pkg.up();Pkg.resolve()`
#Pkg.resolve()
Pkg.instantiate()
using Test

# ------------------------------------------------------------------------------
# Symbol and Expressions

@test :foo10 == Symbol("foo10")

a = Symbol("foo",10)
string(a)

expr = Meta.parse("b = -(a+1) # This is a comment");
typeof(expr)
dump(expr)

a = 1
function foo()
  local a = 2
  expr = :(a + 1)
  return a+1, eval(expr)
end
foo()


a = 1
expr1  = Meta.parse("$a + b + 3")
expr2 = :($a + b + 3)              # Equiv. to expr1
expr3 = quote $a + b + 3 end       # Equiv. to expr1
expr4 = Expr(:call, :+, a, :b, 3)  # Equiv. to expr1
expr = Meta.parse("3+2")
eval(expr) # 5
# this would error: eval(expr1)                        # UndefVarError: b not defined
b = 10
eval(expr1)                        # 14
a = 100
eval(expr1)                        # Still 14
b = 100
eval(expr1)                        # 104

a = 1
function foo()
  local a = 2
  expr = :(a + 1)
  return a+1, eval(expr)
end
foo() #

# ------------------------------------------------------------------------------
# Macros

macro customLoop(controlExpr,workExpr)
  return quote
    for i in $controlExpr
      $workExpr
    end
  end
end
a = 5
@customLoop 1:4 println(i)
@customLoop 1:a println(i)
@customLoop 1:a if i > 3 println(i) end
@customLoop ["apple", "orange", "banana"]  println(i)
@customLoop ["apple", "orange", "banana"]  begin print("i: "); println(i)  end
@macroexpand @customLoop 1:4 println(i)

# ------------------------------------------------------------------------------
# String macro

macro print8_str(mystr)
  limits = collect(1:8:length(mystr))
  for (i,j) in enumerate(limits)
    st = j
    en = i==length(limits) ? length(mystr) : j+7
    println(mystr[st:en])
  end
end
print8"123456789012345678"
