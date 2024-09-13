################################################################################
# Code snippets of the book:
# A. Lobianco (2024) Julia Quick Syntax Reference - A Pocket Guide for Data
# Science Programming, Apress, Ed 2
# https://doi.org/[TODO_DOI]
# Licence: Open Domain
################################################################################

# Chapter 4 : Custom types

using Pkg
cd(@__DIR__)
Pkg.activate(".")
# If using a Julia version different than 1.10 or 1.11 please uncomment and run the following line (reproductibility guarantee will hower be lost). If your version of Julia can't find a suitable combination, try running `Pkg.up();Pkg.resolve()`
#Pkg.resolve()
Pkg.instantiate()
using Test

# ------------------------------------------------------------------------------
# Primitive types

primitive type My10KBBuffer 81920 end
primitive type My10KBBuffer2 <: Integer 81920 end

# ------------------------------------------------------------------------------
# Structures

mutable struct MyOwnType
  field1
  field2::String
  field3::Int64
end
mutable struct MyOwnType2{T<:Number}
  field1
  field2::String
  field3::T
  function MyOwnType2(i::T) where {T <: Number}
    i >= 0 || @error "Only non-negative numbers, please (you provided $i)"
    return new{T}(sqrt(i),string(i),i)
  end
end
@kwdef mutable struct MyOwnType3
  field1         = "foo"
  field2::String 
  field3::Int64  = 1
 end

my_object = MyOwnType("something","something",10)
a = my_object.field3  # 10
my_object.field3 = 20 # only if myObject is a mutable struct
# this would error: MyOwnType(field3 = 10, field1 = "something", field2 = "something") # Error!

my_object = MyOwnType3("aaa","bbb",10)
my_object = MyOwnType3(field2="bbb",field3=10)

function MyOwnType(i::T) where {T <: Number}
  i >= 0 || @error "Only non-negative numbers, please (you provided $i)"
  return MyOwnType(sqrt(i),string(i),i)
end

MyOwnType(3,"bbb",3)
MyOwnType(3)
# This would error: MyOwnType2(5,"aaa",3)
MyOwnType2(3)






# ------------------------------------------------------------------------------
# Abstract types and inheritance

abstract type MyOwnGenericAbstractType end
abstract type MyOwnAbstractType1 <: MyOwnGenericAbstractType end
abstract type MyOwnAbstractType2 <: MyOwnGenericAbstractType end
mutable struct AConcreteType1 <: MyOwnAbstractType1
  f1::Int64
  f2::Int64
end
mutable struct AConcreteType2 <: MyOwnAbstractType1
  f1::Float64
end
mutable struct AConcreteType3 <: MyOwnAbstractType2
  f1::String
end
o1 = AConcreteType1(2,10)
o2 = AConcreteType2(1.5)
o3 = AConcreteType3("aa")
function foo(a :: MyOwnGenericAbstractType)
  println("Default implementation: $(a.f1)")
end
foo(o1) # Default implementation: 2
foo(o2) # Default implementation: 1.5
foo(o3) # Default implementation: aa
function foo(a :: MyOwnAbstractType1)
  println("A more specialised implementation: $(a.f1*4)")
end
foo(o1) # A more specialised implementation: 8
foo(o2) # A more specialised implementation: 6.0
foo(o3) # Default implementation: aa
function foo(a :: AConcreteType1)
     println("A even more specialised implementation: $(a.f1 + a.f2)")
end
foo(o1) # A even more specialised implementation: 12
foo(o2) # A more specialised implementation: 6.0
foo(o3) # Default implementation: aa

@test Vector{Int64} <: AbstractVector{Int64}
@test Int64 <: Number
@test ! (AbstractVector{Int64} <: AbstractVector{Number})
@test ! (Vector{Int64} <: Vector{Number})
foo(x::AbstractVector{T}) where {T<:Number} = return sum(x)
foo([1, 2 ,3])
foo([1.5, 2.5 ,3.5])
v = Union{Float64,Int64}[1, 2.5 ,3]
foo(v)

# ------------------------------------------------------------------------------
# Object-oriented implementation

struct Shoes
   shoesType::String
   colour::String
end
struct Person
  myname::String
  age::Int64
end
struct Student
   p::Person
   school::String
   shoes::Shoes
end
struct Employee
   p::Person
   monthlyIncomes::Float64
   company::String
   shoes::Shoes
end
gymShoes = Shoes("gym","white")
proShoes = Shoes("classical","brown")
Marc = Student(Person("Marc",15),"Divine School",gymShoes)
MrBrown = Employee(Person("Brown",45),3200.0,"ABC Corporation Inc.", proShoes)
function printMyActivity(self::Student)
   println("Hi! I am $(self.p.myname), I study at $(self.school) school, and I wear $(self.shoes.colour) shoes")
end
function printMyActivity(self::Employee)
  println("Good day. My name is $(self.p.myname), I work at $(self.company) company and I wear $(self.shoes.colour) shoes")
end
printMyActivity(Marc)     # Hi! I am Marc, ...
printMyActivity(MrBrown)  # Good day. My name is MrBrown, ...

# ------------------------------------------------------------------------------
# Useful functions

MyType = AConcreteType1
obj    = AConcreteType1(2,10)
supertype(MyType)
subtypes(MyType)
fieldnames(MyType)
isa(obj,MyType)
typeof(obj)
eltype([1,2,3])