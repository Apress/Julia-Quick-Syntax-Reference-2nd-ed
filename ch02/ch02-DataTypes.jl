################################################################################
# Code snippets of the book:
# A. Lobianco (2024) Julia Quick Syntax Reference - A Pocket Guide for Data
# Science Programming, Apress, Ed 2
# https://doi.org/[TODO_DOI]
# Licence: Open Domain
################################################################################

# Chapter 2 : Data types and structures

using Pkg
cd(@__DIR__)
Pkg.activate(".")
#Pkg.resolve()
Pkg.instantiate()
using Test

# ------------------------------------------------------------------------------
# Simple types (Non-Containers)

typemax(Int64)
a = 1 + 2im
a = 2 // 3
a = 3^2
ℯ
MathConstants.e
a = 7 ÷ 3
a = 7 % 3

# ------------------------------------------------------------------------------
# Strings
a = "a string"
b = "a string\non multiple rows\n"
c = """
a string
on multiple rows
"""
a[3] # Returns 's'
myInt = parse(Int,"2017")
myString = string(123)
str1 = "aaa"
struct FooInt 
    int1::Int64
end
my_object = FooInt(3)
a = "$str1 is a string and $(my_object.int1) is an integer"


# ------------------------------------------------------------------------------
# Arrays
a = [1;2;3]
a = [1,2,3]
a = [1 2 3]
a = []
a = Int64[]
T = Float64
a = Array{T,1}()
c = Vector{T}()
n = 3
a=zeros(Int64,n)
a=ones(Int64,n)
a = Array{T,1}(undef,n)
j = 2
a=fill(j, n)
x = [10, "foo", false]
a = Union{Int64,String,Bool}[10, "Foo", false]
@test Any[1,1.5,2.5] == Array{Any,1}([1,1.5,2.5]) 
a = Array{Int64,1}
a = [1,2] 
a[1]
a = collect(5:10)
@test collect(4:2:8) == [4,6,8]
@test collect(8:-2:4) == [8,6,4]
@test reverse(a) == collect(a[end:-1:1])
@test [2015; 2025:2030; 2100] == vcat(2015, 2025:2030, 2100)
a = Any[1,2,3]
b = [4,5]
push!(a,b)
append!(a,b)
c = vcat(1,[2,3],[4,5])
pop!(a)
popfirst!(a)
pos = 2 
deleteat!(a, pos)
pushfirst!(a,b)
a = [1,2,3,2]
sort!(a, rev=false)
sort(a, rev=false)
sortperm(a, rev=false)
unique!(a)
unique(a)
reverse(a)
a[end:-1:1]
b = 2
in(b, a)
if b in a 
end
length(a)
maximum(a)
max(a...)
minimum(a)
min(a...)
sum(a)
cumsum(a)
empty!(a)
a = [1 2 3]
b = vec(a)
using Random
shuffle(a)
shuffle!(a)
isempty(a)
value = 2
myarray = [1,3,2]
findall(x -> x == value, myarray)
my_unwanted_item = 2
deleteat!(myarray, findall(x -> x == my_unwanted_item, myarray))
for (idx,v) in enumerate(myarray) 
println(idx," ",v)
end
names = ["Marc", "Anne"]; gender = ['M','F']; age=[18,16];
for z in zip(names,gender,age)
    println(z)
end

# ------------------------------------------------------------------------------
# Multidimensional and nested arrays
a = [[1,2,3] [4,5,6]]
a = [1 4; 2 5; 3 6]
[1;2;3;; 10;20;30;;; 4;5;6;; 40;50;60;;;]
a = Array{T}(undef, 0, 0, 0)
(n,m,g) = (2,3,4)
a = zeros(n,m,g)
a = zeros(Int64,n,m,g)
a = ones(n,m,g)
a = ones(Int64,n,m,g)
T = Float64
a = Array{T,3}(undef,n,m,g)
j = 2
a = fill(j,n,m,n)
a = rand(n,m,g)
a = [3x + 2y + z for x in 1:2, y in 2:3, z in 1:2]
a = [[1,2,3],[10,20,30]]
a[2][3]
a = [[1,2,3] [10,20,30]]
a[3,2]
a = [1 2 3; 4 5 6; 7 8 9]
a[1:2,:]
selectdim(a,2,[1,2])
a = Union{Int64,Float64}[1,2,3]; push!(a,1.5)
a = [[1,2,3] [10,20,30]]
mask = [[true,true,false] [false,true,false]]
a[mask]
a = [1 2]
a[2]
a[1,2]
a=[ 1 2 3; 4 5 6]
size(a)
ndims(a)
reshape(a, 3, 2)
a = rand(2,1,3); a = dropdims(a,dims=(2))
transpose(a)
a'
permutedims(a)
sortslices(a,dims=1,by=x->(x[2],x[1]),rev=true)
a = [[1 2 3]; [4 5 6]]
reshape(a,3,2)
transpose(a)
@test collect(Iterators.flatten(a)) == vec(a) == a[:]
col1 = [1,2,3]; col2 = [4,5,6]
row1 = [1 2 3]; row2 = [4 5 6]
array1 = rand(3,4,2); array2 = rand(3,4,3)
hcat(col1, col2)
hcat(col1, col2)
n=3
cat(array1, array2, dims=n)

# ------------------------------------------------------------------------------
# Tuples
t = (1,2.5,"a")
t = 1,2.5,"a"
typeof((1,2.5,"a")) == Tuple{Int64,Float64,String}
t=(1,2,3);
@test [t...] == [i[1] for i in t] == collect(t)
t = (a...,)

# ------------------------------------------------------------------------------
# Named Tuples

nt = (a=1, b=2.5)
nt.a
keys(nt)
values(nt)
collect(nt)
pairs(nt)
for (k,v) in pairs(nt) end

# ------------------------------------------------------------------------------
# Dictionaries
mydict = Dict()
mydict = Dict('a'=>1, 'b'=>2, 'c'=>3)
akey = 'd'
avalue = 4
mydict[akey] = avalue
akey = 'b'
delete!(mydict,akey)
map((i,j) -> mydict[i]=j, ['a','b','c'], [1,2,3])
mydict['a']
get(mydict,'a',0)
keys(mydict)
values(mydict)
haskey(mydict, 'a')
in(('a' => 1), mydict)
for (k,v) in mydict
    println("$k is $v")
end
mykeys = [k for (k,v) in mydict if v==2]
d = Dict(:k1=>"v1", :k2=>2) 
nt = (k1="v1", k2=2)
nt.k1
d[:k1]

# ------------------------------------------------------------------------------
# Sets
s = Set()
s = Set{T}()
s = Set([1,2,2,3,4])
push!(s, 5)
delete!(s,1)
set1 = Set([1,3,5]); set2 = Set([3,7])
intersect(set1,set2)
union(set1,set2)
setdiff(set1,set2)

# ------------------------------------------------------------------------------
# DateTime
using Dates
todayDate = today()
nowTime = now()
nowTimeUTC = now(Dates.UTC)
nowTimeUnix = time()
nowTime = Dates.unix2datetime(nowTimeUnix)
christmasLunch = DateTime("2030-12-25T12:30:00", ISODateTimeFormat)
newYearEvenDinner = DateTime("Sat, 30 Dec 2030 21:30:00", RFC1123Format)
christmasDay = Date("25 Dec 2030", "d u yyyy")
newYearDay = Date("2031/01/01", "yyyy/m/d")
d = Date(2030, 12)
dt = DateTime(2030, 12, 31, 9, 30, 0, 0)

Dates.format(newYearDay, "dd/m/yy")
Dates.format(christmasLunch, "dd/mm/yy H:M:SS")

year(christmasDay)
isleapyear(christmasDay)
month(christmasLunch)
monthname(christmasDay)
day(christmasDay)
dayofmonth(christmasDay)
dayname(christmasDay)
dayofweek(christmasDay)
daysofweekinmonth(christmasDay)
dayofweekofmonth(christmasDay)
hour(christmasLunch)
minute(christmasLunch)
second(christmasLunch)
millisecond(christmasLunch)
hollidayPeriod = newYearDay - christmasDay
mealPeriod = DateTime(2030,12,31,23,30) - newYearEvenDinner
convert(DateTime,newYearDay)

nextChristmas  = christmasDay + Year(1)
christmasPresentsOpeningTime = christmasLunch + Hour(3)
a_compounded_period = Week(1) + Day(2)
a_period = mealPeriod
Dates.value(a_period)
Dates.periods(a_compounded_period)
semesters = Dates.Date(2020,1,1):Dates.Month(6):Dates.Date(2022,1,1)
collect(semesters)

# ------------------------------------------------------------------------------
# Memory and copying issues

a = [[[1,2],3],4]
b = a
c = copy(a)
d = deepcopy(a)
# rebinds a[2] to an other objects.
# At the same time mutates object a:
a[2] = 40
b
c
d
# rebinds a[1][2] and at the same
# time mutates both a and a[1]:
a[1][2] = 30
b
c
d
a[1][1][2] = 20
b
c
d
# rebinds a:
a = 5
b
c
d
a = [1, 2]; b = [1, 2];
@test (a == b) == true
@test (a === a) == true
@test (a === b) == false
a = (1, 2); b = (1, 2);
@test (a == b) == true
@test (a === a) == true
@test (a === b) == true

# ------------------------------------------------------------------------------
# Various notes on data types
x = 1.0
T = Int64
convertedObj = convert(T,x)
const x2 = 5
myNewList = parse.(Float64,["1.1","1.2"])

# ------------------------------------------------------------------------------
# Variable references

x = [1,2,3]
y = Ref(x) 
x[2] = 20
y[] 
x = [4,5,6]
y[]

# ------------------------------------------------------------------------------
# Random numbers

rand()
rand(Float32,3)
a = 1; b = 10
rand(a:b)
rand(a:0.01:b)
using Distributions
rand(Uniform(a,b))
import Random:seed!; seed!(1234)
seed!()

using StableRNGs
my_stable_rng = StableRNG(123)
a = rand(my_stable_rng,10)
b = rand(my_stable_rng,10)
my_stable_rng = StableRNG(123)
c = rand(my_stable_rng,10)
b == a # false, it continues the same stream
c == a # true, both are the beginning of identical streams

# ------------------------------------------------------------------------------
# Missing, nothing and N/A 
nothing
missing
0/0
1/0
-1/0

array = [1,missing,2,3]
skipmissing(array)
T = Float64
nonmissingtype(Union{T,Missing})
using Missings
array = [1,2,3]
allowmissing(array)
disallowmissing(array)
