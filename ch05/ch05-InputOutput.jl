################################################################################
# Code snippets of the book:
# A. Lobianco (2024) Julia Quick Syntax Reference - A Pocket Guide for Data
# Science Programming, Apress, Ed 2
# https://doi.org/[TODO_DOI]
# Licence: Open Domain
################################################################################

# Chapter 5 : Input-Output

using Pkg
cd(@__DIR__)
Pkg.activate(".")
# If using a Julia version different than 1.10 or 1.11 please uncomment and run the following line (reproductibility guarantee will hower be lost). If your version of Julia can't find a suitable combination, try running `Pkg.up();Pkg.resolve()`
#Pkg.resolve()
Pkg.instantiate()
using Test

# ------------------------------------------------------------------------------
# Filesystem functions

# Note: this should be os-specific (Linux/Mac in this example)
dir_to_list    = "/etc"
command_to_run = `ls $dir_to_list`
run(command_to_run)

# Note: this should be os-independent, except when subfolder separator is used and joinpath() is not used:
pwd()
readdir(".")
walkdir(".")
ispath("foo")
isfile("test.jl")
isdir("test")
isabspath("/path/to/file.txt")
basename("/path/to/file.txt")
dirname("/path/to/file.txt")
realpath("myinputfile.ods")
cd(@__DIR__)
mkdir("foodir")
mkpath("goodir/foo/baa")
joinpath("aaa","bbb")
rm("foo.txt",force=true,recursive=true)
mv("goodir","newpath")
cp("newpath","alternativepath")

# ------------------------------------------------------------------------------
# Retrive user input

# Put the above script in a julia file, e.g. "command_line_input_processing.jl":
println("Welcome to a julia script.")
if length(ARGS)>0
  println("You used this script with the following arguments:")
  for arg in ARGS
    println("- $arg")
  end
end
function getUserInput(T=String,msg="")
  print("$msg ")
  if T == String
      return readline()
  else
    try
      return parse(T,readline())
    catch
     println("Sorry, I could not interpret your answer. Please try again")
     getUserInput(T,msg)
    end
  end
end
# sentence = getUserInput(String,"Which sentence do you want to be repeated?"); # run this on a terminal 
# n        = getUserInput(Int64,"How many times do you want it to be repeated?"); # run this on a terminal 
# [println(sentence) for i in 1:n]
println("Done!")

# Now call the above script form comand line as `julia command_line_input_processing.jl whatever arguments`

# ------------------------------------------------------------------------------
# Read files

open("afile.txt", "r") do f   # "r" for reading
    filecontent = read(f,String) # attention that it can be used only once. If used a second time, without reopening the file, read() would return an empty string
    # ...operate on the whole file content all at once...
end

open("afile.txt", "r") do f
    for ln in eachline(f)
      println(ln)
      # ... operate on each individual line at a time...
    end
end

# ------------------------------------------------------------------------------
# CSV, XLSX and ODS import

using DelimitedFiles, CSV, DataFrames, XLSX, OdsIO
my_data = convert(Array{Int64},readdlm("myinputfile.csv",',')[2:end,3:end])

my_data = CSV.read("myinputfile.csv", DataFrame,delim=",\t",ignorerepeated=true,types=Dict("Field2" => Union{Missing,Int64}))

XLSX.sheetnames(XLSX.readxlsx("myinputfile.xlsx"))
XLSX.readxlsx("myinputfile.xlsx")["Sheet1"][:]
XLSX.readxlsx("myinputfile.xlsx")["Sheet1"]["B2:D6"]
XLSX.readdata("myinputfile.xlsx", "Sheet1", "B2:D6")
DataFrame(XLSX.readtable("myinputfile.xlsx", "Sheet1"))
DataFrame(XLSX.readtable("myinputfile.xlsx", "Sheet1", "B:D", first_row=2))
ods_read("myinputfile.ods"; sheetName="Sheet1", range=((2,2),(6,4)), retType="DataFrame")

# ------------------------------------------------------------------------------
# JSON import

using JSON3
json_string="""
{
    "lat": 53.204672,
    "long": -1.072370,
    "sp": "Oak",
    "trees": [
                  {
                     "vol": 23.54,
                      "id": 1
                  },
                  {
                     "vol": 12.25,
                      "id": 2
                  }
                ]
}
"""
struct ForestStand
    sp::String
    lat::Float64
    long::Float64
    trees::Array{Dict{String,Float64},1}
end
nott_for = JSON3.read(json_string, ForestStand)
nott_for2 = JSON3.read(json_string)
nott_for2.trees[1].vol

# ------------------------------------------------------------------------------
# Web resources
using Downloads, HTTP, Pipe

data_url ="https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data"
Downloads.download(data_url,"data.csv")

res         = HTTP.get(data_url)
data_binary = res.body
data_string = String(copy(res.body))
data        = CSV.read(data_binary, DataFrame, delim=" ", ignorerepeated=true, header=false)
url_data = "https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data-original"
data = @pipe HTTP.get(url_data).body              |>
       replace!(_, UInt8('\t') => UInt8(' '))     |> 
       CSV.File(_, delim=' ', missingstring="NA",
       ignorerepeated=true, header=false)         |>
       DataFrame;


# ------------------------------------------------------------------------------
# Writing to the terminal

write(stdout, "Hello World");
print("Hello World")
println("Hello World")

# Custom struct display/print
struct ACustomType
    x::Int64
    y::Float64
    z::String
end
foo2 = ACustomType(1,2,"MyObj") # Output: ACustomType(1, 2.0, "MyObj")
println(foo2) # Output: ACustomType(1, 2.0, "MyObj")
# this would error: write(stdout,foo2) # Output: MethodError
import Base.show # Important ! Required to extend the `Base.show` methods
function show(io::IO, ::MIME"text/plain", m::ACustomType)
    print(io,"A brief custom representation of ACustomType")
end
function show(io::IO, m::ACustomType)
    println(io,"An extended custom representation of ACustomType")
    println("($(m.x) , $(m.y)): $(m.z))")
end
foo2 # Output: A brief custom representation of ACustomType
println(foo2) # Output: An extended custom representation of ACustomType\n (1 , 2.0): MyObj)
# this would still error: write(stdout, foo2) # Output: still MethodError

# ------------------------------------------------------------------------------
# Write to file
open("afile.txt", "w") do f  # "w" for writing
    write(f, "First line\n")   # \n for newline
    println(f, "Second line")  # Newline automatically added by println
end
write("afile.txt", "First line\nSecond line")

# ------------------------------------------------------------------------------
# Export to CSV, Excel, ODS

# Export to CSV
my_matrix = Matrix(my_data)
DelimitedFiles.writedlm("myOutputFile.csv", my_matrix,";")
CSV.write("myOutputFile.csv", my_data, delim=';', decimal='.', missingstring="NA")

CSV.write("myOutputFile.csv", CSV.Tables.table(my_matrix), delim=';', decimal='.', missingstring="NA")

# Export to Excel
XLSX.openxlsx("myExcelFile.xlsx", mode="w") do xf # w to write (or overwrite) the file
    sheet1 = xf[1]  # One sheet is created by default
    XLSX.rename!(sheet1, "new sheet 1")
    sheet2 = XLSX.addsheet!(xf, "new sheet 2") # We can add further sheets if needed
    sheet1["A1"] = "Hello world!"
    sheet2["B2"] = [ 1 2 3 ; 4 5 6 ; 7 8 9 ] # Top-right cell to anchor the matrix
end
XLSX.openxlsx("myExcelFile.xlsx", mode="rw") do xf # rw to append to an existing file instead
    sheet1 = xf[1]  # One sheet is created by default
    sheet2 = xf[2]
    sheet3 = XLSX.addsheet!(xf, "new sheet 3") # We can add further sheets if needed
    sheet1["A2"] = "Hello world again!"
    sheet3["B2"] = [ 10 20 30 ; 40 50 60 ; 70 80 90 ] # Top-right cell to anchor the matrix
end

using DataFrames
my_df = DataFrame(f1=[1,2,3], f2=[4,5,6], f3=[7,8,9])
rm("myNewExcelFile.xlsx",force=true)
XLSX.writetable("myNewExcelFile.xlsx", my_df)
rm("myNewExcelFile.xlsx")
XLSX.writetable("myNewExcelFile.xlsx", sheet1=( [ [1, 2, 3], [4,5,6], [7,8,9]], ["f1","f2","f3"] ), sheet2=(collect(DataFrames.eachcol(my_df)), DataFrames.names(my_df) ))

# Exporting to ODS
ods_write("TestSpreadsheet.ods",Dict(
    ("TestSheet",3,2)=>[[1,2,3,4,5] [6,7,8,9,10]],
    ("TestSheet2",1,1)=>["aaa";;],
    ))

# Exporting to JSON
using JSON3
json_string =  JSON3.write(nott_for)
JSON3.@pretty json_string