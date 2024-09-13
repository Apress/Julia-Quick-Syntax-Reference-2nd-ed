################################################################################
# Code snippets of the book:
# A. Lobianco (2024) Julia Quick Syntax Reference - A Pocket Guide for Data
# Science Programming, Apress, Ed 2
# https://doi.org/[TODO_DOI]
# Licence: Open Domain
################################################################################

# Chapter 13 : Utilities

using Pkg
cd(@__DIR__)
Pkg.activate(".")
# If using a Julia version different than 1.10 or 1.11 please uncomment and run the following line (reproductibility guarantee will hower be lost). If your version of Julia can't find a suitable combination, try running `Pkg.up();Pkg.resolve()`
#Pkg.resolve()
Pkg.instantiate()
using Random
Random.seed!(123)

# ------------------------------------------------------------------------------
# Weave for dynamic documents

using Weave;

write("test_weave.jmd","""
    ---
    title :       Test of a document with embedded Julia code and citations
    date :        18 August 2024
    bibliography: biblio.bib
    ---
    ​

    ```{julia;eval=true,echo=false,results="hidden"}
    # (leave two rows from the document headers above)
    # This code is hidden in the output - both the code and its output.
    # You can use to initialize the script, here for example to install the packages
    # that the script requires 
    using Pkg
    Pkg.add(["Plots","DataFrames"])
    ```

    # Section 1

    Weave.jl, annunced in @Pastell:2017, is a scientific report generator/literate programming tool for Julia developed by Matti Pastell, resembling  Knitr for R [see @Xie:2015].

    ## Subsection 1.1
    ​
    This should print a plot. Note that, with `echo=false`, you are not rendering the source code in the final PDF:

    ```{julia;echo=false}
    using Plots
    plot(sin, -2pi, pi, label="sine function")
    ```

    Here instead you will render in the PDF both the script source code and its output:

    ```{julia;}
    using DataFrames
    df = DataFrame(
            colour = ["green","blue","white","green","green"],
            shape  = ["circle", "triangle", "square","square","circle"],
            border = ["dotted", "line", "line", "line", "dotted"],
            area   = [1.1, 2.3, 3.1, missing, 5.2]
        )
    df
    ```

    Note also that you can refer to variables defined in previous chunks (or "cells", following Jupyter terminology):

    ```{julia;}
    df.colour
    ```

    ### Subsubsection

    For a much more complete example see the [Weave documentation](http://weavejl.mpastell.com/stable/).

    # References
    """)

write("biblio.bib",
  """
  @article{   Pastell:2017,
    author  = {Pastell, Matti},
    title   = {Weave.jl: Scientific Reports Using Julia},
    journal = {Journal of Open Source Software},
    vol     = {2},
    issue   = {11},
    year    = {2017},
    doi     = {10.21105/joss.00204}
  }
  @Book{       Xie:2015,
    title     = {Dynamic Documents with R and Knitr.},
    publisher = {Chapman and Hall/CRC},
    year      = {2015},
    author    = {Yihui Xie},
    edition   = {2nd ed},
    url       = {http://yihui.name/knitr/},
  }
  """)

weave("test_weave.jmd", out_path = :pwd)  # HTML, needs pandoc / pandoc-citeproc 
weave("test_weave.jmd", out_path = :pwd, doctype = "pandoc2pdf") # PDF, needs pandoc / pandoc-citeproc / latex packages


# ------------------------------------------------------------------------------
# ZipFile

using ZipFile
zf = ZipFile.Writer("example.zip")
f1 = ZipFile.addfile(zf, "file1.txt", method=ZipFile.Deflate)
write(f1, "Hello world1!\n")
write(f1, "Hello world1 again!\n")
f2 = ZipFile.addfile(zf, "dir1/file2.txt", method=ZipFile.Deflate)
write(f2, "Hello world2!\n")
close(zf) # Important!

zf = ZipFile.Writer("example2.zip")
f1 = ZipFile.addfile(zf, "file1.txt", method=ZipFile.Deflate)
f2 = ZipFile.addfile(zf, "dir1/file2.txt", method=ZipFile.Deflate)
# write(f1, "Hello world1!\n") # Error !
close(zf)

zf = ZipFile.Reader("example.zip");
for f in zf.files
    println("*** $(f.name) ***")
    for ln in eachline(f) # Alternative: read(f,String) to read the whole file
        println(ln)
    end
end
close(zf) # Important!


zf = ZipFile.Reader("example.zip");
for f in zf.files
    println("*** Name: $(f.name)")
    println("- Method: $(f.method)")
    println("- Uncompressd size: $(f.uncompressedsize)")
    println("- Compressed size: $(f.compressedsize)")
end
show(zf)
close(zf) # Important!

# ------------------------------------------------------------------------------
# Interact and Mux: Expose Interacting Models on the Web

# Unfortunatly the last example on the first edition of the book concerning
# exposing your model on the web is broken in recent versions of the ecosystem
# or Julia
# You can find the original example in the folder "mux_interact_version" and
# another version in Genie, that has equally its own problems, at least using
# julia 1.11-rc2, under the "genie_version" folder.