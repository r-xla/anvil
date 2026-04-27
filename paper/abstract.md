Bioinformatics workflows in R — fitting models across thousands of genes, training neural networks on omics features, simulating population dynamics — often hit the limits of the interpreter and get rewritten in C++ or Python.
{anvl} lets users write such computations in plain R and JIT-compile them with the OpenXLA compiler stack, with CPU/GPU execution and first-order automatic differentiation.
A core goal of the package is to offer a low-barrier entry point for R users to leverage GPU acceleration without leaving the language or learning a new framework.
The poster presents the package architecture, a [bio example] benchmark against [baseline], and a worked example of adding a new primitive — all in R, with no C++ required.
This makes {anvl} a practical target for Bioconductor developers who want compiled performance without leaving the language.
Source: <https://github.com/r-xla/anvl>
