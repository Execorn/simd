# SIMD

# 1. What is this project about?
This repository provides 2 educational examples of optimizations using SSE/AVX.
## What is SIMD, SSE/AVX?
### SIMD
Single Instruction, Multiple Data (SIMD) units refer to hardware components that perform the same operation on multiple data operands concurrently. Typically, a SIMD unit receives as input two vectors (each one with a set of operands), performs the same operation on both sets of operands (one operand from each vector), and outputs a vector with the results.
### SSE
In computing, Streaming SIMD Extensions (SSE) is a single instruction, multiple data (SIMD) instruction set extension to the x86 architecture, designed by Intel and introduced in 1999 in their Pentium III series of CPU's. SSE contains 70 new instructions (65 unique mnemonics using 70 encodings), most of which work on single precision floating-point data. SIMD instructions can greatly increase performance when exactly the same operations are to be performed on multiple data objects. Typical applications are digital signal processing and graphics processing. 
### AVX
Advanced Vector Extensions (AVX) are extensions to the x86 instruction set architecture for microprocessors from Intel and Advanced Micro Devices (AMD). AVX provides new features, new instructions and a new coding scheme. AVX2 (also known as Haswell New Instructions) expands most integer commands to 256 bits and introduces new instructions. They were first supported by Intel with the Haswell processor, which shipped in 2013. 

# 2. Testing it out
### 1st example: Drawing Mandelbrot set with SFML using SSE. 2 examples (with, and without optimizations) and makefile are provided.
### 2nd example: Image overlay. Program overlays one image onto another using bit operations and SIMD instructions. Makefile provided (for windows only so far).