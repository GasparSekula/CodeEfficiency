# ⚡ Code Efficiency in the Era of Many-Core CPUs

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![NumPy](https://img.shields.io/badge/NumPy-optimized-013243?logo=numpy)
![Numba](https://img.shields.io/badge/Numba-JIT-00A3E0)
![CuPy](https://img.shields.io/badge/CuPy-GPU-76B900)

> *With the recent development of powerful CPUs, is it still necessary to write highly efficient code?*

This repository contains the codebase, benchmark results, and final research report developed as part of a high-performance computing research challenge at the **Technical University of Munich (TUM)**. 

The project investigates whether hardware abundance mitigates the need for software efficiency by benchmarking various matrix multiplication ($\mathcal{O}(n^{3})$) implementations on an HPC cluster.

---

## 🚀 The Problem

The recent spread of high-performance, many-core processors has fostered the misconception that hardware abundance mitigates the need for software efficiency. This project challenges the "hardware-first" paradigm. 

By analyzing matrix multiplication across different scales and core configurations, this research demonstrates that **unoptimized code leaves most of modern CPU throughput inaccessible.**

### Key Findings 📊
* **70,000x Performance Gap:** At $N=500$ on 16 CPU cores, a naive implementation yielded 0.0047 GFLOPS, while NumPy achieved ~330 GFLOPS on the exact same hardware.
* **Super-linear Scaling:** NumPy recorded an 11.5x speedup when doubling CPU cores from 8 to 16, highlighting the massive impact of cumulative L3 cache and memory channels.
* **Hardware Saturation:** Powerful CPUs act merely as a latent resource. Naive implementations remain fundamentally incapable of scaling, regardless of core count.

---

## 🧰 Methodology

We benchmarked four distinct approaches to matrix computation:
1. **Naive Python:** A standard triple-nested loop, heavily bottlenecked by the Python interpreter and Global Interpreter Lock (GIL).
2. **NumPy:** Wrappers for highly optimized C/Fortran libraries (OpenBLAS/MKL) utilizing cache tiling.
3. **Numba:** LLVM-based Just-In-Time (JIT) compilation bypassing the GIL for parallel execution.
4. **CuPy (Baseline):** GPU-accelerated execution to establish an absolute performance ceiling.

*Tests were executed on an NVIDIA DGX A100 system using dual AMD Rome 7742 processors.*

---

## 📁 Repository Structure

The repository is organized as follows:

```text
.
├── analysis/          # Scripts and notebooks for generating GFLOPS & Scaling plots
├── results/           # Raw benchmark logs and execution times (CSV/JSON)
├── src/               # Source code for the 4 multiplication implementations
├── README.md          # Project overview
├── report.pdf         # The comprehensive research paper detailing findings
└── requirements.txt   # Python dependencies required to run the benchmarks
```
---

## 📖 Read the Full Paper
For an in-depth analysis of temporal overhead, memory bottlenecks, statistical hypothesis testing (Mann-Whitney U test), and detailed hardware specifications, please refer to the full [`report.pdf`](./report.pdf) included in this repository.

---
*Developed by Gaspar Sekula.*