# Stern Algorithm (Canteaut–Chabaud 1998)

## Overview

This repository implements a **probabilistic Information Set Decoding (ISD)** algorithm based on **Stern’s algorithm** and its improved version by **Canteaut & Chabaud (1998)** — adapted for GPU acceleration using **PyTorch**.

The goal of ISD algorithms is to **find codewords of small Hamming weight** in a binary linear code defined by a generator or parity-check matrix.  
Such algorithms are critical for studying the security of code-based cryptosystems (like **McEliece**), and for analyzing error-correcting codes such as BCH and LDPC codes.

---

## 📘 Background

### Information Set Decoding (ISD)
ISD algorithms aim to solve the following problem:

> Given a binary linear code \( C \subseteq \mathbb{F}_2^n \) of dimension \( k \),  
> find a codeword \( c \in C \) of **weight ≤ w**.

Finding such codewords is NP-hard (equivalent to the **Syndrome Decoding Problem**), but ISD provides a **probabilistic approach** that can find small-weight codewords much faster than exhaustive search.

The general ISD idea (first proposed by Prange, 1962) is:
1. Choose a random information set (subset of coordinates of size \(k\)).
2. Assume the restriction of the generator matrix to this set is invertible.
3. Decode with respect to that set to reconstruct a potential low-weight codeword.
4. Repeat randomly until a small-weight codeword is found.

---

## 📖 Algorithm Reference

This implementation follows **Proposition 2** from the paper:

> N. Canteaut and F. Chabaud,  
> *A new algorithm for finding minimum-weight words in a linear code: Application to McEliece’s cryptosystem and to narrow-sense BCH codes of length 511*,  
> IEEE Transactions on Information Theory, vol. 44, no. 1, pp. 367–378, Jan. 1998.  
> [DOI: 10.1109/18.651013](https://doi.org/10.1109/18.651013)

The algorithm improves upon Stern’s original ISD method by:
- Introducing **double partitioning** (`p=2`) for better collision structure,
- Using **hash-based merging** of partial sums,
- Allowing **parallel computation** on GPU.

---

| Parameter     | Description                                           | Typical Value           |
| ------------- | ----------------------------------------------------- | ----------------------- |
| `--hfile`     | Path to parity-check matrix `.npy` file               | `BCH_127_92_PCM_CR.npy` |
| `--w`         | Target codeword weight to find                        | `32`                    |
| `--p`         | Partition parameter (1 = Stern, 2 = Canteaut–Chabaud) | `2`                     |
| `--ell`       | Subset size ( \ell ) for list generation              | `14–18` typical         |
| `--max_iters` | Number of random iterations                           | `10000–100000`          |
| `--out`       | Output `.npy` file for saving codewords               | `found_codewords.npy`   |


```bash
pip install -r requirements.txt
