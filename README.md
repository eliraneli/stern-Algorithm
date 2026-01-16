# Stern's Algorithm for Information Set Decoding

## 📌 Overview

This repository contains a high-performance Python implementation of **Stern’s Algorithm** (specifically the **Canteaut–Chabaud 1998** variant). 

The algorithm is designed to solve the **Minimum Weight Codeword Problem**: finding a codeword of a specific (small) Hamming weight $\( w \)$ in a binary linear code. This implementation leverages **PyTorch** for vectorized bitwise operations and **Python Multiprocessing** to parallelize the probabilistic search across many CPU cores.

---

## 🧠 The Problem Domain

### What is a Linear Code?
A binary linear code $\( C \)$ of length $\( n \)$ and dimension $\( k \)$ is a subspace of \( $\mathbb{F}_2^n \$). It can be defined by a **Generator Matrix** $\( G \)$ (size $\( k \times n \$)). Any codeword $\( c \in C \)$ is a linear combination of the rows of $\( G \)$:
$\[ c = u \cdot G \]$
where $\( u \)$ is an information vector of length $\( k \)$.

### The "Hard" Problem
The core problem this algorithm solves is:
> **Given a random linear code (defined by $G$) and an integer $\( w \)$, find a non-zero codeword $\( c \)$ such that the Hamming weight of $\( c \)$ is $\( w \)$ (or $\le w$).**

### Why is this NP-Complete?
This problem is equivalent to the **Syndrome Decoding Problem (SDP)**, which was proven to be **NP-Complete** by Berlekamp, McEliece, and van Tilborg (1978).

1.  **Linear Constraints are Easy:** Solving for $\( c \)$ such that $\( c \cdot H^T = 0 \)$ (where $\( H \)$ is the parity-check matrix) is trivial using Gaussian elimination. This gives you the null space.
2.  **Weight Constraints are Hard:** Adding the constraint "Hamming weight $\( = w \)$" breaks the linear structure. You are looking for a specific vector in the null space that is extremely sparse (mostly zeros).
3.  **Search Space:** Because you cannot easily "calculate" where the sparse vectors are, you are forced into a combinatorial search. As $\( n \)$ and $\( k \)$ grow, the number of combinations explodes exponentially.

**Significance:** This hardness assumption underpins **Post-Quantum Cryptography** (e.g., the **McEliece** cryptosystem). If you can solve this problem efficiently, you can break these cryptosystems.

---

## ⚙️ How It Works (The Algorithm)

This implementation uses **Information Set Decoding (ISD)**. The general strategy is:

1.  **Permutation:** Randomly permute the columns of the matrix to change which indices are considered "information" vs "redundancy".
2.  **Systematic Form:** Perform Gaussian elimination to get a systematic matrix $\( G_{sys} = [I_k | Z] \)$.
3.  **Meet-in-the-Middle (Stern's Step):** * Split the information rows into two sets.
    * Generate short linear combinations for both sets.
    * Use a **collision search** (hashing/sorting) to find pairs from both sets that "cancel out" partially in the redundancy block $\( Z \)$, leaving a vector of low weight.
4.  **Verification:** If a candidate matches the target weight $\( w \)$, it is output.

**Optimization:** This code implements **Proposition 2** from Canteaut & Chabaud (1998), which uses a double-partitioning strategy (`p=2`) to improve the probability of finding a collision compared to standard Stern (`p=1`).

---

## 📦 Installation

### Setup
```bash
git clone [https://github.com/eliraneli/stern-algorithm.git](https://github.com/eliraneli/stern-algorithm.git)
cd stern-algorithm
pip install -r requirements.txt
```

## 🚀 Usage

The script accepts a **Generator Matrix** ($G$) as input (stored as a `.npy` file) and searches for codewords of a specified Hamming weight.

### Basic Command
```bash
python stern_algorithm.py \
  --hfile matrices/my_generator_matrix.npy \
  --w 32 \
  --p 2 \
  --ell 14 \
  --num_processes 40 \
  --out results/found_codewords.npy
```

### 🔧 Arguments Explained

| Parameter | Description | Typical Value |
| :--- | :--- | :--- |
| `--hfile` | Path to the binary Generator Matrix ($k \times n$) saved in `.npy` format. | `matrix.npy` |
| `--w` | Target codeword weight to find. | `32` |
| `--p` | Partition parameter. <br>• `1`: Standard Stern algorithm.<br>• `2`: Canteaut–Chabaud optimization (Double partitioning). | `2` |
| `--ell` | Subset size ($\ell$) for list generation. Controls collision window size. | `14–18` |
| `--max_iters` | Number of random iterations. | `5000` |
| `--num_processes`| Number of CPU cores to use for parallel processing. | `40–80` |
| `--out` | Output `.npy` file for saving codewords. | `found.npy` |

---

### 📊 Output Format

The algorithm saves the results to the file specified by `--out`. 

* **File Type:** Binary NumPy array (`.npy`).
* **Data Type:** `uint8` (Matrix of 0s and 1s).
* **Shape:** `(N, n)`
    * $N$: The number of unique codewords found.
    * $n$: The length of the code (number of columns in your input matrix).

**How to read the results in Python:**

```python
import numpy as np

# Load the output file
codewords = np.load("results/found_codewords.npy")

print(f"Algorithm found {codewords.shape[0]} unique codewords.")

# Print the first codeword
print("First codeword found:", codewords[0])
```

### 📚 References

This implementation is based on the following papers:

1.  **Stern's Original Method:**
    > J. Stern, *"A method for finding codewords of small weight,"* in Coding Theory and Applications, 1989.

2.  **Canteaut–Chabaud Improvement (Proposition 2):**
    > N. Canteaut and F. Chabaud, *"A new algorithm for finding minimum-weight words in a linear code: Application to McEliece’s cryptosystem and to narrow-sense BCH codes of length 511,"* IEEE Transactions on Information Theory, vol. 44, no. 1, pp. 367–378, Jan. 1998.  
    > [DOI: 10.1109/18.651013](https://doi.org/10.1109/18.651013)

---

## 📜 License

This project is licensed under the MIT License

