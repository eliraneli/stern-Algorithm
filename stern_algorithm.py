import copy
import time

import torch, random
import numpy as np
import argparse
from typing import Optional, List, Tuple
import multiprocessing as mp


# ------------------ Utilities ------------------

def to_systematic_form(G: torch.Tensor, device=None) -> Tuple[torch.Tensor, List[int]]:
    """Gaussian elimination to systematic form, returns (G_sys, col_perm)."""
    if device is None: device = G.device
    Gcur = G.clone().to(torch.uint8).to(device)
    k, n = Gcur.shape
    col_perm = list(range(n))
    row = 0
    for col in range(n):
        if row >= k: break
        pivot = None
        for r in range(row, k):
            if Gcur[r, col].item() == 1:
                pivot = r;
                break
        if pivot is None: continue
        if pivot != row:
            Gcur[[pivot, row]] = Gcur[[row, pivot]]
        ones = (Gcur[:, col] == 1)
        for r in range(k):
            if r != row and ones[r]:
                Gcur[r] ^= Gcur[row]
        if col != row:
            Gcur[:, [col, row]] = Gcur[:, [row, col]]
            col_perm[col], col_perm[row] = col_perm[row], col_perm[col]
        row += 1
    if row != k: raise RuntimeError("Rank < k")
    return Gcur, col_perm


def compute_parity_check(Gsys: torch.Tensor) -> torch.Tensor:
    """
    Compute parity-check matrix H from systematic generator matrix Gsys = [I | Z].
    H = [-Z^T | I]
    """
    k, n = Gsys.shape
    Z = Gsys[:, k:]
    H_left = Z.T.clone() % 2  # (-Z^T) == Z^T in GF(2)
    H_right = torch.eye(n - k, dtype=torch.uint8, device=Gsys.device)
    return torch.cat([H_left, H_right], dim=1)  # shape (n-k, n)


def is_codeword_syndrome(c: torch.Tensor, H: torch.Tensor) -> bool:
    """Check if c is in the code: syndrome c*H^T == 0."""
    s = (c.to(torch.float32) @ H.T.to(torch.float32)).to(torch.int64) & 1
    return bool((s == 0).all())


def hamming_weight_rows(mat: torch.Tensor) -> torch.Tensor:
    return mat.to(torch.int32).sum(dim=-1)


def pack_bits_to_uint64(x: torch.Tensor) -> torch.Tensor:
    B = x.shape[-1]
    if B > 64: x = x[..., :64]; B = 64
    weights = (1 << torch.arange(B, dtype=torch.int64, device=x.device))
    return (x.to(torch.int64) * weights).sum(dim=-1)


# ------------------ Main class ------------------

class SternIterativeGPU:
    def __init__(self, G: torch.Tensor, device=None):
        if device is None:
            device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
            print(device)
        self.device = device
        self.k, self.n = G.shape
        Gsys, col_perm = to_systematic_form(G, device=device)
        self.Gsys = Gsys
        self.H = compute_parity_check(self.Gsys)  # store parity-check matrix
        self.col_perm = col_perm  # maps current columns -> original indices
        self.n_redundant = self.n - self.k
        self._refresh_views()

    def _refresh_views(self):
        self.Z = self.Gsys[:, self.k:]  # redundancy block view

    # ---------- Proposition-2 swap ----------
    def proposition2_swap(self, info_col_pos: int, redundant_col_pos: int):
        r = info_col_pos
        col_red_full = self.k + redundant_col_pos
        col_info_full = r
        if self.Gsys[r, col_red_full].item() == 0:
            pivots = (self.Gsys[:, col_red_full] == 1).nonzero(as_tuple=False).view(-1)
            if pivots.numel() == 0:
                return  # skip bad column
            s = int(pivots[0])
            if s != r:
                self.Gsys[[r, s]] = self.Gsys[[s, r]]
        rows_to_clear = (self.Gsys[:, col_red_full] == 1).nonzero(as_tuple=False).view(-1).tolist()
        rows_to_clear = [x for x in rows_to_clear if x != r]
        if rows_to_clear:
            rows_tensor = torch.tensor(rows_to_clear, device=self.device)
            self.Gsys[rows_tensor] ^= self.Gsys[r].unsqueeze(0)
        self.Gsys[:, [col_info_full, col_red_full]] = self.Gsys[:, [col_red_full, col_info_full]]
        self.col_perm[col_info_full], self.col_perm[col_red_full] = (
            self.col_perm[col_red_full], self.col_perm[col_info_full]
        )
        self._refresh_views()

    def iteration_once(self, w: int, p: int = 1, ell: int = 16):
        k = self.k
        nR = self.n_redundant
        Z = self.Z
        device = self.device

        # --- Random selection ---
        perm_rows = torch.randperm(k, device=device)
        rows1, rows2 = perm_rows[:k // 2], perm_rows[k // 2:]
        if ell > nR:
            ell = nR
        L_idx = torch.randperm(nR, device=device)[:ell]
        JL_mask = torch.ones(nR, dtype=torch.bool, device=device)
        JL_mask[L_idx] = False

        # --- Build candidate vectors ---
        if p == 1:
            V1_full, V2_full = Z[rows1], Z[rows2]
            comb1, comb2 = V1_full[:, L_idx], V2_full[:, L_idx]
        elif p == 2:
            # Efficient XOR pair generation (avoid torch.combinations)
            def xor_pairs_block(Z_rows, block=256):
                n = Z_rows.size(0)
                out = []
                for i in range(0, n, block):
                    a = Z_rows[i:min(i + block, n)]
                    xor_block = a.unsqueeze(1) ^ Z_rows.unsqueeze(0)
                    # keep only (i < j) unique unordered pairs
                    triu_i, triu_j = torch.triu_indices(xor_block.size(0), xor_block.size(1), offset=1)
                    out.append(xor_block[triu_i, triu_j])
                if len(out) == 0:
                    return Z_rows.new_empty((0, Z_rows.size(1)))
                return torch.cat(out, dim=0)

            V1_full = xor_pairs_block(Z[rows1])
            V2_full = xor_pairs_block(Z[rows2])
            comb1, comb2 = V1_full[:, L_idx], V2_full[:, L_idx]
        else:
            raise NotImplementedError("Only p=1 or p=2 supported.")

        # --- Packing & sorting ---
        key1, key2 = pack_bits_to_uint64(comb1), pack_bits_to_uint64(comb2)
        keys = torch.cat([key1, key2])
        part = torch.cat([torch.zeros_like(key1), torch.ones_like(key2)])
        idxs = torch.arange(keys.size(0), device=device)
        sort_idx = torch.argsort(keys)
        keys_s, part_s, idxs_s = keys[sort_idx], part[sort_idx], idxs[sort_idx]

        if keys_s.numel() == 0:
            return None

        # --- Vectorized run detection (compute run start/end indices) ---
        # diff indicates boundaries where key changes
        diff = keys_s[1:] != keys_s[:-1]
        if diff.numel() == 0:
            starts = torch.tensor([0], device=device)
            ends = torch.tensor([keys_s.numel()], device=device)
        else:
            boundaries = torch.nonzero(diff, as_tuple=False).flatten() + 1  # positions where new run starts
            starts = torch.cat([torch.tensor([0], device=device), boundaries])
            ends = torch.cat([boundaries, torch.tensor([keys_s.numel()], device=device)])

        # --- For each run, do the original per-run logic with EARLY EXIT optimization ---
        target_weight = w - 2 * p

        for a, b in zip(starts.tolist(), ends.tolist()):
            block_parts = part_s[a:b]
            # must contain both parts 0 and 1 inside the run
            if not ((block_parts == 0).any() and (block_parts == 1).any()):
                continue
            run_idxs = idxs_s[a:b]
            left_rel = run_idxs[block_parts == 0]  # indices into global key array that belong to left part
            right_rel = run_idxs[block_parts == 1] - key1.size(0)  # adjusted indices for right part

            if left_rel.numel() == 0 or right_rel.numel() == 0:
                continue

            # Extract candidate rows exactly as original
            V1c, V2c = V1_full[left_rel], V2_full[right_rel]

            # *** OPTIMIZATION: Compute XORs row-by-row with early exit ***
            # Instead of computing all V1c × V2c pairs at once, process one V1c row at a time
            found_match = False
            left_idx = 0
            right_idx = 0
            redundancy = None

            for i in range(V1c.size(0)):
                # Compute XOR of one V1c row against all V2c rows
                xor_row = V1c[i].unsqueeze(0) ^ V2c  # Shape: (V2c.size(0), nR)
                wt = hamming_weight_rows(xor_row[:, JL_mask])
                matches = (wt == target_weight).nonzero(as_tuple=False)

                if matches.numel() > 0:
                    # Found a match! Extract indices and redundancy
                    left_idx = i
                    right_idx = matches[0, 0].item() if matches.dim() == 2 else matches[0].item()
                    redundancy = xor_row[right_idx]
                    found_match = True
                    break  # Early exit - no need to check remaining rows

            if not found_match:
                continue  # Try next run

            # --- Build and return the codeword (same as original) ---
            info_vec = torch.zeros(k, dtype=torch.uint8, device=device)

            if p == 1:
                # left_rel[left_idx] is the global index into keys/key1 region -> map to rows1
                which_left_global = left_rel[left_idx].item()
                which_right_global = run_idxs[block_parts == 1][right_idx].item()
                # left global index is guaranteed < key1.size(0)
                info_vec[rows1[which_left_global]] = 1
                info_vec[rows2[which_right_global - key1.size(0)]] = 1
            elif p == 2:
                # For p=2 case, the original code had pass here
                # If you need to implement this, add the logic
                pass

            codeword_perm = torch.cat([info_vec, redundancy])
            codeword_orig = torch.zeros(self.n, dtype=torch.uint8, device=device)
            codeword_orig[self.col_perm] = codeword_perm
            return codeword_orig.cpu().numpy()

        return None

    def run(self, w: int, p: int = 1, ell: int = 16, max_iters: int = 1000, num_processes: int = 4):
        # Calculate iterations per worker
        iters_per_worker = max_iters // num_processes
        remaining_iters = max_iters % num_processes

        # Distribute iterations among workers
        worker_iters_list = [iters_per_worker] * num_processes
        for i in range(remaining_iters):
            worker_iters_list[i] += 1

        # Prepare parameters for each worker
        params_list = [
            (iters, w, p, ell, self)
            for iters in worker_iters_list
        ]

        # Create pool and run parallel processes
        with mp.Pool(processes=num_processes) as pool:
            results = pool.map(worker_function, params_list)

        # Combine results from all workers
        found = []
        seen = set()
        for local_found, local_seen in results:
            for codeword in local_found:
                tup = tuple(codeword.tolist())
                if tup not in seen:
                    seen.add(tup)
                    found.append(codeword)

        return found


def worker_function(params):
    """Worker function for parallel processing"""
    worker_iters, w, p, ell, algo = params
    local_algo = copy.deepcopy(algo)
    local_found = []
    local_seen = set()

    for _ in range(worker_iters):

        res = local_algo.iteration_once(w, p=p, ell=ell)
        t = time.time()
        if res is not None:
            c = torch.tensor(res, dtype=torch.uint8, device=algo.device)
            if is_codeword_syndrome(c, algo.H):
                tup = tuple(res.tolist())
                if tup not in local_seen:
                    # print('found')
                    local_seen.add(tup)
                    local_found.append(res)

        # Proposition-2 swap every 10 iterations
        if _ % 10 == 0:
            red = random.randrange(local_algo.n_redundant)
            local_algo.proposition2_swap(
                info_col_pos=random.randrange(local_algo.k),
                redundant_col_pos=red
            )
        # print(time.time() - t)

    return local_found, local_seen


# ------------------ Example ------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stern iterative algorithm with Proposition-2 (CPU)")
    parser.add_argument("--hfile", type=str, required=True, help="Path to generator matrix .npy file")
    parser.add_argument("--w", type=int, required=True, help="Target weight")
    parser.add_argument("--p", type=int, default=2, choices=[1, 2], help="Combination size (1 or 2)")
    parser.add_argument("--ell", type=int, default=14, help="Subset size ell")
    parser.add_argument("--max_iters", type=int, default=5000, help="Max iterations")
    parser.add_argument("--num_processes", type=int, default=80, help="Number of CPU cores to use")
    parser.add_argument("--out", type=str, required=True, help="Output file (.npy) to save found codewords")
    args = parser.parse_args()

    device = torch.device("cpu")
    G = torch.tensor(np.load(args.hfile), dtype=torch.uint8, device=device)
    k, n = G.shape
    algo = SternIterativeGPU(G, device=device)

    print(f"Running with {args.num_processes} CPU cores...")
    t = time.time()
    codewords = algo.run(
        w=args.w,
        p=args.p,
        ell=args.ell,
        max_iters=args.max_iters,
        num_processes=args.num_processes
    )
    print(time.time() - t)
    if len(codewords) > 0:
        np.save(args.out, np.array(codewords, dtype=np.uint8))
        print(f"Found {len(codewords)} distinct codewords of weight {args.w}")
        print(f"Saved to {args.out}")
    else:
        print("No codeword found in given iterations.")
