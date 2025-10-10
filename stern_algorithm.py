# stern_iterative_prop2_gpu.py
import torch, random
import numpy as np 
import argparse
from typing import Optional, List, Tuple

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
                pivot = r; break
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
    H_left = Z.T.clone() % 2   # (-Z^T) == Z^T in GF(2)
    H_right = torch.eye(n-k, dtype=torch.uint8, device=Gsys.device)
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

    # ---------- One iteration of Stern ----------
    def iteration_once(self, w: int, p: int = 1, ell: int = 16):
        k = self.k; nR = self.n_redundant; Z = self.Z
        perm_rows = torch.randperm(k, device=self.device)
        rows1, rows2 = perm_rows[:k//2], perm_rows[k//2:]
        if ell > nR: ell = nR
        L_idx = torch.tensor(random.sample(range(nR), ell), device=self.device)
        JL_mask = torch.ones(nR, dtype=torch.uint8, device=self.device); JL_mask[L_idx] = 0
        JL_mask = JL_mask.bool()

        if p == 1:
            comb1, V1_full = Z[rows1][:, L_idx], Z[rows1]
            comb2, V2_full = Z[rows2][:, L_idx], Z[rows2]
        elif p == 2:
            idxs1 = torch.combinations(torch.arange(rows1.size(0), device=self.device), r=2)
            idxs2 = torch.combinations(torch.arange(rows2.size(0), device=self.device), r=2)
            if idxs1.numel() == 0 or idxs2.numel() == 0: return None
            V1_full = Z[rows1[idxs1[:,0]]] ^ Z[rows1[idxs1[:,1]]]
            V2_full = Z[rows2[idxs2[:,0]]] ^ Z[rows2[idxs2[:,1]]]
            comb1, comb2 = V1_full[:, L_idx], V2_full[:, L_idx]
        else:
            raise NotImplementedError("Only p=1 or p=2 are supported.")

        key1, key2 = pack_bits_to_uint64(comb1), pack_bits_to_uint64(comb2)
        keys = torch.cat([key1, key2])
        part = torch.cat([torch.zeros_like(key1), torch.ones_like(key2)])
        idxs = torch.arange(keys.size(0), device=self.device)
        sort_idx = torch.argsort(keys); keys_s, part_s, idxs_s = keys[sort_idx], part[sort_idx], idxs[sort_idx]

        runs = []; start = 0
        for i in range(1, keys_s.numel()):
            if keys_s[i] != keys_s[i-1]:
                runs.append((start, i)); start = i
        runs.append((start, keys_s.numel()))

        for a, b in runs:
            block_parts = part_s[a:b]
            if not ((block_parts==0).any() and (block_parts==1).any()): continue
            run_idxs = idxs_s[a:b]
            left_rel = run_idxs[block_parts==0]
            right_rel = run_idxs[block_parts==1] - key1.size(0)
            V1c, V2c = V1_full[left_rel], V2_full[right_rel]
            xor_pairs = (V1c.unsqueeze(1) ^ V2c.unsqueeze(0)).reshape(-1, nR)
            wt = hamming_weight_rows(xor_pairs[:, JL_mask])
            matches = (wt == (w - 2*p)).nonzero(as_tuple=False)
            if matches.numel() > 0:
                idx_match = matches[0].item()
                left_idx, right_idx = idx_match // V2c.size(0), idx_match % V2c.size(0)
                redundancy = V1c[left_idx] ^ V2c[right_idx]
                info_vec = torch.zeros(k, dtype=torch.uint8, device=self.device)
                if p == 1:
                    info_vec[rows1[left_rel[left_idx]].item()] = 1
                    info_vec[rows2[right_rel[right_idx]].item()] = 1
                elif p == 2:
                    # just approximate info weight contribution
                    info_vec[rows1[idxs1[left_rel[left_idx] // 1][0]].item()] = 1
                    info_vec[rows1[idxs1[left_rel[left_idx] // 1][1]].item()] = 1
                    info_vec[rows2[idxs2[right_rel[right_idx] // 1][0]].item()] = 1
                    info_vec[rows2[idxs2[right_rel[right_idx] // 1][1]].item()] = 1
                codeword_perm = torch.cat([info_vec, redundancy])
                # Map back to original order
                codeword_orig = torch.zeros(self.n, dtype=torch.uint8, device=self.device)
                for pos, orig in enumerate(self.col_perm):
                    codeword_orig[orig] = codeword_perm[pos]
                return codeword_orig.cpu().numpy()
        return None

    def run(self, w: int, p: int = 1, ell: int = 16, max_iters: int = 1000):
        found = []
        seen = set()

        for it in range(max_iters):
            res = self.iteration_once(w, p=p, ell=ell)
            if res is not None:
                c = torch.tensor(res, dtype=torch.uint8, device=self.device)
                if is_codeword_syndrome(c, self.H):
                    tup = tuple(res.tolist())
                    if tup not in seen:
                        seen.add(tup)
                        found.append(res)

            #if res is not None:
             #   tup = tuple(res.tolist())
             #   if tup not in seen:
             #       seen.add(tup)
             #       found.append(res)

            if it % 10 == 0:
                red = random.randrange(self.n_redundant)
                self.proposition2_swap(info_col_pos=random.randrange(self.k),
                                       redundant_col_pos=red)

        return found


# ------------------ Example ------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stern iterative algorithm with Proposition-2 (GPU)")
    parser.add_argument("--hfile", type=str, required=True, help="Path to generator matrix .npy file")
    parser.add_argument("--w", type=int, required=True, help="Target weight")
    parser.add_argument("--p", type=int, default=2, choices=[1,2], help="Combination size (1 or 2)")
    parser.add_argument("--ell", type=int, default=14, help="Subset size ell")
    parser.add_argument("--max_iters", type=int, default=5000, help="Max iterations")
    parser.add_argument("--out", type=str, required=True, help="Output file (.npy) to save found codewords")
    args = parser.parse_args()

    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    G = torch.tensor(np.load(args.hfile), dtype=torch.uint8, device=device)
    k, n = G.shape
    algo = SternIterativeGPU(G, device=device)
    codewords = algo.run(w=args.w, p=args.p, ell=args.ell, max_iters=args.max_iters)

    if len(codewords) > 0:
        np.save(args.out, np.array(codewords, dtype=np.uint8))
        print(f"Found {len(codewords)} distinct codewords of weight {args.w}")
        print(f"Saved to {args.out}")
    else:
        print("No codeword found in given iterations.")

    #while True:
     #   G = torch.randint(0,2,(k,n),dtype=torch.uint8,device=device)
     #   try:
     #       algo = SternIterativeGPU(G, device=device)
     #       break
     #   except: continue
    #print("Initial info set:", algo.col_perm[:k])
    #res = algo.run(w=4, p=2, ell=6, max_iters=200)  # try with p=2
    #print("Found codeword in original order:", res)


