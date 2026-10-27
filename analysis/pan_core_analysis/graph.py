"""
SMILES token-level graph distance matrix utilities.

Converts a SMILES string into a token-level distance matrix using RDKit atom
connectivity. Each token is mapped to its corresponding atom in the molecular
graph; special tokens (BOS, EOS, PAD) and non-atom tokens (bonds, branches,
ring closures) are mapped to atom -1 and receive distance -1.0.
"""

from __future__ import annotations

import numpy as np
from rdkit import Chem


def create_token_graph_distance_matrix(smiles: str, tokenizer) -> np.ndarray | None:
    """Build a [seq_len, seq_len] token-level graph-distance matrix for a SMILES string.

    Each cell (i, j) contains the shortest-path graph distance between the atoms
    corresponding to tokens i and j, or -1.0 if either token has no atom mapping.

    The tokenizer is expected to produce a flat list of string tokens and expose
    `bos_token`, `eos_token`, and `pad_token` attributes.

    Returns None if the SMILES cannot be parsed or the atom count does not match
    the number of atom tokens found during parsing.
    """
    mol = Chem.MolFromSmiles(smiles, sanitize=False)
    if mol is None:
        return None
    mol.UpdatePropertyCache(strict=False)
    atom_dist_matrix = Chem.GetDistanceMatrix(mol)

    tokens = tokenizer.tokenize(smiles)
    tokens = [tokenizer.bos_token] + tokens + [tokenizer.eos_token]
    seq_len = len(tokens)

    # Elements that appear without brackets in SMILES (OpenSMILES organic subset)
    organic_subset = {
        "B", "C", "N", "O", "P", "S", "F", "Cl", "Br", "I",
        "b", "c", "n", "o", "p", "s", "*",
    }

    token2atom: list[int] = []
    atom_idx = -1
    branch_stack: list[int] = []
    current_atom = -1
    in_bracket = False

    for token in tokens:
        # Special control tokens
        if token in {tokenizer.bos_token, tokenizer.eos_token, tokenizer.pad_token}:
            token2atom.append(-1)
            continue

        if token == "[":
            in_bracket = True
            atom_idx += 1
            current_atom = atom_idx
            token2atom.append(current_atom)
        elif token == "]":
            token2atom.append(current_atom)
            in_bracket = False
        elif in_bracket:
            token2atom.append(current_atom)
        else:
            if token in organic_subset:
                atom_idx += 1
                current_atom = atom_idx
                token2atom.append(current_atom)
            elif token == "(":
                branch_stack.append(current_atom)
                token2atom.append(current_atom)
            elif token == ")":
                if branch_stack:
                    current_atom = branch_stack.pop()
                token2atom.append(current_atom)
            else:
                # Bond symbols, ring-closure digits, etc.
                token2atom.append(current_atom)

    actual_atoms = mol.GetNumAtoms()
    parsed_atoms = atom_idx + 1
    if parsed_atoms != actual_atoms:
        return None

    dist_matrix = np.full((seq_len, seq_len), -1.0)
    for i in range(seq_len):
        for j in range(seq_len):
            u, v = token2atom[i], token2atom[j]
            if u != -1 and v != -1:
                dist_matrix[i, j] = atom_dist_matrix[u, v]

    return dist_matrix


def collate_distance_matrices(
    dist_matrices: list[np.ndarray | None],
    maxlen: int | None = None,
    pad_value: float = -1.0,
) -> np.ndarray:
    """Pad and stack a list of per-sample distance matrices into a [B, L, L] array.

    None entries (failed molecules) are filled entirely with `pad_value`.

    Args:
        dist_matrices: List of 2-D arrays or None, one per sample.
        maxlen: Target sequence length; if None, inferred from the longest matrix.
        pad_value: Fill value for padding and failed molecules.
    Returns:
        3-D float array of shape [B, maxlen, maxlen].
    """
    valid_sizes = [m.shape[0] for m in dist_matrices if m is not None]
    if not valid_sizes and maxlen is None:
        raise ValueError("All matrices are None and maxlen is not specified.")
    max_len = max(max(valid_sizes) if valid_sizes else 0, maxlen or 0)
    batch_size = len(dist_matrices)

    out = np.full((batch_size, max_len, max_len), pad_value, dtype=float)
    for i, mat in enumerate(dist_matrices):
        if mat is None:
            continue
        curr_len = mat.shape[0]
        out[i, :curr_len, :curr_len] = mat

    return out
