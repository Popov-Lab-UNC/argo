#!/usr/bin/env python3
"""
Enhanced Fragment-based Molecular Generation Script

This script implements a comprehensive workflow for generating new molecules based on fragment clustering:
1. Read molecules from SDF file
2. Fragment each molecule at rotatable bonds
3. Cluster fragments by 3D coordinates
4. Sample from clusters to generate new molecules using sophisticated fragment linking
5. Remove duplicates and exact matches to original dataset
"""

import os
import sys
import numpy as np
import pandas as pd
from collections import defaultdict, Counter
from typing import List, Dict, Tuple, Set
import random
from pathlib import Path

# RDKit imports
from rdkit import Chem
from rdkit.Chem import Draw, AllChem, rdMolDescriptors

# Clustering
from sklearn.cluster import AgglomerativeClustering

# Local imports
from argo.frag_utils import SAFECodec, find_connected_rotatable_bond_ends

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)


def compute_fragment_centroids(mol, bonds, conf_id=-1, mass_weighted=False):
    """
    Compute centroids for fragments created by cutting bonds.
    
    Args:
        mol: RDKit molecule
        bonds: List of (atom_i, atom_j) tuples from find_connected_rotatable_bond_ends
        conf_id: Conformer ID to use for coordinates
        mass_weighted: Whether to use mass-weighted centroids
        
    Returns:
        List of tuples: (fragment_atoms, centroid, n_exits, smiles)
    """
    num_atoms = mol.GetNumAtoms()

    # Build a temporary molecule with cut bonds and attach dummy atoms (*) at the cut ends
    rw = Chem.RWMol(Chem.Mol(mol))
    # Remove the bonds to split fragments
    for i, j in bonds:
        if rw.GetBondBetweenAtoms(i, j) is not None:
            rw.RemoveBond(i, j)
    # Add exit-vector dummies bound to each cut endpoint
    for i, j in bonds:
        a1 = Chem.Atom(0)
        a2 = Chem.Atom(0)
        idx1 = rw.AddAtom(a1)
        idx2 = rw.AddAtom(a2)
        rw.AddBond(i, idx1, Chem.BondType.SINGLE)
        rw.AddBond(j, idx2, Chem.BondType.SINGLE)
    fragged = rw.GetMol()

    # Get fragment membership and fragment molecules (with dummies)
    frags_idx_all = Chem.GetMolFrags(fragged, asMols=False)
    frags_mols = Chem.GetMolFrags(fragged, asMols=True, sanitizeFrags=True)

    # Map original atoms to fragments (ignore dummy atoms)
    atom_to_frag = [-1] * num_atoms
    for frag_idx, frag in enumerate(frags_idx_all):
        for a in frag:
            if a < num_atoms:
                atom_to_frag[a] = frag_idx

    # Count exit vectors per fragment (each cut contributes one to each side)
    exit_counts = [0] * len(frags_idx_all)
    for i, j in bonds:
        fi = atom_to_frag[i]
        fj = atom_to_frag[j]
        if fi != -1:
            exit_counts[fi] += 1
        if fj != -1:
            exit_counts[fj] += 1

    # Coordinates from the original conformer
    conf = mol.GetConformer(conf_id)
    positions = np.array([list(conf.GetAtomPosition(i)) for i in range(num_atoms)])

    masses = None
    if mass_weighted:
        masses = np.array([mol.GetAtomWithIdx(i).GetMass() for i in range(num_atoms)])

    results = []
    for frag_idx, frag in enumerate(frags_idx_all):
        # Use only original atoms (exclude dummy atoms appended to fragged)
        idxs_list = [a for a in frag if a < num_atoms]
        idxs = np.array(idxs_list, dtype=int)
        if mass_weighted:
            w = masses[idxs][:, None]
            centroid = (positions[idxs] * w).sum(axis=0) / w.sum()
        else:
            centroid = positions[idxs].mean(axis=0)
        # SMILES with exit vectors ([*]) preserved
        frag_smiles = Chem.MolToSmiles(frags_mols[frag_idx], isomericSmiles=True, canonical=True)
        results.append((tuple(idxs_list), centroid, exit_counts[frag_idx], frag_smiles))
    return results


def cluster_fragments(fragments, threshold=1.0):
    """
    Cluster fragments based on their 3D centroids.
    
    Args:
        fragments: List of fragment dictionaries with 'centroid' key
        threshold: Distance threshold for clustering
        
    Returns:
        Dictionary mapping cluster_id to list of fragments
    """
    if len(fragments) == 0:
        return {}
    
    # Extract centroids into array
    centroids = np.array([f['centroid'] for f in fragments])
    
    # Perform clustering
    clustering = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=threshold,
        metric='euclidean',
        linkage='single'
    )
    
    # Get cluster labels
    labels = clustering.fit_predict(centroids)
    
    # Group fragments by cluster
    clusters = {}
    for i, label in enumerate(labels):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(fragments[i])
        
    return clusters

'''
def sample_from_cluster(cluster, n_samples, distribution_type='count_based'):
    """
    Sample fragments from a cluster based on distribution.
    
    Args:
        cluster: List of fragment dictionaries
        n_samples: Number of samples to generate
        distribution_type: Type of distribution ('count_based', 'uniform', 'score_based')
        
    Returns:
        List of sampled fragment SMILES
    """
    if len(cluster) == 0:
        return []
    
    if distribution_type == 'count_based':
        # Weight by cluster size (more fragments = higher weight)
        weights = np.ones(len(cluster))
    elif distribution_type == 'uniform':
        # Equal weights
        weights = np.ones(len(cluster))
    elif distribution_type == 'score_based':
        # Weight by docking score (lower score = higher weight)
        scores = np.array([f.get('score', 0) for f in cluster])
        # Convert to weights (lower score = higher weight)
        weights = 1.0 / (scores - scores.min() + 1e-6)
    else:
        weights = np.ones(len(cluster))
    
    # Normalize weights
    weights = weights / weights.sum()
    
    # Sample with replacement
    sampled_indices = np.random.choice(len(cluster), size=n_samples, p=weights, replace=True)
    sampled_fragments = [cluster[i]['smiles'] for i in sampled_indices]
    
    return sampled_fragments
'''

def remove_duplicates_and_similar_molecules(generated_molecules, original_molecules, similarity_threshold=1.0):
    """
    Remove duplicates from generated molecules and molecules similar to original dataset using Tanimoto similarity.
    
    Args:
        generated_molecules: List of generated molecule SMILES
        original_molecules: List of original molecule SMILES
        similarity_threshold: Tanimoto similarity threshold (0.8 = 80% similar)
        
    Returns:
        List of unique generated molecules
    """
    from rdkit import DataStructs
    from rdkit.Chem import AllChem
    
    print(f"Generated {len(generated_molecules)} molecules")
    
    # Remove exact duplicates from generated molecules first
    unique_generated = list(set(generated_molecules))
    print(f"After removing exact duplicates: {len(unique_generated)} molecules")
    
    # Convert original molecules to Morgan fingerprints for similarity comparison
    original_fps = []
    original_mols = []
    for smiles in original_molecules:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
            original_fps.append(fp)
            original_mols.append(mol)
    
    print(f"Converted {len(original_fps)} original molecules to fingerprints")
    
    # Filter generated molecules based on Tanimoto similarity
    filtered_molecules = []
    for smiles in unique_generated:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue
            
        # Generate fingerprint for current molecule
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
        
        # Check similarity against all original molecules
        max_similarity = 0.0
        for orig_fp in original_fps:
            similarity = DataStructs.TanimotoSimilarity(fp, orig_fp)
            max_similarity = max(max_similarity, similarity)
        
        # Keep molecule if similarity is below threshold
        if max_similarity < similarity_threshold:
            filtered_molecules.append(smiles)
        else:
            print(f"Removed molecule with {max_similarity:.3f} similarity to original dataset")
    
    print(f"After removing similar molecules (threshold {similarity_threshold}): {len(filtered_molecules)} molecules")
    
    return filtered_molecules

def find_cluster(smiles, centroid, clustered_fragments):
    # Iterate through each cluster level
    for cluster_level, fragments in clustered_fragments.items():
        # Iterate through each fragment group in this level
        for group_id, group_fragments in fragments.items():
            # Check each fragment in the group
            for fragment in group_fragments:
                # Compare SMILES and check if centroids are close
                if fragment['smiles'] == smiles and np.allclose(fragment['centroid'], centroid, atol=0.1):
                    return cluster_level, group_id
    
    # Return None if no matching cluster is found
    return None

def sample_fragment_from_cluster(cluster_level, group_id, clustered_fragments, distribution_type='random'):
    """
    Get a random fragment from a cluster based on distribution type.
    
    Args:
        cluster_level: Number of exit vectors
        group_id: Cluster group ID
        clustered_fragments: Dictionary of clustered fragments
        distribution_type: 'random' or 'count_based'
        
    Returns:
        Randomly selected fragment SMILES
    """
    # Get all fragments in the specified cluster
    fragments = clustered_fragments[cluster_level][group_id]

    if distribution_type == 'random':
        # Simple random choice from unique SMILES
        smiles = [frag['smiles'] for frag in fragments]
        smiles = list(set(smiles))
        return random.choice(smiles)
    elif distribution_type == 'count_based':
        # Count-based distribution: weight by how many times each SMILES appears
        smiles_counts = {}
        for frag in fragments:
            smiles = frag['smiles']
            smiles_counts[smiles] = smiles_counts.get(smiles, 0) + 1
        
        # Convert to lists for weighted random choice
        smiles_list = list(smiles_counts.keys())
        weights = list(smiles_counts.values())
        
        # Normalize weights
        total_weight = sum(weights)
        normalized_weights = [w / total_weight for w in weights]
        
        # Weighted random choice
        return np.random.choice(smiles_list, p=normalized_weights)


def main():
    """Main function implementing the fragment-based generation workflow."""
    
    # Configuration
    input_file = "PptT/benchmark/hgd_conformers.sdf"
    output_file = "generated_molecules.smi"
    top_n_molecules = 10
    molecules_per_molecule = 1000
    clustering_threshold = 1.0
    
    print(f"Starting fragment-based molecular generation...")
    print(f"Input file: {input_file}")
    print(f"Top {top_n_molecules} molecules will be used as template for generation")
    print(f"Generating {molecules_per_molecule} molecules per input molecule")
    
    # Step 1: Read molecules from SDF file
    print("\n1. Reading molecules from SDF file...")
    if not os.path.exists(input_file):
        print(f"Error: Input file {input_file} not found!")
        print("Please ensure hgd_conformers.sdf is in the current directory.")
        return
    
    suppl = Chem.SDMolSupplier(input_file)
    mols = [mol for mol in suppl if mol is not None]
    print(f"Read {len(mols)} molecules from SDF file")
    
    if len(mols) == 0:
        print("No molecules found in SDF file!")
        return
    
    # Step 2: Fragment each molecule
    print("\n2. Fragmenting molecules...")
    fragments_by_exits = {}
    
    for i, mol in enumerate(mols):
        print(f"Processing molecule {i+1}/{len(mols)}")
        
        # Get rotatable bond ends
        try:
            ends = find_connected_rotatable_bond_ends(mol)
            
            # Get centroids for each fragment
            centroids = compute_fragment_centroids(mol, ends)
            
            # Store each fragment
            for frag_atoms, centroid, n_exits, smiles in centroids:
                if n_exits not in fragments_by_exits:
                    fragments_by_exits[n_exits] = []
                
                fragments_by_exits[n_exits].append({
                    'smiles': smiles,
                    'centroid': centroid,
                    'molecule_idx': i,
                    'fragment_atoms': frag_atoms
                })
        except Exception as e:
            print(f"Error processing molecule {i}: {e}")
            continue
    
    # Print fragment summary
    print("\nFragment summary:")
    for n_exits in sorted(fragments_by_exits.keys()):
        print(f"Fragments with {n_exits} exit vectors: {len(fragments_by_exits[n_exits])}")
    
    # Step 3: Cluster fragments
    print("\n3. Clustering fragments...")
    clustered_fragments = {}
    
    for n_exits, frags in fragments_by_exits.items():
        print(f"\nClustering fragments with {n_exits} exit vectors:")
        clusters = cluster_fragments(frags, threshold=clustering_threshold)
        clustered_fragments[n_exits] = clusters
        print(f"Found {len(clusters)} clusters")
        
        # Print size of each cluster
        for cluster_id, cluster in clusters.items():
            print(f"Cluster {cluster_id}: {len(cluster)} fragments")
    
    # Step 4: Generate molecules for top N molecules
    print(f"\n4. Generating molecules for top {top_n_molecules} molecules...")
    
    all_generated_molecules = []
    original_molecules_smiles = []
    
    # Get SMILES of original molecules
    for mol in mols:
        try:
            smiles = Chem.MolToSmiles(mol, canonical=True)
            original_molecules_smiles.append(smiles)
        except:
            continue
    
    safecodec = SAFECodec(slicer=find_connected_rotatable_bond_ends)

    # Process top N molecules
    for mol_idx in range(min(top_n_molecules, len(mols))):
        print(f"\nProcessing molecule {mol_idx} for generation...")
        
        ends = find_connected_rotatable_bond_ends(mols[mol_idx])
        centroids = compute_fragment_centroids(mols[mol_idx], ends)

        cluster_list = []
        for frag in centroids:
            smiles = frag[3]
            centroid = frag[1]
            cluster = find_cluster(smiles, centroid, clustered_fragments)
            cluster_list.append(cluster)

        print('cluster_list', cluster_list)

        generated_molecules = []
        while len(generated_molecules) < molecules_per_molecule:
            random_fragments = []
            if cluster_list is not None:
                for cluster in cluster_list:
                    cluster_level, group_id = cluster
                    random_frag = sample_fragment_from_cluster(cluster_level, group_id, clustered_fragments, distribution_type='count_based')
                    random_fragments.append(random_frag)
            else:
                print(f"No cluster found for fragment {smiles}")

            # Connect fragments sequentially
            linked_smiles = random_fragments[0]
            for i in range(1, len(random_fragments)):
                linked_smiles = safecodec.link_fragments(linked_smiles, random_fragments[i])

            generated_molecules.append(linked_smiles)

        all_generated_molecules.extend(generated_molecules)

    # Step 5: Remove duplicates and similar molecules
    print(f"\n5. Removing duplicates and similar molecules...")
    final_molecules = remove_duplicates_and_similar_molecules(
        all_generated_molecules, 
        original_molecules_smiles,
        similarity_threshold=1.0
    )
    
    # Step 6: Save results
    print(f"\n6. Saving results...")
    if final_molecules:
        with open(output_file, 'w') as f:
            for smiles in final_molecules:
                f.write(f"{smiles}\n")
        print(f"Saved SMILES to {output_file}")
    else:
        print("No unique molecules generated!")
    
    print("\nFragment-based generation complete!")


if __name__ == "__main__":
    main()
