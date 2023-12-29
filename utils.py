#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 27 11:12:51 2023

@author: weihan
"""
import pandas as pd

def read_fasta_to_dict(fasta_file: str = 'demo.fa') -> dict:
    """
    Read fasta to py dict.
    
    The seq name is dict keys and seq is values.

    Parameters
    ----------
    fasta_file : fasta format file

    Returns
    -------
    A dictionary. 
        Key is sequence name and value is sequence.
    """
    file = open(fasta_file, 'r')
    seq_dict = {} # Define a dict for saving
    for i in file:
        i = i.strip() # Remove space in seq name
        if len(i) > 0: # Skip empty line
            if i[0] == '>':
                i = i.split(' ')
                key = i[0][1:]
                seq_dict[key] = '' # Seq header as key
            else:
                seq_dict[key] += i # Seq as value
    
    return seq_dict


def complement_fasta_dict(fasta_dict: dict) -> dict:
    """
    Obtain complement sequence.
    
    Matching rules can be customized.

    Parameters
    ----------
    fasta_dict : dict
        Output from read_fasta_to_dict, key is seq name and value is sequence.

    Returns
    -------
    dict
        Key is seq name and value is complement sequence.

    """
    transtab = str.maketrans('ATCG','TAGC') # This can be customized
    
    comp_dict = fasta_dict.copy()
    for i in comp_dict:
        comp_dict[i] = comp_dict[i].translate(transtab)
    
    return comp_dict


def reverse_fasta_dict(fasta_dict: dict) -> dict:
    """
    Obtain reverse sequence.
    
    Matching rules can be customized.

    Parameters
    ----------
    fasta_dict : dict
        Output from read_fasta_to_dict, key is seq name and value is sequence.

    Returns
    -------
    dict
        Key is seq name and value is reverse sequence.

    """
    rev_dict = fasta_dict.copy()
    for i in rev_dict:
        rev_dict[i] = rev_dict[i][::-1]
    
    return rev_dict


def reverse_complement_fasta_dict(fasta_dict: dict) -> dict:
    """
    Obtain reverse complement sequence.
    
    Matching rules can be customized.

    Parameters
    ----------
    fasta_dict : dict
        Output from read_fasta_to_dict, key is seq name and value is sequence.

    Returns
    -------
    dict
        Key is seq name and value is reverse complement sequence.

    """
    transtab = str.maketrans('ATCG','TAGC') # This can be customized
    
    rev_comp_dict = fasta_dict.copy()
    for i in rev_comp_dict:
        rev_comp_dict[i] = rev_comp_dict[i].translate(transtab)
        rev_comp_dict[i] = rev_comp_dict[i][::-1]
        
    return rev_comp_dict


def stat_kmer(seq_dict: dict, kmer_num: int) -> pd.DataFrame:
    """
    Stat sequence K-mer frequency.
    
    Row is sequence and col is kmer type.

    Parameters
    ----------
    seq_dict : sequence dict
        Output from fasta_to_dict.py
        
    kmer_num : the kmer length
        The kmer length 'K'

    Returns
    -------
    pd.DataFrame
        Row is sequence name and col is kmer.
        Valus is kmer frequency in this seq.
    """
    # Four base type
    nt_list = ['A','T','C','G']
    
    # Get all kmer combinations
    from itertools import product
    kmer_list = list(product(nt_list, repeat=kmer_num))
    kmer_list = list(map(lambda x:''.join(x), kmer_list))
    
    # Define a dataframe for saving
    kmer_dt = pd.DataFrame(0, index=seq_dict.keys(), columns=kmer_list)
    
    for i in seq_dict:
        this_seq_name = i
        this_seq_seq = seq_dict[i]
        for j in range(len(this_seq_seq)-kmer_num+1):
            this_kmer = this_seq_seq[j:j+kmer_num]
            if this_kmer in kmer_list:
                kmer_dt.loc[this_seq_name, this_kmer] += 1
                
    return kmer_dt


def seq_cosine_similarity(kmer_dt: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate sequence cosine similarity.

    Parameters
    ----------
    kmer_dt : a dataframe of sequence kmer frequency
        Output from stat_kmer.py

    Returns
    -------
    pd.DataFrame
        Row and col are sequence name, values are similarity of each pair.

    """
    from sklearn.metrics.pairwise import cosine_similarity
    
    cosine_value = cosine_similarity(kmer_dt.values, kmer_dt.values)
    cosine_similarity = pd.DataFrame(cosine_value, 
                                     index=kmer_dt.index.tolist(), 
                                     columns=kmer_dt.index.tolist())
    
    return cosine_similarity


def seq_jaccard_similarity(kmer_dt: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate sequence jaccard similarity.

    Parameters
    ----------
    kmer_dt : a dataframe of sequence kmer frequency
        Output from stat_kmer.py

    Returns
    -------
    pd.DataFrame
        Row and col are sequence name, values are similarity of each pair.

    """
    from sklearn.metrics import pairwise_distances
    
    # 'kmer_dt.values' > 0 aim to avoid warning,
    # 'kmer_dt.values' can also work
    jaccard_value = pairwise_distances(kmer_dt.values > 0, kmer_dt.values > 0,
                                       metric='jaccard')
    jaccard_similarity = pd.DataFrame(1 - jaccard_value, 
                                     index=kmer_dt.index.tolist(), 
                                     columns=kmer_dt.index.tolist())
    
    return jaccard_similarity


def stat_miRNA_in_gene(miRNA_com_dict: dict,
                          miRNA_rev_com_dict: dict,
                          gene_dict: dict,
                          match_length:int) -> pd.DataFrame:
    """
    Match miRNA with gene seed region according base complement.
    
    The goal of this function is to detect whether a mature miRNA could match
    a gene by base complement rule. For example, we set match_length is 15, 
    it will detect the frequecy of this 15-base sequence in miRNA and gene, 
    if both miRNA and gene have the 15-base sequence, it means they can match.
    
    Parameters
    ----------
    miRNA_dict : dict
        The miRNA complement dict
    miRNA_rev_com_dict : dict
        The miRNA reverse complement dict
    gene_dict : dict
        The gene dict.
    match_legenth : int
        No longer than miRNA sequence length.

    Returns
    -------
    pd.DataFrame
        Row is gene and miRNA, col is seed sequence type.

    """
    # Stat the seed types
    seed_list = []
    for i in miRNA_com_dict:
        com_seq = miRNA_com_dict[i]
        rc_seq = miRNA_rev_com_dict[i]
        for j in range(len(com_seq)-match_length+1):
            com_seed = com_seq[j:j+match_length]
            if com_seed not in seed_list:
                seed_list.append(com_seed)
            
            rc_seed = rc_seq[j:j+match_length]
            if rc_seed not in seed_list:
                seed_list.append(rc_seed)
                
    # Detect whether one seed in miRNA and gene seq
    node_list = [*gene_dict] + [*miRNA_com_dict]
    seed_num = pd.DataFrame(0, index=node_list, columns=seed_list)
    
    # For gene node
    for i in [*gene_dict]:
        for j in seed_list:
            if j in gene_dict[i]:
                #print(i,j)
                seed_num.loc[i,j] = gene_dict[i].count(j)
    
    # For miRNA node
    for i in [*miRNA_com_dict]:
        for j in seed_list:
            if (j in miRNA_com_dict[i]) or (j in miRNA_rev_com_dict[i]):
            #if j in miRNA_com_dict[i]:
                #print(i,j)
                seed_num.loc[i,j] = miRNA_com_dict[i].count(j) + miRNA_rev_com_dict[i].count(j)
    
    return seed_num