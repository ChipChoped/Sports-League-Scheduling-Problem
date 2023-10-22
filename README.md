# Sports League Scheduling Problem

## Problem description

### Parameters :

- $T$ : Set of teams of size $T_size \in \{n \in \N^+\ |\  n \mod 2 = 0\}$
- $W$ : The number of weeks where $W = T-1$
- $P$ : The number of periods where $P = \frac{T}{2}$

### Variables :

- $S$ : Array of size $W \times P$ where each cell is a couple $(t, t')$ that represents a match between two teams $t, t' \in T$

### Model :

- $\forall (T_n, T_k) \in S, n < k, \forall n, k \in Tsize$
- $\forall t \in T, (\sum_j^P t \in S_{i, j} == 1) = 1, \forall i \in [0..W]$
- $\forall t \in T, (\sum_i^W t \in S_{i, j} == 1) <= 2, \forall j \in [0..P]$

## Strategy

- We first initialize a graph where an edge is a team, an edge is a match and its label the week number
- Swap two matches if at least one of them is conflicting with the model (only matches on the same week)
- Tabou list to prevent from swapping two matches that have been swapped recently