# Chapter 9 — Clustering of High-Dimensional Data

> **Course:** Data Mining and Machine Learning / Bioinspired Computational Methods — Biological Data Mining
> **Instructor:** Prof. Francesco Marcelloni (University of Pisa)
> **Sources merged into this file:** professor's slides (`9-HighDimensionalClustering.pdf`, 15 pages — Chapter 11 of Han–Kamber–Pei) + lecture notes (`9 - High dimension Clustering sbobine.pdf`, 21 pages)
>
> **⚠️ Note on coverage**: the lecture notes for this chapter also carried a **second block, "Clustering with Constraints" (pp. 12–21)**, which has **no counterpart in this slide deck**. That topic has its own sources and its own file — see **`11 - Constrained Clustering/CAPITOLO_11_STUDIA_SOLO_QUESTO_Clustering_with_Constraints.md`**. It is **not** duplicated here.
>
> **Instructions for Claude:** this is the single reference file for Chapter 9. When the user asks study questions about this chapter, use **only** this file as context (no need to re-read the original PDFs unless a detail is missing). The file merges the official slide content with the lecture's discursive explanations and worked examples, keeping both levels of detail.

---

## Chapter outline

1. Why traditional distance measures fail in high dimensions
2. The curse of dimensionality and why subspace clustering
3. Subspace Clustering Method (I): subspace search — bottom-up (CLIQUE) vs. top-down (PROCLUS)
4. Subspace Clustering Method (II): correlation-based methods (PCA, Hough transform, fractal dimensions)
5. Subspace Clustering Method (III): bi-clustering
   - Types of bi-clusters
   - Optimization-based: the δ-Cluster algorithm
   - Enumeration-based: δ-pCluster and MaPle
6. Dimensionality-reduction methods
7. Spectral clustering: the Ng-Jordan-Weiss (NJW) algorithm

*(Clustering with constraints — must-link/cannot-link, COP-k-means, CVQE, obstacles — is Chapter 11.)*

---

## 1. Are traditional distance measures effective in high dimensions?

**The opening question of the chapter**: *are the traditional distance measures which are frequently used in low-dimensional cluster analysis also effective on high-dimensional data?*

**The counter-example discussed in class (Ada / Bob / Cathy):** three customers described by the **same set of binary attributes**. If we compute the **Euclidean distance** between them, **they all turn out to be equally distant** — yet **by inspection, Ada and Cathy clearly look more similar**.

> **Conclusion** (lecture): **traditional approaches based on the Euclidean distance are NOT appropriate when we have to cope with high dimensionality.** The distance is mathematically correct but semantically useless — it cannot see the similarity that a human sees immediately.

**Clustering should not only consider dimensions but also attributes (features).** Two ways of working directly on the features:

- **Feature transformation** — effective **if most dimensions are relevant**. **PCA and SVD** are useful when features are **highly correlated / redundant**.
- **Feature selection** — useful to **find a subspace where the data have nice clusters**.

**Where this matters**: **text documents**, **DNA micro-array data** — the natural high-dimensional application domains.

**The three major challenges:**
1. **Many irrelevant dimensions may MASK the clusters.**
2. **The distance measure becomes meaningless — due to equi-distance.**
3. **Clusters may exist ONLY in some subspaces.**

**Two major families of methods:**

| Family | Idea | Examples |
|---|---|---|
| **Subspace clustering** | Search for clusters **existing in subspaces** of the given high-dimensional space | **CLIQUE**, **PROCLUS**, **bi-clustering** |
| **Dimensionality reduction** | Construct a **much lower-dimensional space** and search for clusters there (possibly **building new dimensions by combining original ones**) | dimensionality-reduction methods, **spectral clustering** |

> **After reducing the dimensionality, the classical clustering algorithms can be applied** — that is the whole point of the second family: it does not invent a new clustering algorithm, it changes the space in which the old ones work.

## 2. The curse of dimensionality (graphs adapted from Parsons et al., KDD Explorations 2004)

- **Data in only one dimension is relatively packed.**
- **Adding a dimension "stretches" the points** across that dimension, making them **further apart**.
- **Adding more dimensions makes the points further and further apart** — **high-dimensional data is extremely sparse**.
- **The distance measure becomes meaningless — due to equi-distance.**

> **The lecture's phrasing**: when we reason with just one dimension, **all objects appear close to each other**. When we add another dimension, objects that looked close **can turn out to be very far**. The phenomenon grows with every added dimension → **sparsity increases** until everything is equidistant from everything.

### Why subspace clustering

**Clusters may exist only in some subspaces.** The worked illustration from the lecture:

- Project the dataset **along dimension $a$**: red and green separate cleanly.
- Project **along dimension $b$**: two clusters become identifiable.
- Project **along the two dimensions together**: we can identify **some** clusters **but not all of them**.

> **Main idea**: **we can distinguish clusters when we focus on SUBSPACES**, so it is better to use those rather than the whole space. **The question then becomes: how do we determine that subspace?** The rest of this chapter is the set of answers.

**Subspace clustering: find clusters in all the subspaces.**

### The overall taxonomy of subspace clustering methods

- **Subspace search methods** — search various subspaces to find clusters
  - **Bottom-up approaches**
  - **Top-down approaches**
- **Correlation-based clustering methods** — e.g. **PCA-based approaches**
- **Bi-clustering methods**
  - **Optimization-based methods**
  - **Enumeration methods**

## 3. Subspace Clustering Method (I): subspace search methods

### Bottom-up approaches
- **Start from low-dimensional subspaces** and **search higher-dimensional subspaces only when there may be clusters in them**.
- **Various pruning techniques** reduce the number of higher-D subspaces to be searched.
- **Ex. CLIQUE** (Agrawal et al., 1998).

> **From the lecture**: CLIQUE — already seen in the clustering chapter — **starts from one dimension and puts together different dimensions to build a subspace**. Crucially, **we only analyse the part of the space where we are sure we can find clusters** (this is the Apriori-style pruning: a dense unit in $k$ dimensions must be dense in all its $(k-1)$-dimensional projections).

### Top-down approaches
- **Start from the full space** and **search smaller subspaces recursively**.
- **Effective only if the LOCALITY ASSUMPTION holds**: it requires that **the subspace of a cluster can be determined by the local neighbourhood**.
- **Ex. PROCLUS** (Aggarwal et al., 1999): a **k-medoid-like** method.

> **⚠️ The locality assumption is the weak point of top-down methods**: if the relevant subspace of a cluster cannot be guessed from a local neighbourhood, the recursive descent starts from the wrong place and never recovers.

## 4. Subspace Clustering Method (II): correlation-based methods

- **Subspace search methods** define similarity **based on distance or density**.
- **Correlation-based methods** are instead based on **advanced correlation models**.

**Ex. PCA-based approach:**
1. **Apply PCA** (Principal Component Analysis) to derive a set of **new, uncorrelated dimensions**;
2. then **mine clusters in the new space or in its subspaces**.

**Other space transformations**: **Hough transform**, **fractal dimensions**.

## 5. Subspace Clustering Method (III): bi-clustering

### 5.1 What bi-clustering is

**Bi-clustering: cluster BOTH objects AND attributes simultaneously**, treating objects and attributes in a **symmetric way**.

> **The key idea** (lecture): starting from the overall dataset, **we find a subspace and cluster it AT THE SAME TIME** — one operation, not two. This is what distinguishes bi-clustering from "select a subspace, then run a normal clustering algorithm in it".

**Four requirements a bi-clustering formulation must satisfy:**
1. **Only a small set of objects participates in a cluster** (*typical of medical problems*).
2. **A cluster only involves a small number of attributes.**
3. **An object may participate in multiple clusters, or in none at all.**
4. **An attribute may be involved in multiple clusters, or in none at all.**

> **⚠️ Notice how radically this differs from classical partitioning clustering (k-means & co.)**: there is **no exhaustive partition** — objects can be left out — and **clusters are not exclusive** — objects can belong to several. There is also **no globally defined similarity/distance measure**.

**Example 1 — gene expression / micro-array data: a gene × sample/condition matrix.**
- The **gene** is our **object** (row);
- the **samples/conditions** are the **attributes** (columns);
- **each element of the matrix is a real number recording the expression level of a gene under a specific condition**.

**Example 2 — clustering customers and products.**
- **Customers** as objects, **products** as attributes.
- **We expect a very high number of products — this is exactly where the high dimensionality comes from.**

### 5.2 Types of bi-clusters (the ideal, noise-free cases)

Let $A = \{a_1, \dots, a_n\}$ be a set of **genes**, $B = \{b_1, \dots, b_m\}$ a set of **conditions**, and $E = [e_{ij}]$ the **gene expression data matrix**.

> **Bi-cluster**: a **submatrix** where genes and conditions follow **some consistent pattern**.

**(1) Bi-clusters with CONSTANT VALUES**
$$e_{ij} = c \qquad \forall i \in I,\; \forall j \in J$$
Every value in the submatrix is the **same constant**. *The easiest case.*

**(2) Bi-clusters with CONSTANT VALUES ON ROWS**
$$e_{ij} = c + \alpha_i$$
where $\alpha_i$ is the **adjustment for row $i$** (the row-dependent offset). **Symmetrically, there are bi-clusters with constant values on columns**, $e_{ij} = c + \beta_j$.

**(3) Bi-clusters with COHERENT VALUES (a.k.a. pattern-based clusters)**

**Rows change in a synchronized way with respect to the columns, and vice versa:**
$$e_{ij} = c + \alpha_i + \beta_j$$
(adjustment **both** for rows **and** for columns).

**Equivalent characterisation** — $I \times J$ is a bi-cluster with coherent values **if and only if** for any $i_1, i_2 \in I$ and $j_1, j_2 \in J$:
$$e_{i_1 j_1} - e_{i_2 j_1} = e_{i_1 j_2} - e_{i_2 j_2}$$

> **The lecture's concrete reading**: in the example matrix, **the difference between 60 and 50 is the same as the difference between 20 and 10** — the two rows move in parallel, whatever their absolute level.

**(4) Bi-clusters with COHERENT EVOLUTIONS ON ROWS**

Here **we are only interested in the up- or down-regulated CHANGES** across genes or conditions, **without constraining the exact values**. For any $i_1, i_2 \in I$ and $j_1, j_2 \in J$:
$$(e_{i_1 j_1} - e_{i_1 j_2})\,(e_{i_2 j_1} - e_{i_2 j_2}) \;\ge\; 0$$

> **What the inequality says** (lecture): **we do not require exactly zero variation — we require the SAME TREND of variation.** If one row goes up between two conditions, the other must go up too (or stay flat); the product of the two differences is then non-negative. **This is the weakest, most permissive of the four definitions** — and biologically the most meaningful one, since what matters in gene expression is often the *direction* of regulation, not the magnitude.

### 5.3 From ideal definitions to real (noisy) data

> **The definitions above only work in theory**: real-world data is **noisy**, so we must **find APPROXIMATE bi-clusters**.

**Two families of methods:**

| | **Optimization-based** | **Enumeration-based** |
|---|---|---|
| **Goal** | find **one submatrix at a time** that achieves the **best significance** as a bi-cluster | **enumerate ALL submatrices** that satisfy the requirements |
| **How noise is controlled** | a **quality measure on the whole submatrix** (average noise) | a **tolerance threshold** on **every** small submatrix (per-element noise) |
| **Search** | **greedy search** → **local optima** (exhaustive computation is impossible) | systematic enumeration exploiting **monotonicity** |
| **Example** | **δ-Cluster algorithm** (Cheng & Church, ISMB'2000) | **δ-pCluster** (H. Wang et al., SIGMOD'2002), **MaPle** (Pei et al., ICDM'2003) |

> **Why greedy is unavoidable**: it is **impossible to compute all the possible combinations of rows and columns** to determine the bi-cluster, **even when the number of rows and columns is reasonable** — the number of submatrices is $2^{|A|} \cdot 2^{|B|}$.

### 5.4 The micro-array illustration

- **Left figure** — micro-array **"raw" data**, 3 genes and their values in a multi-dimensional space: **difficult to find their patterns**. *Analysing the three genes over ALL the features, we cannot recognise any regularity.*
- **Middle figure** — **isolate a subspace** (conditions $b, c, h, j, e$): now the three genes **vary in the same way even if at different levels** → a **bi-cluster with COHERENT VALUES** (a **shift** pattern; scaling patterns are analogous).
- **Right figure** — another subspace where the three lines show **the same trend** → a **bi-cluster with COHERENT EVOLUTION**.

**Three facts this illustration establishes:**
- **No globally defined similarity/distance measure** exists here;
- **clusters may not be exclusive**;
- **an object can appear in multiple clusters**.

> **To sum up** (lecture): **by working on a subspace we can find similarities between the rows and therefore find a pattern** that is completely invisible in the full space.

### 5.5 Bi-Clustering (I): δ-bi-cluster and the δ-Cluster algorithm

**The measures.** For a submatrix $I \times J$:

- **Mean of the $i$-th row:**
  $$e_{iJ} = \frac{1}{|J|} \sum_{j \in J} e_{ij}$$
- **Mean of the $j$-th column:**
  $$e_{Ij} = \frac{1}{|I|} \sum_{i \in I} e_{ij}$$
- **Mean of all the elements of the submatrix:**
  $$e_{IJ} = \frac{1}{|I|\,|J|} \sum_{i \in I,\, j \in J} e_{ij}$$

**The residue of an element** — the quantity that measures how far the element is from the "coherent values" ideal $e_{ij} = c + \alpha_i + \beta_j$:
$$\text{residue}(e_{ij}) \;=\; e_{ij} - e_{iJ} - e_{Ij} + e_{IJ}$$

**The quality of the submatrix as a bi-cluster — the MEAN SQUARED RESIDUE:**
$$H(I \times J) \;=\; \frac{1}{|I|\,|J|} \sum_{i \in I,\, j \in J} \big(e_{ij} - e_{iJ} - e_{Ij} + e_{IJ}\big)^2$$

> **Definition**: a submatrix $I \times J$ is a **δ-bi-cluster** if $H(I \times J) \le \delta$, with $\delta \ge 0$ a threshold.
> - **When $\delta = 0$, $I \times J$ is a PERFECT bi-cluster with coherent values.**
> - **By setting $\delta > 0$, the user specifies the tolerance of AVERAGE noise per element** against a perfect bi-cluster. *(In practice $\delta$ is chosen low, to get a good approximation.)*

**Maximal δ-bi-cluster**: a δ-bi-cluster $I \times J$ such that **there is no other δ-bi-cluster $I' \times J'$ containing it**. We want the **largest** submatrix that still respects the residue bound.

**Computing it is costly ⇒ heuristic greedy search for LOCAL optima. Two-phase computation:**

**Phase 1 — DELETION**
- **Start from the whole matrix**; **iteratively remove rows and columns while the mean squared residue of the matrix is over $\delta$**.
- At each iteration, compute for each row / each column its own mean squared residue:
  $$d(i) = \frac{1}{|J|}\sum_{j \in J}\text{residue}(e_{ij})^2, \qquad d(j) = \frac{1}{|I|}\sum_{i \in I}\text{residue}(e_{ij})^2$$
- **Remove the row or column with the LARGEST mean squared residue** (the biggest offender).

**Phase 2 — ADDITION**
- **Expand iteratively the δ-bi-cluster $I \times J$ obtained in the deletion phase, as long as the δ-bi-cluster requirement is maintained.**
- Consider **all the rows/columns NOT involved** in the current bi-cluster, computing their mean squared residues.
- **Add the row/column with the SMALLEST mean squared residue.**

> **Why two phases** (lecture): the deletion phase shrinks the matrix **past** what is necessary — it stops as soon as the constraint holds, giving a *minimal* solution. The addition phase then **grows it back** as far as the constraint allows, delivering a **maximal** δ-bi-cluster.

**⚠️ It finds only ONE δ-bi-cluster ⇒ it must be run multiple times, REPLACING THE ELEMENTS OF THE OUTPUT BI-CLUSTER BY RANDOM NUMBERS.**

> **Why the random replacement is indispensable** (lecture): the procedure is **completely deterministic — there is no randomness anywhere**. If we simply reran it, **we would get exactly the same δ-bi-cluster every time**. By **overwriting the discovered bi-cluster with random numbers we DESTROY that pattern**, so on the next run the old pattern is no longer there and **the algorithm is forced to look for new patterns in new subspaces**.
>
> **The price**: this generates a number of bi-clusters, but **there is NO guarantee that they are optimal** — greedy search yields **local** optima.

### 5.6 Bi-Clustering (II): δ-pCluster (enumeration)

**A different philosophy**: instead of a constraint on the **average** noise of the whole submatrix ($H(I\times J) \le \delta$), impose a constraint that must hold **on every 2×2 submatrix**, **whatever the size of the matrix** (2×2 or 10×10 — the condition is always the same).

**Starting observation**: a submatrix $I \times J$ is a bi-cluster with **perfect coherent values** iff
$$e_{i_1 j_1} - e_{i_2 j_1} = e_{i_1 j_2} - e_{i_2 j_2}$$

So, for **any 2×2 submatrix** of $I \times J$, define the **p-score**:
$$\text{p-score}\!\begin{pmatrix} e_{i_1 j_1} & e_{i_1 j_2} \\ e_{i_2 j_1} & e_{i_2 j_2}\end{pmatrix} = \big|\,(e_{i_1 j_1} - e_{i_2 j_1}) - (e_{i_1 j_2} - e_{i_2 j_2})\,\big|$$

> **Definition**: $I \times J$ is a **δ-pCluster** (pattern-based cluster) **if the p-score of EVERY 2×2 submatrix of $I \times J$ is at most $\delta$**, where $\delta \ge 0$ specifies the user's tolerance of noise against a perfect bi-cluster.

**⚠️ The key contrast to remember:**

| | **δ-bi-cluster** (mean squared residue) | **δ-pCluster** (p-score) |
|---|---|---|
| What the threshold controls | the **AVERAGE noise** over the submatrix | the noise **on EVERY element** (every 2×2 block) |
| Consequence | a few very bad elements can be **compensated** by many good ones | **no compensation possible** — one bad 2×2 block disqualifies the whole submatrix |

**MONOTONICITY** — the property that makes enumeration feasible:

> **If $I \times J$ is a δ-pCluster, then every $x \times y$ submatrix ($x, y \ge 2$) of $I \times J$ is also a δ-pCluster.**

> **This is exactly the Apriori / downward-closure property** seen in frequent pattern mining. Its practical consequence: **to build a submatrix that is a δ-pCluster we must START FROM A 2×2 submatrix having the property**, and grow from there — any submatrix containing a violating 2×2 block can be pruned immediately.

**A δ-pCluster is MAXIMAL if no more rows or columns can be added while retaining the δ-pCluster property.** **We only need to compute all the MAXIMAL δ-pClusters** — the rest follow by monotonicity.

### 5.7 MaPle: efficient enumeration of δ-pClusters

*(Pei et al., "MaPle: Efficient enumerating all maximal δ-pClusters", ICDM'03.)*

- **Framework: the same as PATTERN-GROWTH in frequent pattern mining**, based on the **downward closure property**.
- **For each condition combination $J$, find the maximal subsets of genes $I$ such that $I \times J$ is a δ-pCluster.**
- **If $I \times J$ is not a submatrix of another δ-pCluster, then $I \times J$ is a MAXIMAL δ-pCluster.**
- **The algorithm is very similar to mining frequent CLOSED itemsets.**

> **Summary of the bi-clustering block** (lecture): these algorithms **do not let us exploit all the possible bi-clusters**, but they are **very useful when we have many objects and want to know whether any pattern can be identified at all**. **Bi-clustering finds a subspace and clusters inside it in a single execution.**

## 6. Dimensionality-reduction methods

**The alternative philosophy**: *in some situations it is more effective to **construct a NEW space** instead of using some subspaces of the original data.*

**The motivating figure**: three clusters arranged so that **any subspace of the original $X$/$Y$ space cannot help** — **all three clusters project into overlapping areas on both the $X$ and the $Y$ axis**. But if we **construct a new dimension** (the dashed diagonal line), **the three clusters become apparent when the points are projected onto it**.

> **The lesson** (lecture): **subspace selection is not always enough.** When the discriminating direction is a **combination** of the original axes, no subset of the original axes will ever reveal the clusters — **we must transform the space**. *"If we can find a new dimension where the clusters are easy to separate, then we have found an optimal separation."* The result is **reduced dimensions AND separated clusters**.

**Two routes:**
- **Feature selection and extraction** — **but it may not focus on finding the clustering structure** (it optimises a different criterion, e.g. variance, not cluster separability).
- **Spectral clustering** — **combines feature extraction AND clustering**: it uses **the spectrum of the similarity matrix of the data** to perform dimensionality reduction **for clustering** in fewer dimensions.
  - **Normalized Cuts** (Shi & Malik, CVPR'97 / PAMI'2000)
  - **The Ng-Jordan-Weiss algorithm** (NIPS'01)

> **What spectral clustering does, in the lecture's words**: we adopt a **transformation with reduction of features**, in which we **find new axes along which the projection isolates the clusters**; we **apply the clustering algorithm in the reduced space** and then **come back to the original one**.

## 7. Spectral clustering: the Ng-Jordan-Weiss (NJW) algorithm

**Input**: a set of objects $o_1, \dots, o_n$, the distance $\text{dist}(o_i, o_j)$ between each pair, and the desired number $k$ of clusters.

**Step 1 — Compute the AFFINITY MATRIX $W$:**
$$w_{ij} = \exp\!\left(-\frac{\text{dist}(o_i, o_j)^2}{2\sigma^2}\right) \quad \text{for } i \ne j, \qquad w_{ii} = 0 \text{ (in NJW)}$$

- **$\sigma$ is a scaling parameter that controls how fast the affinity $w_{ij}$ decreases as $\text{dist}(o_i,o_j)$ increases** — i.e. **how rapidly the affinity matrix falls off with distance**.
- **The matrix measures the AFFINITY between objects** (large $w_{ij}$ = very similar), not their distance.

**Step 2 — Derive $A = f(W)$.** NJW defines the **diagonal matrix $D$** whose $i$-th diagonal entry is the **sum of the $i$-th row of $W$**:
$$D_{ii} = \sum_{j=1}^{n} w_{ij}$$
and then sets
$$A = D^{-1/2}\, W \, D^{-1/2}$$
(the **normalized affinity matrix**).

**Step 3 — Find the $k$ LEADING EIGENVECTORS of $A$.**
- **A vector $v$ is an eigenvector of $A$ if $A v = \lambda v$, where $\lambda$ is the corresponding eigenvalue.**
- **These eigenvectors form the new axes** of the transformed space.
- **We only retain the eigenvectors associated with the HIGH eigenvalues — this is where the dimensionality reduction happens.**

> **This is very similar to Principal Component Analysis** — with the crucial difference that the matrix being decomposed is the **affinity/similarity matrix**, not the covariance matrix of the features. That is why spectral clustering can separate **non-convex (e.g. concave, interleaved) shapes** that PCA + k-means cannot.

**Step 4 — Project and cluster.** Using the $k$ leading eigenvectors, **project the original data into the new space** they define, and **run a clustering algorithm (e.g. k-means) to find $k$ clusters**.

**Step 5 — Map back.** **Assign the original data points to clusters according to how the transformed points were assigned** in the clusters obtained.

> **The pipeline, end to end** (lecture): *start from the dataset → build the affinity matrix → compute the leading eigenvectors of $A$ → transform the data space into one with fewer dimensions → perform clustering in the new space → project back to the original space to rebuild the clustering.* **In this way we can apply a classical clustering algorithm to the whole dataset, but in a different — simpler — space.** There is **no supervision** anywhere in this approach.

### Illustration and comments

- **Spectral clustering is effective in tasks like image processing.**
- **⚠️ Scalability challenge: computing eigenvectors on a large matrix is COSTLY.**
- **It can be combined with other clustering methods, such as bi-clustering.**

**The two half-moons example** (200 data points in each half moon, from Nicola Rebagliati's slide collection, $K = 2$): the two interleaved crescents are **impossible for k-means** in the original space. The **spectral embedding given by the first two eigenvectors** turns them into **two well-separated compact blobs**, and the **partition obtained by NJW** recovers the two moons exactly.

### Summary

> **To cluster in high dimensionality we have two possible strategies** (lecture):
> 1. **Apply directly an approach that copes with high dimensionality — e.g. BI-CLUSTERING** — in which we **reduce the space by identifying a subspace and cluster inside it at the same time**. This makes it easy to identify a cluster limited to a small group of objects.
> 2. **Work with a TRANSFORMATION** that lets us select the subspace **in the transformed space**. The dimensionality is reduced by using only a subset of the new dimensions, while still being able to find the clusters.

---

# PART B — CLUSTERING WITH CONSTRAINTS (moved to Chapter 11)

> **This topic now has its own chapter file.** The lecture notes for Chapter 9 carried an early draft of the "Clustering with Constraints" material, but the topic has its own slide deck and its own lecture notes, merged into:
>
> **`11 - Constrained Clustering/CAPITOLO_11_STUDIA_SOLO_QUESTO_Clustering_with_Constraints.md`**
>
> That file is the **authoritative** one for: must-link / cannot-link constraints, δ- and ε-constraints, conversion to instance-level constraints, hard vs. soft constraints, informativeness and coherence, **COP-k-means**, **CVQE**, and **clustering with obstacles** (visibility graphs, microclusters, VV/MV indices).
>
> Nothing about constrained clustering is duplicated here, so the two files cannot drift apart. Use this file for high-dimensional clustering, and Chapter 11 for constraints.

---

## Key points / potential exam pitfalls

### High dimensionality in general
- **⚠️ The Euclidean distance does not "break" mathematically — it becomes SEMANTICALLY MEANINGLESS.** In the Ada/Bob/Cathy example all pairwise distances are identical while a human sees Ada and Cathy as similar. Be able to state the reason: **equi-distance caused by sparsity**.
- **The three challenges, in order**: **irrelevant dimensions mask clusters**; **distance becomes meaningless**; **clusters exist only in some subspaces**.
- **Feature transformation vs. feature selection**: transformation (**PCA/SVD**) is effective **when most dimensions are relevant** and features are **correlated/redundant**; **selection** is for **finding a subspace where the data cluster nicely**. They answer different situations — do not use them interchangeably.
- **The two big families are NOT the same thing**: **subspace clustering keeps the ORIGINAL dimensions** (a subset of them); **dimensionality reduction BUILDS NEW dimensions** by combining the original ones.

### Subspace search
- **Bottom-up (CLIQUE) = start low-D, grow, prune** using an Apriori-style property. **Top-down (PROCLUS) = start from the full space, recurse into smaller subspaces**, and it is **k-medoid-like**.
- **⚠️ Top-down methods are effective ONLY if the LOCALITY ASSUMPTION holds** — that the subspace of a cluster can be determined from the local neighbourhood. This is the standard exam question about PROCLUS.

### Bi-clustering
- **Bi-clustering clusters OBJECTS AND ATTRIBUTES SIMULTANEOUSLY** — one execution, not "select subspace, then cluster".
- **⚠️ Memorise the four requirements**: few objects per cluster; few attributes per cluster; **objects may belong to several clusters or to none**; **attributes may belong to several clusters or to none**. This means **no exhaustive, exclusive partition** — the opposite of k-means.
- **The four bi-cluster types, in increasing generality**: **constant values** ($e_{ij}=c$) → **constant on rows/columns** ($e_{ij}=c+\alpha_i$) → **coherent values** ($e_{ij}=c+\alpha_i+\beta_j$) → **coherent evolutions** (only the **sign/trend** matters: $(e_{i_1j_1}-e_{i_1j_2})(e_{i_2j_1}-e_{i_2j_2}) \ge 0$).
- **The coherent-values equivalence is the one used to define the p-score**: $e_{i_1j_1}-e_{i_2j_1} = e_{i_1j_2}-e_{i_2j_2}$.
- **⚠️ residue vs. p-score is THE comparison of this chapter**: **the mean squared residue captures the AVERAGE noise** (so good elements can compensate bad ones); **the p-score controls the noise on EVERY element** (no compensation). Both use the same $\delta$ letter — do not confuse the two definitions.
- **$\delta = 0$ ⇒ a PERFECT bi-cluster with coherent values**, for both definitions.
- **The δ-Cluster algorithm: DELETION removes the row/column with the LARGEST mean squared residue; ADDITION adds the one with the SMALLEST.** Getting these two backwards is the classic slip.
- **⚠️ Why the δ-Cluster algorithm must overwrite its output with RANDOM NUMBERS**: the procedure is **fully deterministic**, so rerunning it would return **the same bi-cluster forever**. Randomising the found bi-cluster **destroys the pattern** and forces the search elsewhere. **It gives no optimality guarantee** — greedy ⇒ local optima.
- **δ-pCluster MONOTONICITY**: *if $I \times J$ is a δ-pCluster, every $x \times y$ ($x,y \ge 2$) submatrix is too.* **This is the downward-closure/Apriori property**, and it is what lets MaPle enumerate. **Growth starts from a 2×2 submatrix.**
- **MaPle = pattern-growth on bi-clusters**; it is **"very similar to mining frequent CLOSED itemsets"**, and it enumerates **all MAXIMAL δ-pClusters**.

### Dimensionality reduction and spectral clustering
- **⚠️ Know the motivating figure**: three clusters that **overlap on both $X$ and $Y$** — **no subspace of the original axes helps**, only a **new (diagonal) dimension** separates them. This is the argument for *why* subspace selection is not always sufficient.
- **Feature selection/extraction "may not focus on clustering structure finding"** — it optimises a criterion (e.g. variance) that is **not cluster separability**. Spectral clustering **combines extraction and clustering**, which is precisely its selling point.
- **The NJW pipeline in order**: **affinity matrix $W$ (Gaussian kernel, $w_{ii}=0$) → diagonal $D$ of row sums → $A = D^{-1/2} W D^{-1/2}$ → $k$ leading eigenvectors → project → k-means → map assignments back to the original points.**
- **$\sigma$ controls how fast the affinity decays with distance** — it is the sensitivity knob of the whole method.
- **The eigen-decomposition is on the AFFINITY matrix, not the covariance matrix** — that is the difference from PCA, and the reason spectral clustering handles **non-convex shapes** (the two half-moons) that k-means cannot.
- **⚠️ The scalability challenge: computing eigenvectors of a large matrix is COSTLY** ($n \times n$ where $n$ is the number of objects, not of features).

### Cross-chapter connections
- **The curse of dimensionality** appears here for the fourth time (Chapters 2, 3, 5, 7) — and this chapter is the **dedicated answer** to it for clustering. Compare with **Chapter 7 §8**, where the very same phenomenon forced **subspace search and angle-based methods** for outlier detection: **subspace outliers and subspace clusters are the same idea applied to opposite goals**.
- **CLIQUE and PROCLUS** were introduced in **Chapter 5 (Clustering)**; here they are re-read as **bottom-up / top-down subspace search**. **BIRCH, DBSCAN, k-means, k-medoids/CLARANS** all return in Chapter 11.
- **PCA and SVD** come from **Chapter 3 (Data Preprocessing)**; the NJW algorithm is explicitly presented as **"very similar to PCA"**, but applied to the **affinity matrix**.
- **The monotonicity of δ-pClusters IS the downward-closure/Apriori property** of **Chapters 6 and 8**, and **MaPle is pattern-growth** — the same framework as **FP-Growth** and **closed itemset mining**. Note the contrast with **Chapter 8**, where downward closure was a *liability*; here it is again an *asset*.
- **The inadequacy of Euclidean distance for binary attributes** (the Ada/Cathy example) points back to **Chapter 2's** proximity measures for **binary/asymmetric attributes** and **cosine similarity**.
- **Constrained clustering (Chapter 11)** is the natural sequel: it attacks the same clustering problem from the opposite side — instead of reshaping the SPACE, it injects DOMAIN KNOWLEDGE as constraints.
- **Gene expression / micro-array data** is the running biological application of the course — it also motivated **colossal pattern mining** in **Chapter 8 §3**.

---

*File auto-generated by merging `9-HighDimensionalClustering.pdf` (professor's slides, 15 pages) and `9 - High dimension Clustering sbobine.pdf` (lecture notes, 21 pages). Formulas that the slide deck stored as images (the affinity matrix and its normalization $A = D^{-1/2}WD^{-1/2}$, the row/column/submatrix means, the mean squared residue, the p-score, the coherent-evolution inequality) have been reconstructed in their standard form. The "Clustering with Constraints" block that these lecture notes also contained lives in its own file, `11 - Constrained Clustering/CAPITOLO_11_STUDIA_SOLO_QUESTO_Clustering_with_Constraints.md`, built from that topic's own slides and notes. For questions about this chapter, refer only to this file.*
