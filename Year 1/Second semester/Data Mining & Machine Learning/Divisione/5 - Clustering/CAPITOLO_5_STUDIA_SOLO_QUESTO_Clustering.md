# Chapter 5 — Cluster Analysis

> **Course:** Data Mining and Machine Learning / Bioinspired Computational Methods — Biological Data Mining
> **Instructor:** Prof. Francesco Marcelloni (University of Pisa)
> **Sources merged into this file:** professor's slides (`5-Clustering.pdf`, 98 pages, based on the Han–Kamber–Pei textbook) + lecture notes (`5 - Clustering Sbobine.pdf`, 64 pages)
>
> **Instructions for Claude:** this is the single reference file for Chapter 5. When the user asks study questions about this chapter, use **only** this file as context (no need to re-read the original PDFs unless a detail is missing). The file merges the official slide content with the lecture's discursive explanations and worked examples, keeping both levels of detail.

---

## Chapter outline (official slide structure)
1. Cluster Analysis: Basic Concepts
2. Partitioning Methods
3. Hierarchical Methods
4. Density-Based Methods
5. Grid-Based Methods
6. Evaluation of Clustering
7. Summary

---

## 1. Cluster Analysis: Basic Concepts

**Cluster**: a collection of data objects that are
- **similar** (or related) to one another **within** the same group;
- **dissimilar** (or unrelated) to the objects in **other** groups.

**Cluster analysis** (also called *clustering* or *data segmentation*): finding similarities between data according to the characteristics found in the data, and grouping similar data objects into clusters.

**⚠️ Clustering is UNSUPERVISED learning**: there are **no predefined classes** — it is *learning by observations*, as opposed to *learning by examples* (supervised classification, Chapter 4). We have to discover whether we can group instances such that we find similarity between instances inside each group.

### Typical applications
- As a **stand-alone tool** to get insight into data distribution.
- As a **preprocessing step** for other algorithms:
  - **Summarization**: preprocessing for regression, PCA, classification, association analysis. (E.g., k-nearest neighbours can summarize a set of training instances by replacing them with representative prototypes.)
  - **Compression**: image processing — *vector quantization*.
  - **Finding K-nearest neighbours**: localizing search to one or a small number of clusters.
  - **Outlier detection**: outliers are often viewed as those "far away" from any cluster — instances significantly different from the rest of the data, i.e., not similar to other instances in the dataset.

### Application fields
- **Biology**: taxonomy of living things — kingdom, phylum, class, order, family, genus, species.
- **Information retrieval**: document clustering.
- **Land use**: identification of areas of similar land use in an earth observation database.
- **Marketing**: help marketers discover distinct groups in their customer bases, then develop targeted marketing programs.
- **City-planning**: identifying groups of houses according to house type, value, and geographical location.
- **Earthquake studies**: observed earthquake epicenters should be clustered along continent faults.
- **Climate**: understanding earth climate, finding patterns of atmosphere and ocean.
- **Economic science**: market research.

### What is a good clustering?

> **The fundamental difficulty** (lecture): in clustering there is **no ground truth** — the desired outcome is not predefined and no labeled data are provided. Consequently, it is **not known a priori** whether certain instances should belong to the same cluster. Therefore the quality of a clustering result must be assessed based on the **intrinsic properties of the clusters themselves**.

A good clustering method produces **high-quality clusters**, characterized by:
- **High intra-class similarity**: cohesive **within** clusters (instances in the same cluster are highly similar).
- **Low inter-class similarity**: distinctive **between** clusters (instances in different clusters are well separated).

The quality of a clustering method depends on:
- the **similarity measure** used by the method,
- its **implementation**, and
- its **ability to discover** some or all of the hidden patterns.

There is usually a separate **"quality" function** that measures the "goodness" of a cluster. **It is hard to define "similar enough" or "good enough"** — the answer is typically **highly subjective** (only a qualitative evaluation).

### Dissimilarity / Similarity metric
- Similarity is expressed in terms of a **distance function**, typically a metric: $d(i,j)$. (Two objects are really dissimilar if they are far from each other in terms of some distance.)
- The definitions of distance functions are usually **rather different** for interval-scaled, boolean, categorical, ordinal, ratio, and vector variables *(see Chapter 2, §5)*.
- **Weights** should be associated with different variables based on applications and data semantics.

### Considerations for Cluster Analysis
These describe clustering properties and assumptions — cluster structure, membership, similarity measure, and the feature space in which clustering is performed. They are the criteria used to characterize a clustering algorithm:

- **Partitioning criteria**
  - **Single level vs. hierarchical partitioning** (often multi-level hierarchical partitioning is desirable). In single-level partitioning we apply a clustering algorithm to obtain one set of clusters. In hierarchical partitioning we first obtain coarse-grained clusters (simpler but less precise) and then iteratively refine them into smaller clusters, building a **cluster hierarchy** from more general to more detailed.
- **Separation of clusters**
  - **Exclusive** (e.g., one customer belongs to only one region) vs. **non-exclusive** (e.g., one document may belong to more than one class **with different degrees of membership**, as in **fuzzy clustering**: a document could be 0.7 in one cluster and 0.3 in another).
- **Similarity measure**
  - **Distance-based** (e.g., Euclidean, road network, vector) vs. **connectivity-based** (e.g., density or contiguity → we define a cluster as a **dense zone** of points, so similarity relates to connectivity rather than a real distance).
- **Clustering space**
  - **Full space** (often when low-dimensional; we apply the algorithm to all instances) vs. **subspaces** (often in high-dimensional clustering).
  - A well-known approach is **bi-clustering**, where we **simultaneously** identify subsets of instances *and* subsets of features that exhibit coherent patterns — particularly useful with high-dimensional data.
  - **⚠️ Do not confuse** bi-clustering with **subspace clustering**, which clusters data in different feature subspaces: each cluster may have its own subset of relevant features (**features are not clustered**). Bi-clustering searches for **coherent blocks** in the data matrix and simultaneously identifies clusters of instances *and* clusters of features.

### Requirements and Challenges
- **Scalability**: clustering all the data instead of only on samples.
- **Ability to deal with different types of attributes**: numerical, binary, categorical, ordinal, linked, and mixtures of these.
- **Constraint-based clustering** → *not completely unsupervised*: we introduce external information to obtain clusters more coherent with reality.
  - The user may give inputs on constraints (e.g., two instances should stay in the same cluster, or limits on dimension).
  - Use **domain knowledge** to determine input parameters (e.g., define thresholds, impose real constraints, or fix the number of clusters).
- **Interpretability and usability**: when we generate clusters we should provide interpretability, because people want to know the reason why we grouped things that way.
- **Others**:
  - Discovery of clusters with **arbitrary shape**.
  - Ability to deal with **noisy data**.
  - **Incremental clustering and insensitivity to input order** (some applications receive data in **streaming**, so we want to continuously update the clusters).
  - **High dimensionality** (to cope with the curse of dimensionality).

### Major Clustering Approaches (overview)

| Approach | Idea | Key traits | Typical methods |
|---|---|---|---|
| **Partitioning** | Construct various partitions, evaluate them by some criterion, e.g. minimizing the **sum of squared errors** (distance-based). Given $k$, use an **iterative relocation** technique that improves the partitioning by moving objects between groups. | Finds **mutually exclusive** clusters of typically **spherical** shape; may use mean or medoid to represent the cluster center. Effective for **small- to medium-size** datasets. | **k-means, k-medoids, CLARANS** |
| **Hierarchical** | Create a **hierarchical decomposition** of the set of objects using some criterion. Two strategies: **agglomerative vs. divisive**. | **Con**: once a step (merge or split) is done, it can **never be undone** — cannot correct erroneous merges/splits. May incorporate microclustering or object "linkages". | **Diana, Agnes, BIRCH** (also incremental), **CHAMELEON** |
| **Density-based** | Based on **connectivity and density functions** (not on distance). Clusters are **dense regions** of objects in space, separated by **low-density** regions. Each point must have a minimum number of points within its "neighborhood". | Can find **arbitrarily shaped** clusters with different convexities and concavities. Fixing the parameters indirectly fixes the number of clusters. **Con**: may filter out outliers. | **DBSCAN, OPTICS, DenClue** |
| **Grid-based** | Based on a **multiple-level granularity structure** (speeds up clustering). All clustering operations are performed **on the grid structure** (the quantized space). | Grid resolution is typically **independent of the number of data objects**, but dependent on grid size. **Fast processing time**. Cluster precision depends on grid granularity. | **STING, WaveCluster, CLIQUE** |

**Other methods (only mentioned in class):**
- **Model-based**: a model is hypothesized for each cluster, then we find the best fit of that model. Typical: **EM, SOM, COBWEB**.
- **Frequent pattern-based**: based on analysis of frequent patterns. Typical: **p-Cluster**.
- **User-guided or constraint-based**: clustering considering user-specified or application-specific constraints. Typical: **COD** (obstacles), constrained clustering.
- **Link-based clustering**: objects are often linked together in various ways; massive links can be used to cluster objects. Typical: **SimRank, LinkClus**.

---

## 2. Assessing Clustering Tendency

> **The core problem** (lecture): assume the data points are drawn from a **uniform distribution**. If a clustering algorithm is applied to such a dataset, it will **always produce clusters** — but these clusters do not correspond to any underlying natural structure and are therefore **not meaningful**; they simply result from an artificial partitioning of the data.
>
> **Therefore, before applying any clustering method, it is essential to assess whether the dataset exhibits a clustering tendency**, i.e., the presence of non-random structures.

### The Hopkins Statistic

A **spatial statistic** that tests the **spatial randomness** of a variable as distributed in a space. It assesses clustering tendency by measuring the **probability that a given dataset is generated by a uniform data distribution**.

Given a dataset $D$, regarded as a sample of a random variable $X$, we want to determine how far $X$ is from being uniformly distributed in the data space.

**Method:**

**Step 1 — real points.** Sample $n \ll N$ points $p_1,\dots,p_n$ **uniformly from $D$** (i.e., actual objects in the dataset). For each point $p_i$, find its **nearest neighbor in $D$**. Let $x_i$ be that distance:
$$x_i = \min_{v\in D}\{dist(p_i, v)\}$$

**Step 2 — synthetic points.** Generate a **simulated dataset** drawn from a random **uniform distribution** with $n$ points $q_1,\dots,q_n$, having the **same variation** as the original dataset $D$. (I.e., synthetic objects uniformly distributed in the plane.) For each point $q_i$, find its nearest neighbor **in $D$**. Let $y_i$ be that distance:
$$y_i = \min_{v\in D,\ v\ne q_i}\{dist(q_i,v)\}$$

**Step 3 — compute the statistic:**

$$H = \frac{\sum_{i=1}^{n} y_i}{\sum_{i=1}^{n} x_i + \sum_{i=1}^{n} y_i}, \qquad 0.5 \le H \le 1$$

> **The intuition** (lecture): in the case of natural clusters, we expect **higher distances between synthetic points and real points**, because at least some synthetic points will fall **outside** the cluster and far from other points — while the real sampled points sit inside dense clusters, close to their neighbours.

**Interpretation:**
- **$H \approx 0.5$**: the dataset is **uniformly distributed** in the data space → **no clustering tendency** (the two terms at the denominator would be close to each other).
- **$0.7 \le H \le 0.9$**: the dataset shows a **significant clustering tendency**. In the presence of clustering tendency, $\sum x_i$ (the first term at the denominator) becomes smaller than $\sum y_i$, so $H$ increases.
- **$H \approx 1$**: the dataset contains **highly separated clusters** (natural clusters are present, but we don't know how many).

**Hypothesis testing framework:**
- **Null hypothesis**: the dataset $D$ is **uniformly distributed** (no meaningful clusters).
- **Alternative hypothesis**: the dataset $D$ is **not** uniformly distributed (contains meaningful clusters).

> To properly evaluate the value of $H$ we would need to know its distribution, which serves as a reference. However, **there is no universal distribution** for the statistic — it depends on the underlying assumption about the data. The general idea is to adopt a reference model and analyze the behavior of the statistic under that assumption, to assess whether the observed value is statistically significant.

**Decision rule**: the Hopkins statistic is computed for **several random selections of points**, and the **average** of all results for $H$ is used for the decision: **if $H > 0.75$, it indicates a clustering tendency at the 90% confidence level.**

> **TO SUM UP**: if we apply a clustering method we **always** obtain clusters — but they could represent natural clusters *or* just a partition of uniform data (in which case they are meaningless).

---

## 3. Partitioning Methods

### 3.1 Basic Concept

The goal is to divide a database $D$ of $n$ objects into a set of $k$ clusters, such that the **sum of squared distances is minimized**. Given a fixed number of clusters $k$, the objective is to find a partition that minimizes the chosen cost function — typically defined as the sum of squared distances between each object and the representative (**centroid** or **medoid**) of its cluster. **This leads to the formation of compact clusters.**

**The objective function:**

$$E = \sum_{i=1}^{k}\sum_{p\in C_i}\lVert p - c_i\rVert^2$$

where $c_i$ is the centroid (or medoid) of cluster $C_i$, and $k$ is a **predefined** number of clusters.

> If all clusters are compact, they are also expected to be well separated from each other. **However, partitioning methods always produce a clustering, regardless of whether a true cluster structure exists in the data** — they do not assess the existence of meaningful clusters.

**Finding the global optimum would require exhaustively enumerating all possible partitions**, which is computationally infeasible. Hence **heuristic methods**:
- **k-means** (MacQueen 1967; Lloyd 1957/1982): each cluster is represented by its **centroid** (the mean of the points in the cluster). The most widely used method.
- **k-medoids / PAM** (Partitioning Around Medoids) (Kaufman & Rousseeuw 1987): each cluster is represented by **one of the actual data points** (the **medoid**), making it more robust to outliers.

### 3.2 The k-means Clustering Method

The centroid is defined as the mean value of the points within the cluster:

$$c_i = \frac{1}{|C_i|}\sum_{p\in C_i}p$$

**Input:**
- $k$: the number of clusters (**required as an input!**)
- $D$: a data set containing $n$ objects

**Output**: a set of $k$ clusters.

**Method:**
1. **Arbitrarily choose $k$ objects from $D$** as the initial cluster centers (partition objects into $k$ nonempty subsets).
2. **Repeat:**
   - **(Re)assign** each object to the cluster to which the object is **most similar**, based on the mean value of the objects in the cluster (we only have the centroids, so we assign each object to the **closest centroid**).
   - **Update** the cluster means — recalculate the mean value of the objects for each cluster.
3. **Until** no change in the centroids or in cluster membership.

**Strength:**
- **Efficient**: $O(t\,k\,n)$, where $n$ = number of objects, $k$ = number of clusters, $t$ = number of iterations. Normally $k, t \ll n$.
  - Compare: **PAM** $O(k(n-k)^2)$, **CLARA** $O(k s^2 + k(n-k))$, where $s$ is the sample size.
- **Comment**: often terminates at a **local optimum**.

**Weaknesses:**

| Weakness | Explanation (from the lecture) |
|---|---|
| **Need to specify $k$ in advance** | We must assign a number of clusters to a dataset whose structure we don't know. (There are ways to determine the best $k$ — see the Elbow method.) |
| **Sensitive to initialization** | We select the centroids at the beginning. Since we minimize the cost function with an optimization approach, we may **fall into a local minimum** and not achieve the best clustering. **Remedy**: normally we **run k-means several times with different initializations** (or use initialization algorithms from the literature), and at the end we choose the partition with the **minimum value of the cost function**. This mitigates the dependency on initialization. |
| **Not suitable for non-convex shapes** | In classical k-means, similarity is measured with the **Euclidean distance**. Using the Euclidean distance **determines the shape of the clusters**, because all dimensions are weighted the same way, producing approximately **spherical** clusters. *(The shape of the clusters is always determined by the type of distance used.)* Other distances, such as **weighted distances**, can yield **ellipsoidal** clusters — but with these types of distances we can only obtain **convex** clusters. |
| **Applicable only to objects in a continuous $n$-dimensional space** | Use **k-modes** for categorical data, or **k-medoids** for a wider range of data types. |
| **Sensitive to noisy data and outliers** | In the k-means pseudocode, **every object must belong to a cluster**. So if we fix $k=2$, an outlier will still be assigned to one of them. Since the centroid is the **mean** of the points, the presence of an outlier can **significantly shift the centroid** toward it. **Remedy: k-medoids.** |

### 3.3 Determining the Number of Clusters: the Elbow Method

**The idea**: start with a small value of $k$ and gradually increase it (e.g. from $k=2$ up to $k=10$). For each $k$, compute the cost function (sum of within-cluster variances, $var(k)$), and **plot the curve of $var$ with respect to $k$**. Choose $k$ as the **first or most significant turning point** ("elbow") of the curve.

**Why it works:**
- Increasing $k$ helps **reduce** the sum of within-cluster variance, since each cluster becomes more compact.
- But the **marginal reduction** drops when too many clusters are formed, because **splitting an already cohesive cluster into two gives only a small reduction**.
- **Extreme case**: when $k$ = number of objects, the cost function becomes **zero** — each object forms its own cluster and coincides with its own centroid. So the cost function **always decreases** as $k$ increases; that's why we need the turning-point intuition rather than just minimizing.

**Worked reasoning (from the lecture):**
- With $k=2$: the cost is high, because each centroid must cover many points possibly belonging to different natural clusters → clusters are not very compact.
- With $k=3$ (say, the true number): clusters better represent the natural structure and within-cluster variance **drops sharply**.
- With $k=4$: we start **splitting a natural cluster** into two. The distance between the centroid and its points decreases, but only **marginally**, because those points were already close to each other.

→ Hence the characteristic curve: the cost **decreases rapidly until the natural clusters are identified**, and afterwards decreases much more slowly. **The optimal $k$ is at the elbow.**

> **Bonus diagnostic**: if we obtain a curve **without an elbow** (a smooth decline), this indicates a **uniform distribution** — another way to verify whether natural clusters exist.
>
> **⚠️ Note**: the elbow method is **very specific to partitioning problems**, where we want to define convex clusters.

**Alternative — cross-validation method** *(from the slides)*:
- Divide the dataset into $m$ parts.
- Use $m-1$ parts to obtain a clustering model.
- Use the remaining part to test the quality: e.g., for each point in the test set, find the closest centroid, and use the **sum of squared distances** between all test points and their closest centroids to measure how well the model fits.
- For any $k>0$, repeat $m$ times, compare the overall quality measure across different $k$, and find the number of clusters fitting the data best.

### 3.4 Variations on the k-means Method

Most variants of k-means differ in:
- **Selection of the initial $k$ means**
- **Dissimilarity calculations**
- **Strategies to calculate cluster means**

**Handling categorical data: k-modes**
> We cannot apply k-means directly to categorical data because we cannot compute an average — the centroid cannot be calculated. However, methods exist to compute **distances** between categorical data.
- Replace the **means** of clusters with **modes** (the values that occur **most frequently** in a dataset).
- Use **new dissimilarity measures** to deal with categorical objects.
- Use a **frequency-based method** to update the modes of clusters.

**A mixture of categorical and numerical data: the k-prototype method.**

### 3.5 The k-Medoids Method (PAM)

**Medoid**: the **most centrally located object** in a cluster. Unlike k-means, the prototype of the cluster is **not the mean of the points** but **one of the actual objects in the dataset**.

**The idea:**
- Initial representative objects (medoids) are chosen **randomly**.
- The iterative process of replacing representative objects by non-representative objects continues as long as the **quality of the clustering improves**.
- For each **representative object $O$**, and for each **non-representative object $R$**, **swap $O$ and $R$** and evaluate the effect.
- **Choose the configuration with the lowest cost.**
- The **cost function** is the difference in absolute error-value when a current representative object is replaced by a non-representative object.

**PAM pseudocode:**
```
Input:  k: the number of clusters
        D: a data set containing n objects
Output: a set of k clusters

Method:
  Arbitrarily choose k objects from D as representative objects (seeds)
  Repeat
      Assign each remaining object to the cluster with the nearest representative object
      For each representative object O_j
          Randomly select a non-representative object O_random
          Compute the total cost S of swapping O_j with O_random
          if S < 0 then replace O_j with O_random
  Until no change
```

> **Comment from the lecture**: the algorithm is quite simple but also **very random**, because we don't know how many times we will have to recompute. It works, but with limitations.

**Effect of changing the medoid** (three cases illustrated in the lecture): (1) the assignment of point P to cluster A doesn't change, because A is still the nearest medoid; (2) P is **reassigned** to A; (3) the medoid of the second cluster changes and P is reassigned to the new B.

**Strengths and weaknesses:**
- **PAM is more robust than k-means in the presence of noise and outliers**, because a **medoid is less influenced** by outliers or extreme values than a mean.
- **PAM works efficiently for small datasets but does not scale well for large datasets** — it becomes very computationally complex: **$O(k(n-k)^2)$ for each iteration**.

### 3.6 Speeding up PAM: CLARA and CLARANS

**CLARA** (Clustering LARge Applications — Kaufmann & Rousseeuw, 1990):
- **Draws multiple samples** of the dataset, applies **PAM on each sample** to find the medoids, and gives the **best clustering** as output.
- So we don't use the whole dataset, only samples.

**CLARANS** ("Randomized" CLARA — Ng & Han, 2002):
- The clustering process can be presented as **searching a graph** where **every node is a potential solution**, i.e. a **set of $k$ medoids**.
- **Two nodes are neighbours** if their sets differ by **only one medoid**. Formally, $S_1=\{O_{m1},\dots,O_{mk}\}$ and $S_2=\{O_{w1},\dots,O_{wk}\}$ are neighbours **iff** the cardinality of the intersection $|S_1 \cap S_2| = k-1$.
- Each node is associated with a **cost**, defined as the **total dissimilarity between every object and the medoid of its cluster**. The problem corresponds to searching for a **minimum on the graph**.
- At each step, a **subset of the neighbours** of the current node is explored (**randomly sampled**), and the one providing the **deepest descent in cost** is chosen as the next solution.

**In addition to $k$, CLARANS requires two parameters:**
- **`num_local`**: the number of **local minima** (i.e., independent searches) to be performed.
- **`max_neighbor`**: the **maximum number of neighbours examined** before stopping the current search and restarting from a new random solution.

### 3.7 PAM vs CLARA vs CLARANS — the comparison

**The main difference lies in HOW THE SEARCH SPACE IS EXPLORED:**

| Algorithm | Search Strategy | Advantage | Limitation |
|---|---|---|---|
| **PAM** | Examines **all** neighbouring solutions | Accurate and robust; guarantees a thorough search | Computationally expensive for large datasets |
| **CLARA** | Applies PAM on **samples** of the dataset (draws a sample of nodes at the beginning of a search) | Faster for large datasets | Quality depends on the **representativeness of the samples** |
| **CLARANS** | **Randomly samples neighbours** in the solution space, dynamically at each step; finds several local optima and returns the best | Better exploration of solutions; not restricted to a fixed sample | May still require significant computation |

> These algorithms are often used as an alternative to k-means when **robustness to noise and outliers** is required, since they rely on **medoids** instead of centroids. However, they are generally **more computationally expensive**.

---

## 4. Hierarchical Methods

In hierarchical clustering, we consider the whole dataset and **iteratively partition it into clusters, creating a hierarchy**. Initially, the dataset can be split into two clusters, then each can be further divided, forming a hierarchical structure.

> **Why useful**: when a new data point is observed, it can be assigned at **different levels of the hierarchy** depending on the desired level of granularity.

**Two main approaches:**
- **Agglomerative (AGNES)** — *bottom-up*: start with **one cluster per object** and progressively **merge** clusters at different levels, until all objects belong to a single cluster. Clusters $C_1$ and $C_2$ may be merged if an object in $C_1$ and an object in $C_2$ form the **minimum Euclidean distance** between any two objects belonging to different clusters (one possible way to define similarity between clusters).
- **Divisive (DIANA)** — *top-down*: start with **all objects in a single cluster** and progressively **split** it into smaller clusters. A cluster can be divided according to some principle, e.g. considering the **maximum Euclidean distance** between the closest neighbouring objects in the cluster.

```
Step 0  Step 1  Step 2  Step 3  Step 4
                                          agglomerative (AGNES)
 a ──┐
     ├─ ab ──┐
 b ──┘       │
             ├─ abcde
 c ──┐       │
     ├─ cde ─┘
 d ─┬┘
    ├─ de
 e ─┘
                                          divisive (DIANA)
Step 4  Step 3  Step 2  Step 1  Step 0
```

### 4.1 The core problem: linkage metrics

> **The fundamental difference from partitioning** (lecture): in partitioning methods we reason about similarity between an **object and the centroid** of a cluster. Here, to decide whether to merge, we must define **similarity/distance between CLUSTERS**. This is the main problem.

- Hierarchical clustering frequently deals with the **matrix of distances (dissimilarities)** or similarities between training samples — also called the **connectivity matrix**. We **don't start from objects**: we compute similarity/dissimilarity between objects and start from a **distance matrix** as the clustering criterion.
- **This method does NOT require the number of clusters $k$ as an input**, but needs a **termination condition**.
- To merge or split **subsets** of points rather than individual points, the distance between individual points has to be **generalized to the distance between subsets**. Such a derived proximity measure is called a **linkage metric**, constructed from elements of the connectivity matrix.
- **The type of linkage metric used significantly affects hierarchical algorithms**, since it reflects the particular concept of closeness and connectivity.

**Major inter-cluster linkage metrics**: the underlying dissimilarity measure (usually distance) is computed for **every pair of points** with one point in the first set and another in the second set. Then a specific operation — **minimum** (single link), **average** (average link), or **maximum** (complete link) — is applied:

$$d(C_1,C_2) = \text{operation}\{\,d(x,y)\ \mid\ x\in C_1,\ y\in C_2\,\}$$

**Single link (nearest neighbour)**: the distance between two clusters is determined by the distance between the **two closest** objects (nearest neighbours) belonging to different clusters.
$$d_{single}(A,B) = \min_{x\in A,\,y\in B} d(x,y)$$
> This rule tends to **"string objects together"** to form clusters, and the resulting clusters often represent **long chains** — a phenomenon known as the **chaining effect**.

**Complete link (furthest neighbour)**: the distance between two clusters is determined by the **greatest** distance between any two objects (furthest neighbours) belonging to different clusters.
$$d_{complete}(A,B) = \max_{x\in A,\,y\in B} d(x,y)$$
> This method usually performs well when objects form naturally distinct and compact groups ("clumps"). If the clusters tend to be **elongated or chain-like**, this method is **inappropriate**.
>
> **In practice**: we compute the complete-link distance for all possible pairs of clusters and merge the two clusters whose **maximum** pairwise distance is the **smallest** (i.e., the **minimum of the maximum** distances).

**Pair-group average**: the distance between two clusters is the **average** distance between all pairs of objects in the two different clusters.
> Very efficient when objects form natural distinct "clumps", **but it performs equally well with elongated, "chain"-type** structures — this is its advantage over complete link.

**The four distance measures (formal):**

$$dist_{min}(C_i,C_j) = \min_{p\in C_i,\,p'\in C_j}\{\lVert p-p'\rVert\}$$
$$dist_{max}(C_i,C_j) = \max_{p\in C_i,\,p'\in C_j}\{\lVert p-p'\rVert\}$$
$$dist_{mean}(C_i,C_j) = \lVert m_i - m_j\rVert$$
$$dist_{avg}(C_i,C_j) = \frac{1}{n_i n_j}\sum_{p\in C_i}\sum_{p'\in C_j}\lVert p-p'\rVert$$

### 4.2 Algorithms derived from the distance choice

- **Nearest-neighbour clustering algorithm** (a.k.a. **minimal spanning tree algorithm**): when an algorithm uses the **minimum** distance to measure the distance between clusters.
  - **Single-linkage algorithm**: the clustering process is **terminated when the minimum distance between the nearest clusters exceeds a user-defined threshold**.
  - If we view data points as **nodes of a graph**, with edges forming a path between nodes in a cluster, then merging two clusters $C_i, C_j$ corresponds to **adding an edge between the nearest pair of nodes** in $C_i$ and $C_j$. **The resulting graph will generate a tree.**
- **Farthest-neighbour clustering algorithm**: when an algorithm uses the **maximum** distance to measure the distance between clusters.
  - **Complete-linkage algorithm**: the clustering process is terminated when the **maximum** distance between nearest clusters exceeds a user-defined threshold.
  - Viewing data points as nodes of a graph with edges linking nodes, we can think of each cluster as a **complete subgraph** — with edges connecting **all** nodes in the cluster.
  - **Tends to minimize the increase in diameter** of the clusters at each iteration. **High quality** in the case of clusters that are compact and of approximately equal size.

### 4.3 The Dendrogram

The final aim is to create **different levels corresponding to different partitions**. A tree structure called a **dendrogram** is used to represent the process of hierarchical clustering.
- **Level 0** corresponds to all objects being separate (leaves); at each level we represent, through links, which objects belong to the same cluster.
- **A clustering of the data objects is obtained by CUTTING the dendrogram at the desired level**; each connected component then forms a cluster.
- **Cutting at different levels gives different partitions of the data.**

> **⚠️ Important**: the shape of these clusters is in any case a **convex shape** — this **does not change** by using the hierarchical approach instead of partitioning.

> **TO SUM UP**: we don't need to fix parameters (in particular not $k$), only choose the **type of linkage metric** to use — and the partition we obtain is a **convex cluster**, as in the partitioning method.

### 4.4 AGNES (Agglomerative Nesting)

Introduced in Kaufmann and Rousseeuw (1990). Uses the **single-link method** and the **dissimilarity matrix**.
- Initially, **each object is placed into its own cluster**.
- **Clusters are merged according to some criterion** — for instance, if the distance between two objects belonging to two different clusters is the minimum distance between any two objects from different clusters (single-linkage approach).
- In the single-linkage approach, **each cluster is represented by all of the objects in the cluster**, and the similarity between two clusters is measured by the similarity of the **closest pair** of data points belonging to different clusters.
- **The cluster merging process repeats until all objects are eventually merged into one cluster.**

**Worked example (5 objects A–E, single link, from the dissimilarity matrix):**
- We start from **five clusters**, each containing a single element (level 0 of the dendrogram).
- **$d = 1$**: merge the clusters characterized by the **lowest distance**. Considering A: the closest point is B, at distance 1 (and vice versa). Considering C: minimum distance is also 1, with respect to D. For E: there is no point at distance 1.
  → We obtain **two merges: A∪B and C∪D**, giving clusters **{AB}, {CD}, {E}** (level 1).
- **Recompute the distance matrix** and repeat at this level. **$d = 2$**: merge **(A∪B) ∪ (C∪D)** → **{ABCD}, {E}** (level 2).
- **$d = 3$**: finally connect **(ABCD) ∪ (E)** → a single cluster (level 3).

### 4.5 DIANA (Divisive Analysis)

Introduced in Kaufmann and Rousseeuw (1990). It is the **inverse order of AGNES**. The algorithm constructs a hierarchy of clusters **starting with one large cluster containing all $n$ samples**; clusters are divided until each cluster contains only a **single sample**.

**At each stage**, the cluster with the **largest dissimilarity between any two of its samples** is selected (because it is not a natural cluster). To divide the selected cluster:
- The algorithm first looks for its **most disparate sample** — the one with the **largest average dissimilarity** to the other observations of the selected cluster. This observation **initiates the "splinter group"**.
- In subsequent steps, the algorithm **reassigns observations that are closer to the "splinter group" than to the "old party"**. The result is a division of the selected cluster into two new clusters.

**The Algorithm:**
1. **Find the object which has the highest average dissimilarity to all other objects.** This object initiates a new cluster — a sort of **splinter group**.
2. For each object $i$ **outside** the splinter group, compute
$$D_i = [\text{average } d(i,j),\ j \notin \text{splinter group}] - [\text{average } d(i,j),\ j \in \text{splinter group}]$$
3. **Find an object $h$ for which the difference $D_h$ is the largest.** If $D_h$ is **positive**, then $h$ is on average **closer to the splinter group** → move it there.
4. **Repeat Steps 2 and 3 until all differences $D_h$ are negative** (it no longer makes sense to move any object to the new group). The dataset is then split into two clusters.
5. **Select the cluster with the largest diameter.** The **diameter** of a cluster is the **largest dissimilarity between any two of its objects**. Then divide this cluster following steps 1–4.
6. **Repeat Step 5 until all clusters contain only a single object.** *(This is the ideal termination; most of the time, with many objects, we stop before.)*

**Worked example (from the lecture, one feature per object):**
- Start with **all objects in a single cluster**. Look for the **most dissimilar object** by computing the **mean value for each row** of the dissimilarity matrix (we compute the mean because we want the *average* dissimilarity between that object and all others).
- Split the dataset into two clusters: one containing the most dissimilar object, the other containing all remaining objects.
- Now decide whether to **freeze** the configuration or **move more points** from the larger cluster to the new one — since we don't know if the objects inside the larger cluster are all similar to each other. To decide, find the most dissimilar element in the large cluster and compute the mean dissimilarity **excluding the row and column of the object just removed**.
- In this case the second object turns out to be more similar to the new cluster than to the original one → **move it**. Repeat for the remaining objects.
- **The procedure stops when the remaining most-dissimilar point is more similar to the original cluster** (X5 in the example) — i.e., all $D_h$ are negative.
- Finally, decide which cluster to split further by comparing the **diameters** ($diam\ B_4 > diam\ A_4$ → split $B_4$). Continue until each object forms its own cluster.

> **Dendrogram direction**: in **agglomerative** clustering we start from the leaves and generate the dendrogram **bottom-up**; in **divisive** clustering we create it **top-down**. At each step we generate a level in the dendrogram.

### 4.6 Strengths and Weaknesses of Hierarchical Clustering

**Major weaknesses:**
- **Can never undo what was done previously** — no possibility to improve the generated hierarchy (a wrong merge/split is permanent).
- **Do not scale well**: time complexity of **at least $O(n^2)$**, where $n$ is the number of total objects.

**Major strengths:**
- **You get a hierarchy** instead of an amorphous collection of groups. At each level we obtain a different partition → **many possible solutions**, and we can **cut the dendrogram at different levels** to choose the best.
- **No need to specify $k$.** If you want $k$ groups, just **cut the $(k-1)$ longest links**.
- **In general they give better-quality clusters than k-means–like methods.**

### 4.7 Extensions: integrating hierarchical and distance-based clustering
- **BIRCH** (1996): uses a **CF-tree** and incrementally adjusts the quality of sub-clusters. **Can manage streaming data.**
- **CHAMELEON** (1999): hierarchical clustering using **dynamic modeling**.

### 4.8 BIRCH (Balanced Iterative Reducing and Clustering Using Hierarchies)

An **agglomerative** clustering algorithm designed for clustering a **large amount of numerical data**.

**What problem does BIRCH solve?**
- Most existing algorithms **do not consider** the case where datasets are **too large to fit in main memory**.
- They **do not concentrate on minimizing the number of scans** of the dataset. **I/O costs are very high.**
> Contrast with k-means: we have to iterate several times, and at **each** iteration we compute distances between objects and centroids — i.e., we **scan the entire dataset** repeatedly.

**Complexity: $O(n)$**, where $n$ is the number of objects to be clustered. When we use BIRCH, we essentially **store a summarization of the objects rather than the objects themselves**.

#### The insertion mechanism (walkthrough from the lecture)

BIRCH associates a **diameter** with each cluster:
- If a new object $O_2$ arrives and lies **inside the diameter** of cluster 1, it **belongs to that cluster**. If cluster 1 becomes **too large** after adding $O_2$, a **new cluster is generated** starting from $O_2$; when a new cluster is created, we **update the node** that stores the new entry.
- Same for $O_3$: if cluster 1 becomes too large, split again by creating a new cluster and updating the node with the link to it.
- When $O_4$ arrives, if cluster 2 remains compact (inside the fixed diameter), $O_4$ belongs to cluster 2. In this case we **do not need to create a new node**, but we still need to **update some information** in the tree, because the cluster no longer contains just a single point.
- $O_5$ cannot be added to any existing cluster → generate a new cluster. **But there is a limit on the number of entries a node can contain**. Therefore we must **create a new level** in the tree: **split the leaf node into two leaf nodes and generate a non-leaf node pointing to them**.
- This continues as new objects arrive. When the number of entries exceeds the limit, new levels are created. **The tree is built dynamically and stores only summarized information.**

#### Clustering Feature (CF) — the key idea

**How can we express the information in a cluster in a very reduced way?** BIRCH exploits the **Clustering Feature**, three numbers summarizing the statistics for a given cluster — the **0-th, 1st and 2nd moments** of the cluster from a statistical point of view. They are used to **compute centroids** and to **measure the compactness and distance** of clusters.

$$CF = (N,\ LS,\ SS)$$

where:
- **$N$** = the number of objects (data points) in the cluster,
- **$LS = \sum_{i=1}^{N}x_i$** = the **linear sum** of the data points,
- **$SS = \sum_{i=1}^{N}x_i^2$** = the **squared sum** of the data points.

**A CF entry has sufficient information to calculate centroid, radius, diameter and many other measures:**

**Centroid** — the "middle" of a cluster:
$$\vec{x_0} = \frac{\sum_{i=1}^{n}x_i}{n} = \frac{LS}{n}$$

**Radius** — square root of the average distance from any point of the cluster to its centroid:
$$R = \sqrt{\frac{\sum_{i=1}^{n}(x_i - x_0)^2}{n}} = \sqrt{\frac{n\cdot SS - LS^2}{n^2}}$$

**Diameter** — square root of the average mean squared distance between **all pairs** of points in the cluster:
$$D = \sqrt{\frac{\sum_{i=1}^{n}\sum_{j=1}^{n}(x_i-x_j)^2}{n(n-1)}} = \sqrt{\frac{2n\cdot SS - 2LS^2}{n(n-1)}}$$

> **⚠️ Why this enables streaming** (the crucial property): when a new point arrives we **do not need to recompute everything from scratch**, because all these values are obtained by **summation**. We simply **add the new value to $LS$**, **add its squared value to $SS$**, and **increase $N$ by 1**. By maintaining only these three values, cluster statistics can be **updated continuously**.

**Additivity across the tree**: at the node level, the CF is obtained as the **sum of the CFs of its children** (e.g., $CF_{node} = CF_{C1} + CF_{C2}$). So the information stored above the leaves is the **aggregation of the CFs contained in the leaf nodes** — we have a summarization at each level. This is useful because when a new point arrives, we can decide where it should go **just by analyzing which centroid is closest**.

#### The CF-Tree

A **CF-tree** is a **height-balanced tree** that stores the clustering features for a hierarchical clustering.

**Parameters:**
- **$B$ = Branching factor**: the maximum number of **children per non-leaf node**.
- **$T$ = Threshold parameter**: the maximum **diameter of subclusters** stored at the leaf nodes.
- **$L$ = Max number of entries in a leaf.**

**Key rules — REMEMBER:**
- **A leaf node represents a cluster.**
- A **sub-cluster in a leaf node must have a diameter no greater than the threshold $T$**.
- A point is **inserted into the leaf node (cluster) to which it is closer**.
- When an item is inserted into a cluster at a leaf node, the restriction $T$ must be satisfied, and **the corresponding CF must be updated**.
- **If there is no space on the node, the node is split.**
- **CF entry in parent = sum of CF entries of its children.**
- **Leaf nodes are connected via `prev` and `next` pointers.**

#### The BIRCH algorithm — four phases

**Phase 1: scan DB to build an initial in-memory CF tree.**
- If the **threshold condition is violated**:
  - If there is **room to insert** → insert the point as a single cluster.
  - If **not**:
    - **Leaf node split**: take the **two farthest CFs** and create two leaf nodes; put the remaining CFs (including the new one) into the **closest** node.
    - **Update CF for non-leaves.** Insert a new non-leaf entry into the parent node.
    - We may have to **split the parent as well**. **Splitting the root increases the tree height by one.**
- If **not** violated → insert the point into the **closest cluster**.
- *(Memory management)*: choose an initial threshold and start inserting points. If mid-way the CF-tree size exceeds available memory, **increase the threshold** and **convert the partially built tree into a new tree** — the rebuild is performed **from the leaf nodes of the old tree**, with **no need to re-read all the objects**. Repeat until the entire dataset is scanned.

**Phase 2**: **a bridge between phases 1 and 3** — builds a **smaller CF-tree by increasing the threshold**.
> **Why it's useful** (lecture): if the generated tree is very large, we can modify the thresholds accordingly. **These thresholds can be adjusted without scanning the entire dataset again**, because the algorithm works with **summarized values** rather than the original objects.

**Phase 3**: apply a **global (arbitrary) clustering algorithm to the sub-clusters** given by the leaf entries of the CF-tree → **improves clustering quality**. (In the clusters identified previously, points are grouped only if the diameter constraint is not violated; after building the CF-tree we can further analyze and possibly **merge** clusters.)

**Phase 4**: **scan the entire dataset to label the data points.** Outlier handling.

> **Outlier handling**: BIRCH can handle outliers because, given the **diameter limitation**, an outlier will end up **alone in its own cluster** — we will have an entry in a leaf node with no other points in that cluster.

**Strengths:**
- Finds a good clustering with a **single scan**, and improves quality with a **few additional scans**.
- **Complexity $O(n)$** — usually other algorithms are $O(n^2)$.

**Weaknesses:**
- **Handles only numeric data** (we can compute $LS$ and $SS$ only for numerical data).
- **Sensitive to the insertion order of data points**: when we insert the first point we generate the first cluster and decide whether to split depending on the diameter constraint — **if we change the order of the data objects we may obtain a different CF-tree**. *(Receiving data as a stream is not such an important limitation, because we usually process them in the order observed.)*
- Since we **fix the size of leaf nodes**, the resulting clusters **may not always be very natural**.
- **Clusters tend to be spherical**, because of the use of **radius and diameter** measures.

---

## 5. Density-Based Methods

Clustering based on **density** (a local cluster criterion), such as density-connected points. **They are not based on distances** like the previous two families. The idea is to identify the **dense zones** in the data and generate clusters of **different shapes** following these dense zones.

**Major features:**
- **Discover clusters of arbitrary shape.** *(This modeling capability depends on the parameters we are able to set — in density-based algorithms these parameters are very critical.)*
- **Handle noise and detect outliers** (a side effect of how clusters are created).
- **One scan** of the overall dataset.
- **Need density parameters as a termination condition.**

**Several interesting studies:**
- **DBSCAN**: Ester et al. (KDD'96) — the first one discovered.
- **OPTICS**: Ankerst et al. (SIGMOD'99) — *not really a clustering algorithm*, but used to **order the data objects so we can determine the parameters for DBSCAN**.
- **DENCLUE**: Hinneburg & Keim (KDD'98) — needs some mathematical formulation.
- **CLIQUE**: Agrawal et al. (SIGMOD'98) — more grid-based.

### 5.1 Basic Concepts and Definitions

**Two input parameters:**
- **Eps**: the **maximum radius of the neighbourhood**. Considering an object, we fix a radius determining a sphere representing that point's neighbourhood.
- **MinPts**: the **minimum number of points** that must be contained in the Eps-neighbourhood of a point.

> **⚠️ By changing these parameters, the result of the algorithm may change, because we are effectively using different definitions of density.**
> **With the first parameter we define the VOLUME of the neighbourhood; with the second parameter we express our IDEA OF DENSITY to the algorithm.**

**Core object**: if the **Eps-neighbourhood** ($N_{Eps}$) of an object contains **at least MinPts** objects, then that object is called a **core object**. Points that are not core objects are called **border objects**.
*Example*: eps = 1 cm, MinPts = 3 → m and p are core objects.

**Directly density-reachable**: an object $p$ is **directly density-reachable** from an object $q$ w.r.t. Eps, MinPts if:
1. $p$ belongs to $N_{Eps}(q)$, and
2. $q$ is a **core object**, that is $|N_{Eps}(q)| \ge MinPts$.

*Example*: m is directly density-reachable from p **and vice versa**. q is directly density-reachable from m, **but not vice versa**, because q isn't a core object.

**Density-reachable**: an object $p$ is **density-reachable** from an object $q$ w.r.t. Eps, MinPts if there is a **chain** of objects $p_1,\dots,p_n$, with $p_1 = q$, $p_n = p$, such that $p_{i+1}$ is **directly** density-reachable from $p_i$.

*Example*: q is density-reachable from p because q is directly density-reachable from m and m is directly density-reachable from p. **p is NOT density-reachable from q**, because q is not a core object. *(Note: density-reachability is **not symmetric**.)*

**Density-connected**: an object $p$ is **density-connected** to an object $q$ w.r.t. Eps, MinPts if there is an object $o$ such that **both $p$ and $q$ are density-reachable from $o$** w.r.t. Eps and MinPts.
> **$p$ and $q$ can be either core or border points, while $o$ MUST be a core point.**

*Example*: p, q and m are all density-connected. Similarly, r and s are density-reachable from o, and o is density-reachable from r → o, r, s are all density-connected.

**Quick summary:**
- **Directly density-reachable**: one step from a core point.
- **Density-reachable**: reachable through a **chain** of core points.
- **Density-connected**: two points belong to the **same dense region**.

### 5.2 DBSCAN (Density-Based Spatial Clustering of Applications with Noise)

**A cluster is defined as a MAXIMAL SET OF DENSITY-CONNECTED POINTS.** It discovers clusters of **arbitrary shape** in spatial databases **with noise**.

**Three types of points:**
- **Core points** — have at least MinPts within Eps.
- **Border points** — not core, but in the **neighbourhood of a core point**.
- **Outliers (noise)** — neither core nor border objects.

**How it works:**
- Searches for clusters by **checking the Eps-neighborhood of each object** in the database.
- If the Eps-neighborhood of an object $p$ contains more than MinPts, a **new cluster with a core object is created**.
- DBSCAN **iteratively collects directly density-reachable objects** from these core objects — this may involve **merging a few density-reachable clusters**.
- The process **terminates when no new point can be added to any cluster**.

**Formal cluster definition** — DBSCAN adopts the **closure of density-connectedness** to find connected dense regions as clusters. Each closed set is a density-based cluster. A subset $C$ in $D$ is a cluster if:
1. For any two objects $o_1, o_2$ in $C$, $o_1$ and $o_2$ are **density-connected**, **and**
2. There does **not** exist an object $o$ in $C$ and another object $o'$ in $(D-C)$ such that $o$ and $o'$ are density-connected.
> I.e., **we can't find any other points outside the cluster that are density-connected with the points in the cluster.**

**Input**: $D$ (dataset of $n$ objects), **eps** (radius parameter), **MinPts** (neighborhood density threshold). **Output**: a set of density-based clusters. **Complexity: $O(n^2)$.**

> **⚠️ NB**: if an object isn't a core object we mark it as **noise**, but this does **not** automatically mean it is an outlier — it can also be a **border point**, in which case it **is added to the cluster**.

**Why DBSCAN is powerful (the comparison example):** three datasets — (1) a spherical cluster, (2) a **concave** cluster (a problem for k-means, which identifies only spherical clusters), (3) both concave and convex clusters. Applying **CLARANS** (same result as k-means) vs. **DBSCAN**:
- **k-means/CLARANS** tends to identify **spherical clusters of similar sizes** → it does **not** identify the big circle and the small ones (the natural clusters); instead it groups together points that merely appear close.
- **DBSCAN** models and discovers the **natural clusters by following the density** of the data. In the third case, DBSCAN also **detects outliers**, while k-means assigns them to clusters.
> **DBSCAN works so well in this example because the density of points in the different clusters is very similar.**

### 5.3 DBSCAN's Achilles heel: sensitivity to parameters

**The problem case: clusters with DIFFERENT DENSITIES in different zones.** For classical DBSCAN, we are forced to set the two parameters **for the whole dataset**, without distinguishing between zones of different density.

- **Large Eps** → the neighborhood becomes very large → **many points become core points**, and different regions may become density-connected → DBSCAN tends to **merge clusters together**, and small dense clusters are **absorbed into the larger one**.
- **Small Eps** → the neighborhood becomes very small and the **density requirement increases** → DBSCAN can detect the very dense clusters, but the **sparse cluster may not be recognized** and many points are classified as **noise** — all objects inside the sparse region become outliers.

> **Therefore: DBSCAN works well when clusters have similar densities, but has difficulties when the dataset contains clusters with different densities.** (**OPTICS** gives the possibility to assign different values of Eps for different zones.)

**DBSCAN also has problems with high-dimensional data**: in high-dimensional spaces we encounter the **curse of dimensionality** — points tend to appear very far from each other, and it becomes difficult to identify dense regions. (With high dimensionality the problem persists because **we cannot visualize** the optimal solution.)

**Further illustrations** (the effect of varying Eps): changing Eps from 0.5 to 0.4 raises the density requirement — in the first case the points of the less dense cluster are all considered as belonging to the same cluster; with the stricter requirement they get **split into different clusters**. Similarly, with Eps = 5.0 all points belong to a **single** cluster (very low density requirement); with a higher requirement the points are divided into several clusters.

> **Conclusion: the choice of Eps and MinPts is VERY CRITICAL.**

### 5.4 Heuristic for determining Eps and MinPts

> In clustering we have no ground truth, so it is quite difficult to understand whether what we obtain is a good solution or not.

**The idea: FIX MinPts first, then determine an appropriate Eps.**
> **Why this order**: MinPts is an **integer**, so tuning is simpler — we can change it step by step. **Eps is a real value** and requires much finer tuning.

**The k-dist method:**
- For a given $k$ we define a function **k-dist**, mapping each point to the **distance from its $k$-th nearest neighbour**, where $k$ corresponds to MinPts (e.g. fix MinPts = 4).
- **If we set Eps equal to a point's k-nearest-neighbour distance, by definition that object becomes a core object.**
- **Sort the points in DESCENDING order of their k-dist values**: the graph of this function gives hints about the **density distribution**. If we choose an arbitrary point $p$, set Eps = k-dist($p$) and MinPts = $k$, then **all points with an equal or smaller k-dist value will be core points**.
- **The threshold point is the FIRST POINT IN THE FIRST "VALLEY" of the sorted k-dist graph.** After that line we have a similar distance between objects. All the points **on the right** are core points by definition (equal or smaller k-dist); the points **on the left** can be **outliers or border points**.
- **The idea is to choose Eps so that we don't have too many outliers or border points.**

> **This approach only gives an IDEA of the possible values of Eps — a starting point that will then be tuned.**

### 5.5 OPTICS (Ordering Points To Identify the Clustering Structure)

Ankerst, Breunig, Kriegel, and Sander (SIGMOD'99).

**The problem it addresses**: *the intrinsic cluster structure cannot be characterized by GLOBAL density parameters.*

**First alternative considered — a hierarchical clustering algorithm (e.g. single-link).** Drawbacks:
- **Single-link effect**: clusters connected by a line of few points having small inter-object distance are **not separated**.
- The **dendrograms are hard to understand** for more than a few hundred objects.

**Second alternative — a density-based partitioning algorithm with different parameter settings** (the original DBSCAN doesn't allow working in this direction). Drawbacks:
- There are an **infinite number** of possible parameter values.
- Even with a very large number of different values — requiring a lot of secondary memory to store the different cluster memberships for each point — **it is not obvious how to analyze the results**, and we may still miss the interesting clustering levels.

**The solution**: run an algorithm that produces a **special ORDER of the database** with respect to its density-based clustering structure, containing information about **every clustering level** of the dataset (up to a "**generating distance**" eps), and that is **very easy to analyze**.

**The key observation — density-based clusters are MONOTONIC with respect to the neighborhood threshold:**
> For a constant MinPts value, density-based clusters with respect to a **higher density** (i.e., a **lower** value of eps) are **completely contained** in density-connected sets obtained with respect to a **lower density**.
> *Example*: $C_1$ and $C_2$ are density-based clusters w.r.t. $eps_2 < eps_1$, and $C$ is a density-based cluster w.r.t. $eps_1$ **completely containing** the sets $C_1$ and $C_2$.

**The IDEA**: process a **set of distance parameter values at the same time** — in practice, to use DBSCAN for an **infinite number of distance parameters $eps_i$** which are smaller than a "generating distance eps". The aim is to determine the **optimal Eps in different parts of the space**.

**⚠️ Unlike DBSCAN, OPTICS does NOT assign cluster memberships** (we don't have clusters at the end). Instead, it **stores the order in which the objects are processed**, plus the information that would be used by an extended DBSCAN to assign cluster memberships. **This information consists of only two values per object:**

- **Core-distance** of an object $p$: the **smallest eps value that makes $p$ a core object**. If $p$ is not a core object, the core-distance is **undefined**. *(Since we fix MinPts, we can determine the core-distance of an object.)*
- **Reachability-distance** of an object $q$ from $p$: the **minimum radius value that makes $q$ directly density-reachable from $p$**. $p$ has to be a core object and $q$ must be in the neighborhood of $p$. Therefore:

$$\text{reachability-distance}(p,q) = \max\big(\text{core-distance}(p),\ dist(p,q)\big)$$

> **NB**: if a point is **inside** the core-neighborhood of $p$, the reachability distance **equals the core distance**. If $p$ is not a core object, the reachability-distance is undefined.

**The processing order (algorithm):**
- OPTICS begins with an **arbitrary object** from the input database as the current object $p$ (selected randomly). It retrieves the eps-neighborhood of $p$, determines the **core-distance**, and sets the **reachability-distance to undefined**.
- **If $p$ is not a core object**, OPTICS simply **moves on to the next object** in the OrderSeeds list (or the input database if OrderSeeds is empty).
- **If $p$ IS a core object**, then for each object $q$ in the eps-neighborhood of $p$, OPTICS **updates $q$'s reachability-distance from $p$** and **inserts $q$ into OrderSeeds** if $q$ has not yet been processed.
- The objects contained in **OrderSeeds are sorted by their reachability-distance** to the closest core object from which they have been directly density-reachable. In each step of the WHILE loop, the object with the **smallest reachability-distance** in the seed-list is selected (`OrderSeeds::next()`).
- The eps-neighborhood of this object and its core-distance are determined; then the object is **written to the file OrderedFile** with its core-distance and its current reachability-distance.
- **The iteration continues until the input is fully consumed and OrderSeeds is empty.**
- **At the end, we don't have clusters — we have the LIST OF OBJECTS in the order in which they were processed.** Complexity: worst case **$O(n^2)$**.

**Worked example (from the lecture)**: since A is the first point analyzed, its starting reachability distance is **infinite (undefined)**. B and I are characterized by the **same** reachability distance, so we select **one of them arbitrarily**. Later, C and I again have the same reachability distance → again choose randomly. If we start analyzing I, once all points in its neighborhood are processed we proceed with C. In this way we obtain **2 valleys corresponding to the 2 real clusters** in the dataset.

**Reading the OPTICS reachability plot:**
- **Valleys (low reachability-distance values) = very dense regions = clusters.**
- **Peaks = separations between different clusters.**
- A **horizontal threshold line ($\varepsilon'$)**: all points **below** that line are in a cluster; when the graph rises **above** the line, the cluster ends.

> **Parameter comparison with DBSCAN**: OPTICS also has 2 parameters, **but the situation is very different**. Here **Eps is a "generating distance" — a sort of upper bound — and is NOT so critical.** This algorithm lets us define the **best trade-off between MinPts and Eps**. (Using only DBSCAN it's very hard to find the best combination, because we'd have to define what the best clustering is for a specific domain, without a ground truth.)

**Good for both automatic and interactive cluster analysis**, including finding intrinsic clustering structure; can be represented graphically or with visualization techniques.

### 5.6 DENCLUE (DENsity-based CLUstEring)

Hinneburg & Keim (KDD'98). It has **very solid mathematical foundations** — a clustering method based on a set of **density distribution functions**.

**The problem it solves (in DBSCAN and OPTICS)**: density is calculated by **counting the number of objects** in a neighborhood defined by eps. Such density estimates can be **highly sensitive to the radius value** used. *(DENCLUE also has parameters, but they are less critical.)*

**The IDEA**: clusters are, by definition, high-density regions. This density is modeled with a **continuous function (kernel)** in space (e.g. a Gaussian with a standard deviation determining the width of the function). The overall density is the **sum of the kernel functions placed on the single points**.

> **The mountain analogy** (lecture): suppose we place a Gaussian function at each data point. When points are **close** to each other, the Gaussians **overlap and their contributions sum up, forming a higher peak (a hill)**. When points are **far apart**, the sum produces **valleys** between them. Each observed object acts as an **indicator of high probability density** in its surrounding region, and the estimated density at a given point depends on its distance from the observed objects. **Clusters correspond to MOUNTAINS.**

**Kernel density estimation**: let $x_1,\dots,x_n$ be an i.i.d. sample of a random variable $f$. We associate a kernel function with each point. The kernel density approximation of the probability density function is:

$$\hat{f}_h(x) = \frac{1}{nh}\sum_{i=1}^{n}K\!\left(\frac{x-x_i}{h}\right)$$

where $n$ is the total number of points and $h$ is the standard deviation (bandwidth). The **kernel $K()$** is a non-negative real-valued integrable function satisfying, for all $u$:

$$\int_{-\infty}^{+\infty}K(u)\,du = 1, \qquad K(-u) = K(u)$$

**Example kernels:**
$$K_{Gauss}\!\left(\frac{x-x_i}{h}\right) = \frac{1}{\sqrt{2\pi}}e^{-\frac{(x-x_i)^2}{2h^2}}$$
$$K_{Square}\!\left(\frac{x-x_i}{h}\right) = \begin{cases}0 & \text{if } \left|\frac{x-x_i}{h}\right| > 1\\ 1 & \text{otherwise}\end{cases}$$

> **Comparing the two** (lecture): with the **square kernel** the resulting profile resembles a **"Dolomites profile"** — not very smooth. With the **Gaussian kernel** the profile is **smoother**. In the example, a **flat mountain in the center** corresponds to a **less dense** cluster, while the other three clusters show **higher peaks**.

**Density attractors — the cluster definition:**
- DENCLUE searches for **local maxima of the density function** (the peaks), called **density attractors**.
- **Every point is associated to its attractor.** At the end we obtain a set of objects attracted to the same attractor → they **belong to the same cluster**.
- To **avoid trivial local maxima** (i.e., outlier "clusters"), DENCLUE uses a **noise threshold $\xi$** and only considers those density attractors $x^*$ such that:

$$\hat{f}(x^*) \ge \xi$$

**These attractors are the centers of clusters.**

**Assignment via step-wise hill-climbing** (to find the attractors we "climb the hill" and arrive at the peak). Mathematically, we **follow the gradient** of the function — start from a point, then move in the direction of highest density.

**The algorithm:**
Until there exist samples in the dataset, select a sample $x$. The density attractor for $x$ is computed by hill-climbing:

$$x^0 = x, \qquad x^{j+1} = x^j + \delta\frac{\nabla\hat{f}(x^j)}{\lVert\nabla\hat{f}(x^j)\rVert}$$

where $\delta$ is a parameter controlling the **speed of convergence**, and

$$\nabla\hat{f}(x) = \frac{1}{h^{d+2}n}\sum_{i=1}^{n}K\!\left(\frac{x-x_i}{h}\right)(x_i - x)$$

with $d$ = space dimension and $K$ Gaussian.

The hill-climbing procedure **stops at step $k>0$ if $\hat{f}(x^{k+1}) < \hat{f}(x^k)$**, and assigns $x$ to the density attractor $x^* = x^k$.

> **⚠️ On $\delta$**: if $\delta$ is **too large** we move very quickly along the slope of the mountain — **the risk is moving too fast and passing to the other side without identifying the peak.**

**Efficiency heuristic**: this algorithm is **very expensive computationally**, because we should apply hill-climbing for **each** point. So: **the algorithm stores all points $x'$ with $d(x^j,x')\le h/2$** for any step $0<j<k$ during the hill-climbing procedure, and **attaches these points to the cluster of $x^*$ as well**. Using this heuristic, **all points located close to the path** from $x$ to its density attractor **can be classified without applying hill-climbing to them**.
> **The intuition**: points located close to the ascent path are likely to converge to the **same** density attractor, so they can be assigned to the same cluster directly — significantly improving efficiency.

**Outliers**: an object $x$ is an **outlier or noise** if it converges in the hill-climbing procedure to a local maximum $x^*$ with

$$\hat{f}(x^*) < \xi$$

> I.e., during the procedure a point may reach a local maximum whose density is **too low to be representative of a cluster**. In the mountain analogy: **high and wide hills represent real clusters, while small isolated ones correspond to outliers.**

**Arbitrary-shaped clusters — merging attractors:**
> **Density attractors can be MERGED if they are connected through regions of sufficiently high density.** This allows the algorithm to generate clusters with **arbitrary shapes**. In other words: **two attractors belong to the same cluster if, along the path connecting them, the value of the density function is always greater than $\xi$.**

Formally, an arbitrary-shape cluster (w.r.t. two constants $h$ and $\xi$) for the set of density attractors $X$ is a subset $C \subseteq D$ where:
$$\forall x\in C\ \exists x^*\in X:\ \hat{f}(x^*)\ge\xi$$
$$\forall x_1^*,x_2^*\in X:\ \exists\text{ a path }P\in F^d\text{ from }x_1^*\text{ to }x_2^*\text{ with }\forall p\in P:\ \hat{f}(p)\ge\xi$$

**Effect of the parameters $h$ and $\xi$:**
- **$h$ (the standard deviation)**: when we **increase** $h$, the Gaussians start to **overlap** and their sum becomes **smoother**, forming a **single smooth "mountain"**. Consequently, **high values of $h$ do not allow us to clearly identify different clusters.** **$h$ is critical: it can lead to a different number of detected clusters.**
- **$\xi$**: determines the **threshold used to identify outliers**, but it **also affects whether different density attractors can be connected**. E.g., with $\xi=2$, only some attractors are connected → a single cluster forms; with $\xi=1$, **more attractors become connected**, which may lead to **merging different clusters**.

**How to choose $h$ and $\xi$:**
- **Suggested choice for $h$**: consider different values of $h$ and determine the **largest interval between $h_{max}$ and $h_{min}$ where the number of density attractors $m(h)$ remains CONSTANT**.
- **Suggested choice for $\xi$**: if the database is noise-free, all density attractors of $D$ are significant and $\xi$ should be chosen in
$$0 \le \xi \le \min_{x^*\in X}\{f^D_C(x^*)\}$$

**Major features:**
- **Solid mathematical foundation.**
- **Good for datasets with large amounts of noise** — we can rule out this noise by changing $\xi$.
- Allows a **compact mathematical description** of arbitrarily shaped clusters in high-dimensional datasets.
- **Significantly faster** than existing algorithms — e.g. **faster than DBSCAN by a factor of up to 45** (thanks to assigning all points near the ascent path to the same attractor).
- **Complexity (with optimization): $O(N\log N)$.**
- **But needs an accurate choice of the parameters $h$ and $\xi$**: $h$ determines the influence of a point in its neighbourhood; $\xi$ describes whether a density attractor is significant, allowing a reduction in the number of density attractors.

**Implementation — two steps:**

**Step 1: Initial pre-clustering based on a GRID** (to speed up the calculation of the density function):
- The **minimal bounding (hyper-)rectangle** of the dataset is divided into **$d$-dimensional hypercubes with edge length $2h$**.
- **Only hypercubes which actually contain data points are determined** — we restrict our analysis to cubes we're sure contain something.
- The hypercubes are **numbered depending on their relative position** from a given origin. The keys of the populated cubes can be efficiently stored in a **randomized search-tree or a B⁺-tree**.
- For each populated cube $c$, in addition to the key, we store: the **number of points $N_c$** belonging to $c$, **pointers** to those points, and the **linear sum** $\sum_{x\in c}x$.
- This information is used in the clustering step for **fast calculation of the mean of a cube**, $mean(c)$. Since clusters can spread over more than one cube, **neighboring populated cubes have to be accessed** — to speed this up we **connect** neighboring populated cubes. Formally, two cubes $c_1,c_2\in C_p$ are **connected** if:
$$d(mean(c_1), mean(c_2)) < 4\sigma$$
*(we use a Gaussian kernel, therefore $h=\sigma$).*

**Step 2: the actual clustering step** (the one that really speeds up the process):
- Only the **highly populated cubes $C_{sp}$** and cubes **connected to** a highly populated cube are considered in determining clusters:
$$C_{sp} = \{c\in C_p \mid N_c \ge \xi_c\}$$
$$C_r = C_{sp}\cup\{c\in C_p \mid \exists c_s\in C_{sp} \text{ and } \exists connection(c_s,c)\}$$
- For $x\in c$ and $c, c_1 \in C_r$, we set:
$$near(x) = \{x_1\in c_1 \mid d(mean(c_1),x)\le k\sigma \text{ and } \exists connection(c_1,c)\}$$
- The limit $k\sigma$ is chosen such that **only marginal influences are neglected**; **$k=4$ is sufficient for practical purposes**.
- The resulting **local** density function is:
$$\hat{f}^D_{Gauss}(x) = \sum_{x_1\in near(x)}e^{-\frac{d(x,x_1)^2}{2\sigma^2}}$$
- The density attractor is then computed with the same hill-climbing iteration on $\hat{f}^D_{Gauss}$, stopping at $k$ when $\hat{f}^D(x^{k+1}) < \hat{f}^D(x^k)$, taking $x^*=x^k$ as the new density attractor.

> At the end of the process we still only find the attractors, **but in a much more efficient way**. The hypercube idea is only there to make the process faster — **the intuition behind the algorithm doesn't change.**
>
> **⚠️ NB**: if we choose too small a value for the standard deviation, we risk identifying a **very high number of outliers**.
>
> **Scalability comparison**: the computational effort increases very much with increasing data size **for DBSCAN**. For **DENCLUE it remains quite stable**, because the process is genuinely optimized.

### 5.7 ⚠️ Density-based methods and high dimensionality

**ALL density-based clustering methods are NOT fully effective when clustering high-dimensional data** (they suffer from the **curse of dimensionality**), because methods relying on **near or nearest-neighbor information** do not work well in high-dimensional spaces. *(DBSCAN suffers from the choice of parameters more than DENCLUE.)*

- In high-dimensional datasets, it is **very unlikely that data points are nearer to each other than the average distance** between data points, because of the **sparsely filled space**.
- As a result, **as the dimensionality increases, the difference between the distance to the nearest and to the farthest neighbors of an object goes to ZERO.**

---

## 6. Grid-Based Methods

Introduced to **speed up** the clustering process. **The goal is not to obtain extremely precise clusters, but rather a faster solution.**

The algorithm uses a **multi-resolution grid data structure**. At the **bottom level** (where we have the points) a grid is used to represent the data space; then a **hierarchy of levels** is built, creating a **coarse representation** of the data. The information at the bottom level is used at higher levels to **summarize** the situation at a coarser resolution.

**General characteristics of grid-based methods:**
- The data space is divided into **rectangular cells (grid cells)**.
- **The clustering algorithm does not operate directly on the data points, but on the GRID CELLS.**
- **The computational cost depends more on the NUMBER OF CELLS than on the number of data points**, which often makes these methods **more scalable**.

**Two methods analyzed**: **STING** (Wang, Yang & Muntz, 1997) and **CLIQUE** (Agrawal et al., SIGMOD'98) — both grid-based *and* subspace clustering.

### 6.1 STING (a STatistical INformation Grid approach)

- The spatial area is divided into **rectangular cells**, with **several levels of cells corresponding to different levels of resolution**.
- **Each cell at a high level is partitioned into a number of smaller cells in the next lower level** (higher-level information summarizes lower-level information).
> **Why represent higher-level regions with fewer, larger cells**: if a cell has **low density it can be immediately discarded**; if it is **dense** it is further explored using smaller cells at lower levels.

**Statistical information** of each cell is **calculated and stored beforehand** and is used to answer queries. **Parameters of higher-level cells can easily be calculated from parameters of lower-level cells.** Parameters include:
- **count, mean, standard deviation, min, max**
- **type of distribution** — normal, uniform, exponential, or **none** (if unknown) — obtained by the user or by **hypothesis tests**. *(The more information we have, the more precise the solution.)*

The statistical parameters for the cells in the **lowest layer** are computed **directly from the values present in the table** when data are loaded into the database; the parameters for all other levels are computed **from their respective children cells** in the lower level.

**Query processing — a TOP-DOWN approach** (possible because we first collect info at each level). An **SQL-like language** is used to describe queries. Two common query types: (1) find a region specifying certain constraints; (2) take in a region and return some attribute of the region.

**The procedure:**
1. **Start from a pre-selected layer** — typically with a small number of cells. *(It does not have to be the topmost layer.)*
2. For **each cell in the current layer**, compute the **confidence interval** (or estimated range of probability) reflecting the **cell's relevance to the given query**. It is calculated using the **statistical parameters** of each cell.
3. From the computed interval, **label the cells as relevant or irrelevant** for this query.
4. **Remove the irrelevant cells** from further consideration.
5. When finished with the current layer, **proceed to the next lower level**, examining **only the remaining relevant cells**. Repeat until the **bottom layer** is reached.
6. At this point, **if the query specifications are met, the regions of relevant cells that satisfy the query are returned**. Otherwise, the data falling into the relevant cells are **retrieved and further processed** until they meet the query requirements.

> **What this really is** (lecture): it is not a proper clustering algorithm, but **it is as if we dynamically generate the clusters (sets of relevant hypercubes) we need to answer the query**. Focusing on the coarse-granularity level at layer 1 isolates the cells that are relevant at the bottom level — this is how we speed up the procedure.

**Advantages:**
- **Query-independent, easy to parallelize, incremental update.**
- **Generation of the clusters: complexity $O(n)$.**
- **Query processing time: $O(g)$**, where $g$ is the number of grid cells at the **lowest** level.

**Disadvantages:**
- **All the cluster boundaries are either horizontal or vertical — no diagonal boundary is detected** (we impose the shape of the grid onto the cluster).

> **Relation to DBSCAN**: STING is **faster** than DBSCAN, but the regions returned by STING are an **approximation** of the result of DBSCAN. **As the granularity approaches zero, the regions returned by STING approach the result of DBSCAN.**

### 6.2 CLIQUE (Clustering In QUEst)

Agrawal, Gehrke, Gunopulos, Raghavan (SIGMOD'98).

**Automatically identifies SUBSPACES of a high-dimensional data space that allow better clustering than the original space.** It is also used in **frequent pattern analysis**.

> **CLIQUE can be considered BOTH density-based AND grid-based**, because we exploit a grid but we also exploit the idea of **density inside the hypercubes** found by the grid.

**How it works:**
- It partitions **each dimension into the same number of equal-length intervals**.
- It partitions an **$m$-dimensional data space into non-overlapping rectangular units**.
- **A unit is DENSE if the fraction of total data points contained in the unit exceeds the input model parameter** (we still need parameters).
- **A cluster is a MAXIMAL SET OF CONNECTED DENSE UNITS within a subspace.**

**Major steps:**
1. **Partition the data space** and find the number of points that lie inside each cell of the partition (creating a grid, as in STING).
2. **Identify the subspaces that contain clusters using the APRIORI PRINCIPLE**: *if a $k$-dimensional unit is dense, then so are its projections in $(k-1)$-dimensional space.* Therefore the candidate dense units in the $k$-th dimensional space are generated **from the dense units found in $(k-1)$-dimensional space**.
3. **Identify clusters**: determine both **dense units** in all subspaces of interest and **connected dense units** in all subspaces of interest.
4. **Generate a minimal description** for the clusters: determine **maximal regions** that cover a cluster of connected dense units.
5. **Determination of the minimal cover** for each cluster.

**Understanding the Apriori property in practice:**
> If a unit is dense in $k$ dimensions, then **all its projections in $(k-1)$ dimensions must also be dense**. This is a **necessary condition** (density propagation across dimensions).
>
> **Reasoning in the opposite direction**: suppose a unit is dense only if it contains at least 5 points. If a region contains only 3 points — even if they are spread along the entire $y$-axis — the unit **cannot** be considered dense.
>
> **The Apriori property reduces the search space**: starting from the $(k-1)$-dimensional space, we generate candidate $k$-dimensional units by considering **only those $(k-1)$-dimensional units that are dense**. So instead of exploring all possible $k$-dimensional units, we focus only on those originating from dense $(k-1)$-dimensional units.
>
> **⚠️ However**, even with this pruning, **we still need to COUNT how many points fall into each candidate unit** to verify whether it is actually dense — the fact that we start from dense $(k-1)$-units does **not** automatically imply that the $k$-dimensional region is dense.

**The algorithm — two steps:**

**Step 1**: partition the $d$-dimensional data space into **non-overlapping rectangular units**, identifying the **dense units** among these.
- CLIQUE partitions **every dimension into intervals** and identifies intervals containing **at least $l$ points**, where $l$ is the **density threshold**.
- Then it **iteratively joins** two $k$-dimensional dense cells $c_1$ and $c_2$, in subspaces $(D_{i1},\dots,D_{ik})$ and $(D_{j1},\dots,D_{jk})$ respectively, **if** $D_{i1}=D_{j1},\dots,D_{ik-1}=D_{jk-1}$ **and** $c_1,c_2$ **share the same intervals in those dimensions**. The join generates a new $(k+1)$-dimensional **candidate cell** $c$ in space $(D_{i1},\dots,D_{ik},D_{jk})$.
- CLIQUE **checks whether the number of points in $c$ passes the density threshold**. The iteration **terminates when no candidates can be generated or no candidate cells are dense**.

**Step 2**: use the dense cells in each subspace to **assemble clusters**, which can be of **arbitrary shape**.
- **Minimum Description Length (MDL) principle**: use the **maximal regions** to cover connected dense cells, where a **maximal region** is a **hyperrectangle where every cell falling into it is dense**, and **the region cannot be extended further in any dimension** in the subspace.
- **Finding the best description of a cluster in general is NP-Hard.** Thus CLIQUE adopts a **simple greedy approach**: start with an arbitrary dense cell, find a maximal region covering it, then work on the remaining dense cells not yet covered. **The greedy method terminates when all dense cells are covered.**

**Worked example (salary, vacation, age — 3-dimensional):**
- After plotting the objects, **each dimension is split into intervals of equal length**, forming a 3-D grid where each unit is a 3-D rectangle. **The goal is to find the dense 3-D rectangular units.**
- Instead of searching directly in the 3-D space, CLIQUE **exploits the Apriori property** and first finds dense units in **lower-dimensional subspaces**: we identify dense regions in the **salary–age plane** and in the **vacation–age plane**.
- A dense region in the **salary–age** plane can be **extended inwards** (along the vacation direction); a dense region in the **vacation–age** plane can be **extended upwards** (along the salary direction).
- The **intersection** of these extended regions gives the **candidate search space** in which 3-dimensional dense units exist. We then also find the dense units in the **salary–vacation** plane and intersect its extension with the candidate space, obtaining all 3-D dense units.
- **⚠️ However, the fact that projections are dense does not GUARANTEE that the corresponding 3-D unit is dense — so we must count the number of objects again in the 3-D cell to verify it.**

> **TO SUM UP: we used the dense units in subspaces in order to find the dense units in the 3-dimensional space. After finding the dense units, it is very easy to find clusters.**

**Strengths:**
- **Automatically finds subspaces of the highest dimensionality** such that high-density clusters exist in those subspaces.
- **Insensitive to the order of records** in input and does **not presume some canonical data distribution**.
- **Scales linearly with the size of input** and has **good scalability as the number of dimensions increases**.

**Weaknesses:**
- Obtaining a meaningful clustering **depends on proper tuning of the grid size and the density threshold** (we still have parameters).
- **The accuracy of the clustering result may be degraded at the expense of the simplicity of the method** — using a coarse grid speeds up the process but yields coarse clusters.

> **The aim is always to reduce the computational effort.**

---

## 7. Clustering Evaluation

There are **3 steps**:

1. **Assessing clustering tendency**: clustering analysis on a dataset is meaningful **only when there is a nonrandom structure** in the data → see the **Hopkins statistic** (§2).
2. **Determining the number of clusters** in a dataset → see the **Elbow method** and cross-validation (§3.3).
3. **Measuring clustering quality**: measures to assess **how well the clusters fit the dataset**, and measures that **score clustering** so we can **compare two sets of clustering results** on the same dataset.

> **The core difficulty** (lecture): normally we have different clustering algorithms applicable to different domains. **It isn't easy to define the best one, because we don't have the ground truth.** In classification we have labeled objects, so we can directly identify whether the algorithm correctly identifies classes.

**Two families of metrics: EXTRINSIC and INTRINSIC.**

| | **Extrinsic** | **Intrinsic** |
|---|---|---|
| Supervision | **Supervised** — the ground truth **IS** available | **Unsupervised** — the ground truth is **unavailable** |
| What it does | Compare a clustering **against the ground truth** using a clustering quality measure | Evaluate the goodness of a clustering **by using the definition of a cluster**: how well the clusters are **separated**, and how **compact** they are |
| Domain relevance | Evaluates how **meaningful** the clusters are within a specific **application domain** | Does **not** refer to the application domain |
| Example metric | **BCubed precision and recall** | **Silhouette coefficient** |

> **When each is used** (lecture): **extrinsic evaluation is typically used in RESEARCH when introducing a new clustering algorithm** — the idea is to start from a **labeled dataset** (typically datasets used for classification), apply the clustering algorithm, and compare the resulting clusters with the true class labels. This demonstrates the algorithm is effective because it identifies clusters corresponding to the true classes. *(Most used in research, not for real applications.)*
> **Intrinsic evaluation** instead assesses whether an algorithm performs well in terms of **cluster separation and compactness**, without relying on any information from the application domain.

### 7.1 Extrinsic Methods: the 4 essential criteria

A clustering quality measure **$Q(C, C_g)$** — for a clustering $C$ given the ground truth $C_g$ — is **good** if it satisfies these **4 essential criteria**:

1. **Cluster homogeneity**: **the purer, the better.** When we discover clusters we want to know whether they contain objects belonging to the same class (pure = same label).
2. **Cluster completeness**: should assign objects belonging to the **same category in the ground truth** to the **same cluster**. If we have 3 classes in our dataset, we would like to find 3 clusters merging completely with the three classes.
3. **Rag bag**: putting a **heterogeneous object into a PURE cluster** should be **penalized more** than putting it into a **rag bag** (i.e., a "miscellaneous"/"other" category). *(Some clustering algorithms introduce a "noise" category for exactly this purpose.)*
4. **Small cluster preservation**: **splitting a SMALL category into pieces is MORE HARMFUL than splitting a large category** into pieces — small clusters should be preserved.

### 7.2 BCubed Precision and Recall

Implemented using the concepts of **precision** and **recall** encountered in classification. It evaluates precision and recall **for every object** in a clustering, according to the ground truth. **The goal is to match clusters with classes, and vice versa.**

- **The PRECISION of an object** indicates **how many other objects in the SAME CLUSTER belong to the SAME CATEGORY** as that object. *(Ideally precision = 1.)*
- **The RECALL of an object** reflects **how many objects of the SAME CATEGORY are assigned to the SAME CLUSTER** as that object.
- **If both precision and recall are 1, the goal is achieved: the clusters perfectly correspond to the classes.**

**Formally**: let $D=\{o_1,\dots,o_n\}$ be a set of objects and $C$ a clustering on $D$. Let $L(o_i)$ be the **category** of $o_i$ given by the ground truth, and $C(o_i)$ the **cluster_ID** of $o_i$ in $C$. For two objects $o_i, o_j$ ($i\ne j$), the **correctness** of the relation between them in clustering $C$ is:

$$Correctness(o_i,o_j) = \begin{cases}1 & \text{if } \big(L(o_i)=L(o_j)\big) \Longleftrightarrow \big(C(o_i)=C(o_j)\big)\\[4pt] 0 & \text{otherwise}\end{cases}$$

> I.e., correctness = 0 when the two objects belong to the same cluster but **not** to the same class, **or vice versa**.

$$Precision_{BCubed} = \frac{1}{n}\sum_{i=1}^{n}\frac{\displaystyle\sum_{\substack{o_j:\,j\ne i\\ C(o_i)=C(o_j)}}Correctness(o_i,o_j)}{\big|\{o_j \mid j\ne i,\ C(o_i)=C(o_j)\}\big|}$$

$$Recall_{BCubed} = \frac{1}{n}\sum_{i=1}^{n}\frac{\displaystyle\sum_{\substack{o_j:\,j\ne i\\ L(o_i)=L(o_j)}}Correctness(o_i,o_j)}{\big|\{o_j \mid j\ne i,\ L(o_i)=L(o_j)\}\big|}$$

> **Reading the two**: for **precision** we consider objects belonging to the **same CLUSTER** ($C$) and evaluate the correctness of all such combinations; for **recall** we express the sum referring to objects belonging to the **same CLASS** ($L$).
>
> **These metrics aim to evaluate how well the clustering result matches the KNOWLEDGE OF THE DOMAIN** — how meaningful the clusters are within a specific application domain. **We are NOT evaluating cluster quality in terms of compactness and separation**, as is done with intrinsic methods.

### 7.3 Intrinsic Methods: the Silhouette Coefficient

The aim is to evaluate **how much the clusters satisfy the DEFINITION of a cluster** (compact and separated groups of objects). **But we cannot say whether clusters are meaningful from a domain point of view.**

> **Granularity**: the silhouette is computed **object by object**. To evaluate an entire cluster we compute the **average** of the silhouette coefficients of its objects.

**For each object $o$ in $D$:**
- Compute **$a(o)$** = the **average distance between $o$ and all the other objects in the cluster to which $o$ belongs**. *(We expect $a$ to be very small, because we want compact clusters.)*
$$a(o) = \frac{\sum_{o'\in C_i,\ o'\ne o}dist(o,o')}{|C_i| - 1}$$
> We divide by $|C_i| - 1$ because $C_i$ also contains $o$ itself, and we can't compute the distance between $o$ and itself.

- Compute **$b(o)$** = the **MINIMUM average distance from $o$ to all clusters to which $o$ does NOT belong**. *(We expect $b$ to be very high, because we want separated clusters.)*
$$b(o) = \min_{C_j,\ 1\le j\le k,\ j\ne i}\left\{\frac{\sum_{o'\in C_j}dist(o,o')}{|C_j|}\right\}$$
> I.e., compute the average distance between $o$ and all points of cluster 1, the average distance between $o$ and all points of cluster 2, …, then take the **minimum**.

**The silhouette coefficient:**

$$s(o) = \frac{b(o) - a(o)}{\max\{a(o),\ b(o)\}}$$

**Interpretation** — the value is between **−1 and 1**:
- The **smaller $a(o)$, the more compact** the cluster.
- **When $s(o)$ approaches 1**: the cluster containing $o$ is **compact** and $o$ is **far away from other clusters** — **the preferable case**. *(If $b$ is very large it becomes the max, and the ratio → 1, since $a$ becomes negligible.)*
- **When $s(o)$ is negative** (i.e., $b(o) < a(o)$): in expectation, **$o$ is CLOSER to the objects of another cluster than to objects of its own cluster**. **This is a bad situation and should be avoided.** *(Symmetrically, if $a$ dominates, the result → −1.)*

**Aggregating**:
- To measure a **cluster's fitness** within a clustering → compute the **average silhouette of all objects in the cluster**.
- To measure the **quality of a whole clustering** → use the **average silhouette of all objects in the dataset**.

> **⚠️ CRITICAL LIMITATION: the silhouette coefficient is usable only for CONVEX clusters.**
> This is because **two points belonging to a CONCAVE cluster can have a very large average distance** (for example, if they are separated by the concave region), **but this does not mean the cluster is not compact**.
> → **We can apply the silhouette to the output of k-means, but we should be careful when applying it to the output of DBSCAN**, because DBSCAN can also model concave clusters.

---

## 8. Summary (from the final slides)

- **Cluster analysis** groups objects based on their **similarity** and has wide applications.
- Measures of similarity can be computed for **various types of data**.
- Clustering algorithms can be categorized into **partitioning methods, hierarchical methods, density-based methods, grid-based methods, and model-based methods**.
- **K-means and K-medoids** are popular **partitioning-based** clustering algorithms.
- **BIRCH and CHAMELEON** are interesting **hierarchical** clustering algorithms; there are also **probabilistic** hierarchical clustering algorithms.
- **DBSCAN, OPTICS, and DENCLUE** are interesting **density-based** algorithms.
- **STING and CLIQUE** are **grid-based** methods, where **CLIQUE is also a subspace clustering algorithm**.
- **The quality of clustering results can be evaluated in various ways.**

---

## 9. Key points / potential exam pitfalls

### Foundations
- **The defining difference from Chapter 4: there is NO GROUND TRUTH.** Everything hard about clustering — evaluation, choosing $k$, choosing parameters, deciding whether the result is meaningful — flows from this single fact.
- **A clustering algorithm ALWAYS returns clusters, even on uniform random data.** This is why **assessing clustering tendency comes FIRST**, before running anything. The Hopkins statistic answers "are there clusters at all?", not "how many?".
- **Hopkins direction of reasoning**: $x_i$ = distances from **real** sampled points to their nearest neighbours; $y_i$ = distances from **synthetic uniform** points to their nearest neighbours in $D$. **$H\approx0.5$ → uniform (no tendency); $H\to1$ → highly separated clusters; decision threshold $H>0.75$ at 90% confidence.** The mechanism: with real clusters, synthetic uniform points land in empty space → $\sum y_i$ grows → $H$ rises.
- **Bi-clustering ≠ subspace clustering.** Bi-clustering clusters **instances AND features simultaneously** (coherent blocks in the data matrix); subspace clustering clusters instances in different feature subspaces — **features themselves are not clustered**.

### Partitioning
- **$k$ must be given in advance** — this is k-means' defining constraint, and the reason the Elbow method exists.
- **The Elbow method works because the cost function ALWAYS decreases with $k$** (it reaches exactly 0 when $k=n$). So you can't minimize it — you look for the **turning point** where marginal improvement collapses. **No elbow → likely uniform data.** The elbow method is **specific to partitioning/convex-cluster problems**.
- **The distance determines the cluster SHAPE**: Euclidean weights all dimensions equally → **spherical** clusters; weighted distances → **ellipsoidal**. Either way, **only convex clusters** are reachable. This is *the* structural limitation that motivates density-based methods.
- **k-means is sensitive to initialization** (can land in a local minimum) → **run it multiple times with different initializations and keep the partition with the lowest cost**.
- **k-means vs k-medoids on outliers**: the centroid is a **mean**, so a single outlier **drags it**; the medoid is an **actual data object**, so it is far less influenced. This is the whole reason k-medoids exists.
- **PAM / CLARA / CLARANS differ ONLY in how the search space is explored**: PAM examines **all** neighbours (accurate, expensive, $O(k(n-k)^2)$ per iteration); CLARA runs **PAM on samples** (fast, but quality depends on sample representativeness); CLARANS **randomly samples neighbours dynamically at each step** (better exploration, still costly). In the CLARANS graph, **nodes are sets of $k$ medoids and two nodes are neighbours iff they share $k-1$ medoids.**
- **k-modes for categorical data** replaces means with **modes**; **k-prototype** for mixed categorical + numerical.

### Hierarchical
- **The central problem shifts** from "distance between an object and a centroid" (partitioning) to **"distance between two CLUSTERS"** → hence **linkage metrics**.
- **Single link causes the CHAINING EFFECT** (long stringy clusters); **complete link is inappropriate for elongated/chain-like clusters** but good for compact clumps; **average link works well in BOTH cases** — this is the standard comparison question.
- **Complete-link merging is a min-of-max**: compute the maximum pairwise distance for every cluster pair, then merge the pair with the **smallest** such maximum.
- **No $k$ needed** — but a **termination condition IS needed**. To get $k$ groups from a finished dendrogram, **cut the $(k-1)$ longest links**.
- **⚠️ Hierarchical clustering still yields CONVEX clusters** — switching from partitioning to hierarchical does **not** buy you arbitrary shapes. Only density-based methods do.
- **Merges/splits are irreversible** — the defining weakness — and complexity is at least **$O(n^2)$**.
- **DIANA's stopping rule for one split**: keep moving objects into the splinter group **while $D_h > 0$**; stop when **all $D_h$ are negative**. Then choose the **next cluster to split by largest DIAMETER**.

### BIRCH
- **CF = (N, LS, SS)** — memorize this triple. Its power is **additivity**: a parent's CF is the **sum** of its children's CFs, and inserting a new point means just $N{+}1$, $LS{+}x$, $SS{+}x^2$. **This is precisely what makes BIRCH incremental / streaming-capable.**
- **Three CF-tree parameters**: **$B$** (branching factor, max children per non-leaf), **$T$** (max **diameter** of subclusters at leaves), **$L$** (max entries in a leaf). **A leaf node represents a cluster.**
- **Complexity $O(n)$ with a single scan** — versus $O(n^2)$ typical elsewhere. Phase 2 can rebuild a smaller tree **by raising the threshold WITHOUT rescanning the dataset**, because only summaries are stored.
- **BIRCH's weaknesses are structural**: **numeric data only** (LS/SS require numbers), **sensitive to insertion order** (different order → different CF-tree), and **clusters tend to be spherical** because radius/diameter are the operative measures.
- **Outlier handling comes free**: an outlier ends up **alone in its own leaf entry** because of the diameter constraint.

### Density-based
- **The three reachability notions are NOT interchangeable and directly-density-reachable is NOT symmetric.** Memorize: *directly* = one step from a **core** point; *density-reachable* = a **chain** of core points; *density-connected* = both reachable from a **common core point $o$** (and **$o$ must be core**, while $p,q$ may be border). A cluster = a **maximal set of density-connected points**.
- **"Marked as noise" ≠ "is an outlier"** in DBSCAN — a non-core point may still be a **border point** and get absorbed into a cluster.
- **DBSCAN's fatal case is VARYING DENSITY**: large Eps merges distinct clusters; small Eps recovers dense clusters but declares the sparse region entirely noise. **DBSCAN works well only when clusters have similar densities.** OPTICS is the answer to exactly this.
- **The Eps heuristic order matters: FIX MinPts FIRST (integer, easy to tune), then derive Eps** (real-valued, needs fine tuning) from the **sorted k-dist plot** — take the first point of the **first "valley"**.
- **OPTICS does NOT produce clusters.** It produces an **ordering** plus two values per object (**core-distance**, **reachability-distance**). You read clusters off the reachability plot: **valleys = clusters, peaks = separations**, with a horizontal threshold line. **Reachability-distance$(p,q)=\max(\text{core-dist}(p), dist(p,q))$.**
- **The monotonicity observation** that justifies OPTICS: for fixed MinPts, clusters at **higher density (lower eps)** are **completely contained** in clusters at lower density. In OPTICS, **Eps is only a generating distance / upper bound — it is NOT critical**, unlike in DBSCAN.
- **DENCLUE replaces counting with a continuous kernel density function.** Clusters = **density attractors** (local maxima) with $\hat f(x^*)\ge\xi$; points are assigned by **hill-climbing along the gradient**. **Arbitrary shapes arise by MERGING attractors connected by a path where density stays $\ge\xi$ throughout.**
- **DENCLUE's two parameters do different jobs**: **$h$** (bandwidth/σ) controls **smoothness** — too large merges everything into one smooth mountain, too small explodes the outlier count; **$\xi$** sets the **noise threshold AND controls whether attractors merge**. Choose $h$ as the **largest interval where the number of attractors $m(h)$ stays constant**.
- **DENCLUE's speed-up heuristic**: points within $h/2$ of the hill-climbing **path** are assigned to the same attractor **without running hill-climbing on them** — this is what gives up to a **45× speedup over DBSCAN**, with $O(N\log N)$ complexity.
- **⚠️ ALL density-based methods degrade in high dimensions** — as dimensionality grows, **the difference between nearest- and farthest-neighbour distances goes to zero**, so "dense region" loses meaning.

### Grid-based
- **The defining trait: the algorithm operates on CELLS, not on data points** → **cost depends on the number of cells, not the number of objects** → scalability. The trade-off is **precision for speed**.
- **STING's boundaries are always horizontal/vertical — never diagonal**, because the grid shape is imposed on the clusters. **STING approaches DBSCAN's result as granularity → 0.** Complexities: **$O(n)$** to build, **$O(g)$** per query ($g$ = cells at the lowest level).
- **CLIQUE is BOTH grid-based and density-based** — and it is the **subspace** clustering algorithm of the chapter.
- **CLIQUE's Apriori property is a NECESSARY, NOT SUFFICIENT condition**: if a $k$-dim unit is dense then all its $(k-1)$-dim projections are dense — so dense projections only give you **candidates**. **You must still count the points in the $k$-dimensional cell to confirm density.** This is the most common misunderstanding here.
- **CLIQUE step 2 uses MDL + a greedy approach** because finding the best cluster description is **NP-Hard**.

### Evaluation
- **Three evaluation steps in order: (1) tendency → (2) number of clusters → (3) quality.**
- **Extrinsic = supervised (ground truth available), used mainly in RESEARCH** to validate a new algorithm on labeled classification datasets. **Intrinsic = unsupervised**, judging only compactness and separation — it **cannot tell you whether clusters are meaningful in the domain**.
- **The 4 extrinsic criteria**: homogeneity, completeness, **rag bag** (polluting a *pure* cluster is worse than dumping into a miscellaneous bin), **small cluster preservation** (splitting a *small* category hurts more than splitting a large one).
- **BCubed precision vs recall — which set you sum over**: **precision** ranges over objects in the **same CLUSTER**; **recall** ranges over objects in the **same CLASS**. The `Correctness` function is an **if-and-only-if**: it is 1 only when "same class" and "same cluster" agree.
- **Silhouette: $a$ = own-cluster average distance (want SMALL = compact), $b$ = MINIMUM average distance to another cluster (want LARGE = separated).** $s(o)\to 1$ is ideal; **$s(o)<0$ means the object is closer to another cluster than its own — a genuinely bad assignment.** Divide by $|C_i|-1$ in $a(o)$ because $o$ itself is in $C_i$.
- **⚠️ Silhouette assumes CONVEX clusters** — two points in a **concave** cluster can be far apart on average without the cluster being bad. So it is fine for **k-means output** but **must be used cautiously on DBSCAN output**, which produces concave clusters.

### Cross-chapter connections
- **Curse of dimensionality** (Ch. 2 §5.7, Ch. 3 §4.1) strikes again here — it is the reason **density-based methods fail in high dimensions** and part of why **subspace clustering (CLIQUE)** exists.
- **Distance/dissimilarity measures** are entirely inherited from **Chapter 2 §5** (Minkowski/Euclidean, nominal, ordinal, mixed types) — clustering just consumes them.
- **Normalization** (Ch. 3 §5.1) matters here for exactly the same reason as in k-NN: attributes on larger scales would dominate the distance and therefore dominate the clustering.
- **Precision/recall** are reused from **Chapter 4 §9.3**, but redefined **per object** (BCubed) rather than per class.

---

*File auto-generated by merging `5-Clustering.pdf` (professor's slides, 98 pages) and `5 - Clustering Sbobine.pdf` (lecture notes, 64 pages). For questions about this chapter, refer only to this file.*
