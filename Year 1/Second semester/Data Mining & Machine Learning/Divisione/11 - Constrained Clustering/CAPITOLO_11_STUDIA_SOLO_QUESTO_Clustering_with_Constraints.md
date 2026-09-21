# Chapter 11 — Clustering with Constraints

> **Course:** Data Mining and Machine Learning / Bioinspired Computational Methods — Biological Data Mining
> **Instructor:** Prof. Francesco Marcelloni (University of Pisa)
> **Sources merged into this file:** professor's slides (`11-ConstrainedClusterAnalysis.pdf`, 13 pages — Chapter 11 of Han–Kamber–Pei) + lecture notes (`11 - Clustering with constrain sbobine.pdf`, 10 pages)
>
> **Instructions for Claude:** this is the single reference file for Chapter 11. When the user asks study questions about this chapter, use **only** this file as context (no need to re-read the original PDFs unless a detail is missing). The file merges the official slide content with the lecture's discursive explanations and worked examples, keeping both levels of detail.

---

## Chapter outline
1. Why constraint-based cluster analysis?
2. Categorization of constraints
   - Constraints on instances (must-link / cannot-link)
   - Constraints on clusters (δ-constraint, ε-constraint)
   - Constraints on similarity measurements
   - Hard vs. soft constraints
3. Converting cluster-level constraints into instance-level constraints
4. Enforcing constraints: strict vs. partial
5. Conflicting or redundant constraints; informativeness and coherence
6. The effect of constraints on the solution space
7. Handling hard constraints: the COP-k-means algorithm
8. Handling soft constraints: the CVQE algorithm
9. Speeding up constrained clustering: obstacles, visibility graphs, microclusters

---

## 1. Why constraint-based cluster analysis?

Two reasons given on the very first slide:

- **We need USER FEEDBACK: users know their applications the best.**
- **Fewer parameters but more user-desired constraints.**

**The motivating example (slides): a bank manager wishes to locate four ATMs in an area containing an obstacle** (a river). **Ignoring the obstacle produces the clusters on the right — which are wrong for the application.**

> **The lecture's fuller version of the same example**: when applying a clustering algorithm such as **k-means**, it is **not possible to explicitly incorporate domain-specific constraints**, such as those imposed by a **river**. The algorithm operates **solely on the given data points** (e.g. buildings), **without accounting for geographical barriers**.
>
> As a result the clustering may identify, say, **four clusters based purely on spatial proximity**. But **placing an ATM at the centroid of each cluster is a suboptimal solution**, because **people located on opposite sides of the river experience significant inconvenience in reaching the service**.
>
> **⚠️ The moral**: **standard clustering algorithms may be perfectly CORRECT from a computational perspective and still be UNSUITED to the application domain.** This is what motivates methods able to **incorporate and model such constraints — using constraints to aid and bias the clustering.**

**The goal remains the usual one**, now subject to domain knowledge: **partition unlabeled data into clusters such that examples in the same cluster are similar, examples in separate clusters are different, AND the constraints are maximally respected.**

## 2. Categorization of constraints

### 2.1 Constraints on INSTANCES

> **They specify how a pair (or a set) of instances should be grouped in the cluster analysis.** Two kinds — **must-link vs. cannot-link**:
> - **`must-link(x, y)`**: **$x$ and $y$ should be grouped into one cluster**;
> - **`cannot-link(x, y)`**: **$x$ and $y$ should NOT be in the same cluster.**

**Constraints can be defined using VARIABLES**, e.g.
$$\texttt{cannot-link}(x,y) \quad \text{if } \text{distance}(x,y) > d$$

> **What changes in practice** (lecture): with a standard algorithm such as k-means, **the resulting clusters are determined solely by the spatial distribution of the data points** — the approach **does not account for any prior knowledge or structural relationship among the data**. When **must-link and cannot-link constraints are incorporated**, the clustering is **guided by additional information**, allowing a **more accurate representation of the underlying data structure**: in particular, **must-link constraints force certain pairs of points into the same cluster**, reshaping the clusters according to **domain-specific requirements**.

### 2.2 Constraints on CLUSTERS

> **They specify a requirement on the clusters themselves** (not on single instances): e.g. the **minimum number of objects in a cluster**, the **maximum diameter of a cluster**, the **shape of a cluster** (e.g. convex), the **number of clusters** (e.g. $k$ in k-means).

> **Note on "shape"** (lecture): **convexity derives from the choice of the distance we use** — it is not an independent knob.

Two formal instances:

**δ-constraint (MINIMUM SEPARATION)** — for any two clusters $S_i$, $S_j$ with $i \ne j$, and for any two instances $s_p \in S_i$, $s_q \in S_j$:
$$D(s_p, s_q) \;\ge\; \delta$$

**ε-constraint** — for any cluster $S_i$ with $|S_i| > 1$, for **every** instance $s_p \in S_i$ there **exists** another instance $s_q \in S_i$, $s_q \ne s_p$, such that:
$$D(s_p, s_q) \;\le\; \varepsilon$$

> **⚠️ The two do different jobs**: the **δ-constraint is about SEPARATION BETWEEN clusters**; the **ε-constraint can be seen as a constraint on the COMPACTNESS of a cluster** — **no point may sit isolated inside its own cluster**.

### 2.3 Constraints on SIMILARITY MEASUREMENTS

> **They specify a requirement that the similarity CALCULATION must respect.**
>
> **Slide example**: to cluster people as **moving objects in a plaza**, while the **Euclidean distance** is used to give the walking distance between two points, **a constraint on the similarity measurement is that the trajectory implementing the shortest distance CANNOT CROSS A WALL**.
>
> **Lecture version**: if there is a **building at the centre of a square**, we **cannot use the Euclidean distance** between two people in that square, because it represents **the shortest path without taking the obstacle into account**.

### 2.4 Hard vs. soft constraints

- **A constraint is HARD if a clustering that violates it is UNACCEPTABLE.**
- **A constraint is SOFT if a clustering that violates it is not preferable but is ACCEPTABLE when no better solution can be found. Soft constraints are also called PREFERENCES.**

> **The lecture's added rationale**: a **hard** violation means the result **is not a physical solution** at all. **Soft** constraints are called preferences **because they belong to an optimization problem** — **we are only expected to satisfy them, and in any case we obtain a physically valid solution.**

## 3. Converting cluster-level constraints into instance-level constraints

**Cluster-level constraints can be converted into instance-level ones:**

- **δ-constraint (minimum separation)** → **for every point $x$, MUST-LINK all points $y$ such that $D(x,y) < \delta$**
  ⇒ **a CONJUNCTION of must-link (ML) constraints.**
- **ε-constraint** → **for every point $x$, must-link to AT LEAST ONE point $y$ such that $D(x,y) \le \varepsilon$**
  ⇒ **a DISJUNCTION of ML constraints.**

> **⚠️ This will generate MANY instance-level constraints** — the conversion is conceptually clean but potentially explosive. **The conjunction/disjunction distinction (ALL $y$ vs. AT LEAST ONE $y$) is the examinable detail.**

## 4. Enforcing constraints

- **Strict enforcement**: **find the best FEASIBLE clustering respecting ALL constraints.**
- **Partial enforcement**: **find the best clustering MAXIMALLY respecting the constraints.**

## 5. Conflicting or redundant constraints

**The standard counter-example:**
- $\texttt{must-link}(x,y)$ if $\text{dist}(x,y) < 5$
- $\texttt{cannot-link}(x,y)$ if $\text{dist}(x,y) > 3$

> **If a data set has two objects $x$, $y$ with $\text{dist}(x,y) = 4$, then NO clustering can satisfy both constraints simultaneously.**

**How can we measure the quality and usefulness of a set of constraints?**

- **INFORMATIVENESS**: **the amount of information carried by the constraints that is BEYOND the clustering model.** Given a data set $D$, a clustering method $A$ and a set of constraints $C$, the informativeness of $C$ w.r.t. $A$ on $D$ is measured by **the FRACTION OF CONSTRAINTS IN $C$ THAT ARE UNSATISFIED by the clustering computed by $A$ on $D$.**
- **COHERENCE** of a set of constraints: **the degree of AGREEMENT AMONG THE CONSTRAINTS THEMSELVES**, which can be measured by **the redundancy among the constraints**.

> **How to read informativeness**: if the unconstrained algorithm **already satisfies** all your constraints, they **add nothing** — informativeness 0. The constraints are informative **exactly to the extent that they contradict what the algorithm would have done anyway**.

## 6. The effect of constraints on the solution space

**Constraints divide the set of all plausible solutions into two sets — feasible and infeasible:**
$$S = S_F \cup S_I$$

- **Constraints effectively REDUCE the search space to $S_F$.**
- **All the elements of $S_F$ share a common property.**
- **So it is NOT unexpected that we find solutions with a desired property, and find them QUICKLY.**

> **⚠️ The counter-intuitive but explicitly stated point**: **hard constraints can make the problem EASIER, not harder** — they prune the space of candidate solutions. *(The lecture adds: this holds when we have **hard** constraints.)*

## 7. Constraint-based clustering methods (I): handling HARD constraints

**Handling hard constraints = strictly respecting the constraints in cluster assignments** — a **feasibility problem**. The reference algorithm is **COP-k-means**, with two ingredients:

### (1) Generate SUPER-INSTANCES for must-link constraints
- **Compute the TRANSITIVE CLOSURE of the must-link constraints.**
- **To represent such a subset, REPLACE all those objects by their MEAN.**
- **The super-instance also carries a WEIGHT, which is the number of objects it represents.**

> **Why this works** (lecture): when a **super-instance is assigned to a cluster, all the original objects it represents are necessarily assigned to the same cluster** — the must-links are satisfied **by construction**. It is a **preprocessing step** that **enforces that all linked elements stay together**.
>
> **Why the weight matters**: when applying classical k-means, **during the computation of the centroid a super-instance acts as a STRONGER ATTRACTOR** because of its higher weight compared to individual points. **As a result the centroid is shifted towards the super-instance**, reflecting the contribution of all the objects it stands for.

**Transitive closure, concretely:**
> **Transitive closure = augmenting the initial set of constraints with all the implicit constraints that logically follow from them.** Suppose we have `must-link(A,B)` and `must-link(B,C)`. Applying transitive closure yields `must-link(A,C)`. The set becomes `must-link(A,B)`, `must-link(B,C)`, `must-link(A,C)` — so **$A$, $B$ and $C$ belong to the same CONNECTED COMPONENT and must be assigned to the same cluster.**

### (2) Conduct MODIFIED k-means clustering to respect cannot-link constraints
- **Modify the centre-assignment process of k-means into a NEAREST FEASIBLE CENTRE assignment.**
- **An object is assigned to the nearest centre SUCH THAT the assignment respects all cannot-link constraints.**

### Where the constraints come from

> At the beginning, **must-link and cannot-link constraints are typically NOT available**. We usually start from an **unlabeled dataset**, although **a subset of instances may carry labels**. Those labeled instances can be exploited to **derive pairwise constraints**:
> - **instances belonging to the SAME class induce MUST-LINK constraints**;
> - **instances belonging to DIFFERENT classes induce CANNOT-LINK constraints.**
>
> **The goal is to generate all the possible must-link and cannot-link constraints from the available information**, and this is where **transitive closure plays a key role**.

### The worked figure sequence (COP-k-means, Weight–Height example)

1. **A set of unlabeled data points** in the Weight–Height feature space, with **must-link constraints (green)** and **cannot-link constraints (orange)**. Applying **transitive closure to the must-links**, all connected points are grouped into a single set and **replaced by a super-instance**, typically the **mean** of those points. *(Slide: "ML points averaged".)*
2. **Some points are labeled** (marked "X"), which **allows further propagation of the constraints**. **Each super-instance behaves as a single point but with a weight equal to the number of instances it represents**, making it **more influential in shifting the centroids in k-means**. *(Slide: "3 nearest feasible assignment".)*
3. **The combined effect of constraints and super-instances is fully visible**: the final result is a **clean separation into clusters that respect both the structure of the data and the imposed constraints**.

> **In summary: super-instances are a fundamental PREPROCESSING step, because they reduce the complexity of the problem.**

## 8. Constraint-based clustering methods (II): handling SOFT constraints

**Treated as an OPTIMIZATION problem: when a clustering violates a soft constraint, a PENALTY is imposed on the clustering.**

**Overall objective: optimize the clustering quality AND minimize the constraint-violation penalty.**

**Ex. the CVQE algorithm (Constrained Vector Quantization Error): conduct k-means clustering while enforcing constraint-violation penalties.**

**Objective function: the sum of distances used in k-means, adjusted by the constraint-violation penalties.**

- **Penalty of a MUST-LINK violation**: if objects $x$ and $y$ must-be-linked **but are assigned to two different centres $c_1$ and $c_2$**, then **$\text{dist}(c_1, c_2)$ is added to the objective function** as the penalty.
- **Penalty of a CANNOT-LINK violation**: if objects $x$ and $y$ cannot-be-linked **but are assigned to a common centre $c$**, then **$\text{dist}(c, c')$ is added**, where **$c'$ is the CLOSEST CLUSTER TO $c$ THAT CAN ACCOMMODATE $x$ OR $y$**.

$$\textbf{Objective} \;=\; \text{k-means error} \;+\; \text{penalty}_{\text{must-link}} \;+\; \text{penalty}_{\text{cannot-link}}$$

> **How to read the penalties**: both are **distances between CENTROIDS**, not between the points involved. They measure **how far the solution had to be displaced** in order to violate the constraint — a must-link violation costs more when the two hosting centres are far apart; a cannot-link violation costs the distance to the **nearest cluster that could have hosted one of the two objects**.
>
> **The key property of the soft setting**: **if we cannot satisfy the soft constraints, the solution REMAINS a physically valid solution.**

## 9. Speeding up constrained clustering

**It is costly to compute some constrained clusterings.** The reference case is **clustering with OBSTACLE objects** *(Tung, Hou & Han, "Spatial clustering in the presence of obstacles", ICDE'01)*:

- **Cluster people as moving objects in a plaza.**
- **Euclidean distance is used to measure the walking distance.**
- **But the constraint on similarity measurement is that the trajectory implementing the shortest distance CANNOT CROSS A WALL.**
- **The distance has to be derived by GEOMETRIC COMPUTATIONS: the computational cost is high when a large number of objects and obstacles are involved.**

> **The strategic move** (lecture): **shift the problem from explicitly handling constraints at the INSTANCE level to incorporating them directly into the DISTANCE COMPUTATION.** Instead of enforcing constraints separately, **we redefine the distance metric so that it inherently respects them**, which simplifies the clustering process.
>
> **⚠️ Note that imposing constraints is expensive on two fronts: it requires modifying the clustering ALGORITHM and redefining how DISTANCES are computed.**

### 9.1 Visibility

> **A point $p$ is VISIBLE from another point $q$ if the straight line joining $p$ and $q$ does not intersect any obstacle.**

**A VISIBILITY GRAPH is the graph $VG = (V,E)$ such that:**
- **each vertex of the obstacles has a corresponding node in $V$**;
- **two nodes $v_1, v_2 \in V$ are joined by an edge in $E$ if and only if the corresponding vertices are VISIBLE TO EACH OTHER.**

**Let $VG' = (V', E')$ be the visibility graph created from $VG$ by adding the two additional points $p$ and $q$ to $V'$**, where $E'$ contains an edge joining two points of $V'$ **iff the two points are mutually visible**. Then:

> **The shortest path between two points $p$ and $q$ will be a SUBPATH of $VG'$.**

### 9.2 Microclusters and join indices

To reduce the cost of distance computation between any two pairs of objects, **several pre-processing and optimization techniques** can be used.

**(a) MICROCLUSTERS** — **group points that are close together into microclusters**, so that we **compute the distance between two microclusters rather than between individual points**. This is done by:
1. **first TRIANGULATING the region $R$ into triangles**, and
2. **then grouping nearby points in the same triangle into microclusters, using a method similar to BIRCH or DBSCAN.**

**By processing microclusters rather than individual points, the overall computation is reduced.**

**(b) Precomputed JOIN INDICES**, built on the computation of the shortest paths:
- **(1) VV indices — for any pair of OBSTACLE VERTICES;**
- **(2) MV indices — for any pair of MICROCLUSTER and OBSTACLE VERTEX.**

**Use of the indices helps further optimize the overall performance.**

> **The result**: using such precomputation and optimization strategies, **the distance between any two points (at the granularity level of a microcluster) can be computed EFFICIENTLY**. **Thus the clustering process can be performed in a manner similar to a typical efficient k-medoids algorithm such as CLARANS, and achieve good clustering quality for large data sets.**

**The closing illustration**: the same plaza clustered **without taking obstacles into account** vs. **taking obstacles into account** — the two results are visibly different, and only the second is usable.

### 9.3 Final framing

> **Using clustering with constraints we are adopting a SEMI-SUPERVISED approach** (lecture): in a normal clustering algorithm **all the data are unlabeled and we have no guide of any kind**. When we have **some labeled points**, we can exploit them to **guide the algorithm** to generate clusters that respect the imposed constraints. **We adopt an unsupervised approach with some labeled points — whose number is much smaller than that of the unlabeled ones.**

---

## Key points / potential exam pitfalls

### Motivation and categorization
- **⚠️ The core message of the chapter**: a clustering can be **computationally correct and still wrong for the domain** (the river/ATM example). Constraints exist to inject **domain knowledge**, and the pitch is **"fewer parameters, more user-desired constraints"**.
- **Three categories of constraints — on INSTANCES, on CLUSTERS, on SIMILARITY MEASUREMENTS.** Be able to give an example of each; the similarity one is the **wall/plaza** case.
- **Constraints on instances can be defined via VARIABLES**, e.g. `cannot-link(x,y) if dist(x,y) > d` — they need not be enumerated by hand.
- **⚠️ δ-constraint = separation BETWEEN clusters ($D(s_p,s_q) \ge \delta$ for points in different clusters); ε-constraint = COMPACTNESS WITHIN a cluster (every point has some other point of its own cluster within $\varepsilon$).** These two are the easiest pair to swap under pressure.
- **"Shape" constraints (e.g. convexity) DERIVE from the choice of distance** — not an independent parameter.

### Conversion, enforcement, quality
- **δ-constraint ⇒ CONJUNCTION of must-links (with ALL $y$ closer than $\delta$); ε-constraint ⇒ DISJUNCTION of must-links (with AT LEAST ONE $y$ within $\varepsilon$).** And **the conversion may generate a huge number of instance-level constraints.**
- **Strict enforcement = best FEASIBLE clustering respecting ALL constraints; partial enforcement = best clustering MAXIMALLY respecting them.** These map onto **hard** and **soft** constraints respectively.
- **Hard = violation unacceptable (no physical solution); soft = violation undesirable but acceptable ⇒ "preferences" ⇒ penalties, and the solution stays valid.**
- **The conflicting-constraint example is fully numeric**: `must-link if dist<5` + `cannot-link if dist>3` are **unsatisfiable for any pair at distance 4**.
- **⚠️ Informativeness = the fraction of constraints NOT satisfied by the clustering the algorithm produces on its own** (i.e. how much new information they carry **beyond the model**). **Coherence = agreement among the constraints themselves, measurable through their redundancy.** Do not define informativeness as "how many constraints there are".
- **⚠️ Constraints split the solutions into $S = S_F \cup S_I$ and reduce the search to $S_F$ — so hard constraints can make the search FASTER, not slower.**

### COP-k-means (hard constraints)
- **Two ingredients, in order: (1) transitive closure of must-links → WEIGHTED super-instances (replace by the mean); (2) nearest FEASIBLE centre assignment for cannot-links.**
- **⚠️ The WEIGHT of a super-instance is what makes it a stronger attractor in centroid computation** — forgetting the weight is the classic incomplete answer.
- **Must-links are satisfied BY CONSTRUCTION (through the super-instances); cannot-links are satisfied by MODIFYING THE ASSIGNMENT STEP.** The two constraint types are handled by two entirely different mechanisms.
- **Transitive closure: `ML(A,B)` + `ML(B,C)` ⇒ `ML(A,C)`** ⇒ one **connected component** ⇒ one cluster.
- **Constraints are typically DERIVED from a few labeled instances**: same class ⇒ must-link, different class ⇒ cannot-link. **This is what makes constrained clustering SEMI-SUPERVISED.**

### CVQE (soft constraints)
- **Objective = k-means error + must-link penalty + cannot-link penalty.**
- **⚠️ Both penalties are distances between CENTROIDS, never between the two constrained points.** Must-link violation ⇒ $\text{dist}(c_1,c_2)$ where the two objects landed; cannot-link violation ⇒ $\text{dist}(c,c')$ with **$c'$ the closest cluster that CAN ACCOMMODATE $x$ or $y$** (that qualifier is part of the definition).

### Obstacles and speed-ups
- **A point $p$ is VISIBLE from $q$ if the segment $pq$ does not intersect any obstacle.** The **visibility graph $VG$ has the OBSTACLE VERTICES as nodes** and **mutual visibility as edges**; adding $p$ and $q$ gives $VG'$, and **the shortest path between $p$ and $q$ is a SUBPATH of $VG'$.**
- **Microclusters are built by TRIANGULATING the region first, then grouping nearby points within the same triangle** using a **BIRCH- or DBSCAN-like** method.
- **⚠️ Two index types: VV = obstacle vertex × obstacle vertex; MV = microcluster × obstacle vertex.** (There is no "MM" index in the scheme — distances are always routed through obstacle vertices.)
- **The final clustering runs like an efficient k-medoids, e.g. CLARANS**, at the **granularity of microclusters**.

### Cross-chapter connections
- **k-means and k-medoids/CLARANS, BIRCH and DBSCAN** all come from **Chapter 5 (Clustering)**: COP-k-means and CVQE are **modifications of k-means**, and the obstacle speed-ups reuse **BIRCH/DBSCAN-style microclustering** plus **CLARANS**.
- **The visibility graph** connects this chapter to **Chapter 10 (Graph Clustering)** — a graph is built specifically to make a **shortest-path (geodesic) distance** computable in the presence of obstacles, which is exactly the geodesic-distance notion of Chapter 10 §5.
- **Semi-supervision** links to **Chapter 4 (Classification)** — labels inducing must-link/cannot-link pairs — and to the **semi-supervised outlier detection** of **Chapter 7 §6**, where a few labels similarly guide an unsupervised procedure.
- **Constraint-based mining in Chapter 8 §2** is the same philosophy applied to frequent patterns: **user-supplied constraints that both focus the result and PRUNE the search space**, with the same headline property that **constraints reduce the space of candidate solutions** ($S_F$ here, pattern-space pruning there).
- **The choice of distance determining cluster shape** points back to **Chapter 2's** proximity measures.

---

*File auto-generated by merging `11-ConstrainedClusterAnalysis.pdf` (professor's slides, 13 pages) and `11 - Clustering with constrain sbobine.pdf` (lecture notes, 10 pages). Formulas garbled by PDF text extraction (the δ- and ε-constraints, the CVQE penalty definitions) have been reconstructed in their correct form. Note: an earlier draft of these lecture notes was appended to the Chapter 9 notes; this file is the authoritative version for the topic. For questions about this chapter, refer only to this file.*
