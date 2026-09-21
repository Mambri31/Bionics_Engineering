# Chapter 15 — Data Stream Analysis

> **Course:** Data Mining and Machine Learning / Bioinspired Computational Methods — Biological Data Mining
> **Instructor:** Prof. Francesco Marcelloni (University of Pisa)
> **Sources merged into this file:** professor's slides (`15-DataStream.pdf`, 23 pages — labelled "Chapter 13. Data Stream Analysis" in the deck's own numbering) + lecture notes (`15 - DataStream sbobine.pdf`, 21 pages — lessons **L16–L17, 28/04 – 05/05/2026**)
>
> **Instructions for Claude:** this is the single reference file for Chapter 15. When the user asks study questions about this chapter, use **only** this file as context (no need to re-read the original PDFs unless a detail is missing). The file merges the official slide content with the lecture's discursive explanations and worked examples, keeping both levels of detail.

---

## Chapter outline (the official slide structure)
1. **Introduction** — what a data stream is, the problems, the computational model, concept drift, data structures, window models
2. **Clustering** — Adaptive Streaming k-Means, MuDi-Stream, comparison
3. **Classification** — VFDT (Hoeffding trees) and CVFDT

---

# PART 1 — INTRODUCTION

## 1. What is a data stream?

> **A DATA STREAM is a potentially UNBOUNDED, ORDERED SEQUENCE of instances.** A data stream $S$ may be written as
> $$S = \{x_1, x_2, x_3, \dots, x_N\}$$
> **where $x_i$ is the $i$-th data instance, a $d$-DIMENSIONAL FEATURE VECTOR, and $N \to \infty$.**

> **⚠️ The definition contains TWO important aspects** (lecture):
> 1. **UNBOUNDED** — data can continue to arrive **indefinitely**, so **we cannot assume all instances can be stored permanently.** Memory is limited while the incoming data may be extremely large or even infinite. **And even if we imagined having enough storage, applying ML algorithms to millions or billions of stored instances would be computationally very expensive.**
> 2. **ORDERED SEQUENCE** — instances arrive **one after another, following a temporal order.** **This order MATTERS**, because data are received progressively and must be processed as they arrive.

**Where streams come from**: **sensors** are the archetype — modern systems rely on many sensors continuously collecting measurements over time — but many other systems generate streams too.

### 1.1 ⚠️ The distinction that frames the whole chapter

> **A critical issue is that the STATISTICAL PROPERTIES of the data may CHANGE OVER TIME.** A model learned at time $t$ **may not perform well at time $t+N$**, because the distribution may have changed. **So it is not enough to build a model once and assume it will remain valid forever.**

> **⚠️ APPLYING a model to a stream vs. LEARNING a model FROM a stream — do not confuse these.**
> - In a **classical** classification problem a **training set is available**, a model is learned from it, and it is **then applied** to new data. Those new data **may well arrive as a stream, but the LEARNING PHASE IS ALREADY COMPLETE** — the task is only to apply a previously trained model.
> - **Here the problem is different: WE WANT TO LEARN THE MODEL FROM THE STREAM ITSELF.** The **model EVOLVES while data continues to arrive** — e.g. we continuously receive labelled instances and use them to **update the classifier over time.**
>
> **In traditional data mining the model is built OFFLINE on a fixed dataset. In data stream mining the model must be learned and updated ONLINE, while the data is arriving.**

**A simplifying assumption used throughout**: in a classification problem it is assumed that **the NUMBER OF CLASSES remains CONSTANT over time** — the stream may evolve and its distribution may change, but the set of possible classes is fixed. *(More advanced approaches, associated with **online learning**, allow the number of classes to vary, at the cost of additional complexity.)*

## 2. Problems in mining data streams

| **Traditional data mining REQUIRES** | **Why it fails on streams** |
|---|---|
| **the ENTIRE dataset to be present** | **impractical (and impossible) to store the whole dataset** |
| **MULTIPLE SCANS of the overall dataset** | **impractical (and impossible) to perform multiple scans** |
| **RANDOM ACCESS to instances** | **random access is EXPENSIVE** |
| **computationally HEAVY learning phases** | **only SIMPLE CALCULATION per datum is possible, due to time and space constraints** |

> **The lecture's elaboration of each point:**
> - **Entire dataset available**: in classification the training set is accessible before learning begins; in clustering the full dataset is generally available before the algorithm runs.
> - **Multiple scans**: **c-means clustering** is the example — the optimization repeatedly updates prototypes and memberships, and **each iteration requires accessing the dataset again.** Feasible when data are stored and finite; **not realistic when data arrive continuously.**
> - **Random access**: *"the difference can be compared to accessing data in **RAM versus reading data from a sequential TAPE**. In RAM it is possible to directly access a specific memory location, whereas in a stream the data must be processed IN THE ORDER IN WHICH THEY ARRIVE."*
> - **Heavy training**: algorithms such as **SVM** require considerable computational effort — problematic because **incoming data often arrive at high speed, so each instance must be processed very rapidly.**

### 2.1 Motivation: where streams appear

> **A growing number of applications generate streams of data:**
> - **performance measurements in network monitoring and traffic management**;
> - **log records generated by web servers**;
> - **tweets on Twitter**;
> - **transactions in retail chains, ATM operations in banks**;
> - **sensor network data.**
>
> **Application characteristics: MASSIVE VOLUMES of data, and records arriving at a RAPID RATE.**

> **The worked motivating case from the lecture — the vaccine-perception study.** Some years before the COVID-19 pandemic, a study analysed **the perception of Italian citizens regarding vaccines through Twitter data**. The motivation was **vaccination coverage**: campaigns are effective at population level only if a sufficiently large proportion of individuals is vaccinated, so **understanding public perception became important.**
> **Twitter data is a clear example of streaming data** — tweets are generated continuously and very rapidly. **⚠️ And during the analysis it became evident that a model built using tweets collected up to a certain time COULD BECOME OUTDATED AFTER ONLY A FEW WEEKS**, because **the way people discussed vaccines changed over time, especially in relation to specific events.** The model **had to be updated periodically.**
>
> **This illustrates the central idea: A MODEL LEARNED AT A GIVEN TIME MAY NOT REMAIN RELIABLE INDEFINITELY.**

## 3. The computational model

```
Data Streams  e_1, …, e_n  ──►  Stream Processing Engine  ──►  (Approximate) Answer
                                          ▲
                                          │
                                  Synopsis in Memory
```

> **The logic** (lecture): stream mining algorithms **do NOT work directly on the whole stream.** They **process the incoming data in order to build a SYNOPSIS in memory** — **a compact summary of the stream observed so far.** The idea is to **extract and preserve the most relevant information while avoiding storing all the original data.** **The actual ML algorithm is then applied to the SYNOPSIS rather than to the complete dataset.**

**The THREE stream processing requirements:**
> 1. **SINGLE PASS** — **each record is examined AT MOST ONCE.** When a new instance arrives it is used to update the synopsis, **not stored expecting to be revisited.**
> 2. **BOUNDED STORAGE** — **limited memory $M$ for storing the synopsis**, not the whole dataset.
> 3. **REAL-TIME** — **per-record processing time (to maintain the synopsis) must be LOW**, because the system must keep pace with the arrival rate.

> **⚠️ An important nuance** (lecture): **it IS possible to use more complex ML algorithms — but typically NOT directly on the raw stream in real time.** They may be **applied OFFLINE to the synopsis.** **The critical real-time component is the CONSTRUCTION AND MAINTENANCE OF THE SUMMARY**: the stream must be processed at the rate at which data arrives, while **more expensive analysis can be performed later on the compact representation.** This online/offline split is the architecture of *every* algorithm in this chapter.

### 3.1 Approximate answers and their bounds

> **Generally, algorithms compute APPROXIMATE answers** — it is **difficult to compute answers accurately with limited memory.** *(Not only because the data are summarized, but because exact answers are often impossible when memory is bounded.)*

**But approximate does not mean uncontrolled — two kinds of guarantee:**

| Type | Guarantee |
|---|---|
| **DETERMINISTIC bounds** | **the algorithm computes an approximate answer, but with BOUNDS ON THE ERROR** — the maximum possible error is known, so the result is **interpretable and controlled, not arbitrary** |
| **PROBABILISTIC bounds** | **with probability at least $1-\delta$, the computed answer is within a factor $\varepsilon$ of the actual answer** — $\delta$ is the **probability of failure**, $\varepsilon$ the **tolerated approximation error** |

> **⚠️ A useful corollary on the slide**: **single-pass algorithms for processing streams are ALSO APPLICABLE TO (MASSIVE) TERABYTE DATABASES.** Even when data are not technically a stream, **their size may make multiple scans or full in-memory storage impractical** — so stream strategies help there too.

> **The trade-off in one line** (lecture): *"exact answers are often SACRIFICED in favour of EFFICIENCY, SCALABILITY and REAL-TIME APPLICABILITY."*

## 4. Concept drift

> **CONCEPT DRIFT: an UNFORESEEN CHANGE in the statistical properties of data stream instances OVER TIME.**

> **A concrete illustration** (lecture): in a classification problem the initial distribution may contain **90% of instances of one class and 10% of another**; as time passes **these proportions may change.** Consequently **a model learned at a certain moment may no longer be reliable later**, because the statistical characteristics used to train it are **no longer representative of the current stream.**

**THE FOUR TYPES — know all four and be able to distinguish them:**

| Type | Definition |
|---|---|
| **SUDDEN** | **Between two consecutive instances the change occurs AT ONCE**, and after this time **only instances of the new class are received.** *A sharp transition; the previous distribution is rapidly replaced.* |
| **GRADUAL** | **The number of instances belonging to the PREVIOUS class DECREASES gradually while the number belonging to the NEW class INCREASES over time.** **⚠️ During a gradual drift, instances of BOTH previous and new classes are visible.** |
| **INCREMENTAL** | **Data instances belonging to the previous class EVOLVE to a new class STEP BY STEP.** After the drift is completed **the previous class disappears.** **⚠️ The instances arriving during the drift are of TRANSITIONAL FORMS and DO NOT HAVE TO BELONG TO EITHER of the classes.** |
| **RECURRING** | **The data instances change between two or more statistical characteristics SEVERAL TIMES.** **⚠️ NEITHER of the classes disappears PERMANENTLY — both arrive IN TURNS.** |

> **⚠️ Gradual vs. incremental is the classic confusion.** In **gradual** drift you see a **mixture of old and new instances**, each still recognisably belonging to one concept or the other. In **incremental** drift you see **intermediate, transitional instances** that belong to **neither**. *Gradual = mixing; incremental = morphing.*
>
> **And recurring drift is the hardest**, because **the model must not only adapt to change, but may need to RECOGNISE THAT A PREVIOUSLY OBSERVED CONCEPT HAS RETURNED.**

## 5. Data structures for the synopsis

> **It is not possible to store and manage the whole input data. Only a SYNOPSIS of the input stream is stored: special data structures enable us to INCREMENTALLY SUMMARIZE the input stream.**

**Four commonly used data structures:**

| # | Structure | What it stores |
|---|---|---|
| 1 | **Feature vectors** | **a summary of the data instances** — descriptive information capturing relevant characteristics, rather than every single object |
| 2 | **Prototype arrays** | **only a number of REPRESENTATIVE instances that exemplify the data** — the stream is approximated through a smaller set of meaningful examples |
| 3 | **Coreset trees** | **the summary in a TREE structure** — allows the synopsis to be managed **hierarchically**, useful when the stream is large and must be updated efficiently |
| 4 | **Grids** | **the data DENSITY in the feature space** — the space is partitioned into cells and the algorithm tracks how many points fall into each region |

## 6. The window model

> **It is more efficient to process RECENT data instead of the whole data.**
>
> **A WINDOW is a portion, or sub-stream, of the complete data stream** — it **defines the limited interval of instances the algorithm operates on at a given time.**

### 6.1 Damped window model

> **RECENT DATA HAVE MORE WEIGHT THAN OLDER DATA: the importance of the instances DECREASES BY TIME.**
> **Usually implemented using DECAY FUNCTIONS which scale down the weight of the instances depending on the time passed since the instance was received:**
> $$f(t) = 2^{-\lambda t}$$
> **where $\lambda$ is the DECAY RATE: a HIGHER decay rate means a MORE RAPID DECREASE in the value.**

> **Why it matters** (lecture): this model is **particularly useful in the presence of CONCEPT DRIFT** — if the distribution changes over time, **the most recent instances are generally more representative of the current situation.** A **lower $\lambda$** keeps older instances relevant for longer.

### 6.2 Landmark window model

> **The whole data between two LANDMARKS is included in the processing and ALL INSTANCES HAVE EQUAL WEIGHT.**
> **⚠️ CONSECUTIVE WINDOWS DO NOT INTERSECT** — the new window **begins exactly from the point the previous window ends.**
> Let $w$ be the window length; the instances of the $m$-th window are the block of consecutive instances between the corresponding landmarks.

### 6.3 Sliding window model

> **The window SWAPS ONE INSTANCE AT EACH STEP: the older instance moves OUT and the most recent instance moves IN, FIFO style.**
> **All instances in the window have EQUAL WEIGHT**, but **⚠️ CONSECUTIVE WINDOWS MOSTLY OVERLAP** — since the window shifts by one instance, most elements are shared between one window and the next.
> The window always contains **the most recent $w$ instances.**

> **⚠️ The three models compared — the two axes are WEIGHTING and OVERLAP:**
>
> | Model | Weights | Consecutive windows |
> |---|---|---|
> | **Damped** | **UNEQUAL** (decay with age) | *(conceptually continuous)* |
> | **Landmark** | **EQUAL** | **DO NOT overlap** |
> | **Sliding** | **EQUAL** | **LARGELY overlap** |
>
> **Landmark and sliding both weight equally — they differ in OVERLAP. Damped is the only one with unequal weights.**

> **The role of windowing** (lecture): it is a **PREPROCESSING strategy.** *First the stream must be summarized because the complete data cannot be stored; then specific data structures represent this summary; finally window models define **which portion of the stream contributes to the current synopsis** and how the structures are updated.* **It determines whether older data should be progressively FORGOTTEN (damped), processed in separate NON-OVERLAPPING BLOCKS (landmark), or continuously REPLACED by newer data (sliding).**

---

# PART 2 — CLUSTERING

## 7. Why clustering dominates data stream mining

> **In most cases TRUE CLASS LABELS ARE NOT AVAILABLE for stream instances, and there is NO PRIOR KNOWLEDGE about the number of classes. Therefore CLUSTERING, being UNSUPERVISED, is one of the most suitable methods for data streams.**
> *(Zubaroğlu, A., Atalay, V., "Data stream clustering: a review", Artificial Intelligence Review, 2020.)*

> **⚠️ The lecture explains WHY the literature contains far more stream CLUSTERING algorithms than stream CLASSIFICATION algorithms:** classification requires **labelled streaming instances** — each incoming item should carry a class label so the classifier can be updated. **But in many real streaming scenarios data arrive continuously while LABELS ARE NOT IMMEDIATELY AVAILABLE, or not available at all.**
> *(The situation is slightly different in **REGRESSION**, where it may be possible to **predict a value and later OBSERVE the true value**, allowing the model to be updated.)*

**A second reason clustering is attractive**: it lets us **analyse the EVOLUTION of clusters over time.**

> **The goal is not only to find clusters at a specific moment, but to observe HOW THEY CHANGE as new data arrive.** As instances arrive, **some clusters may GROW, SHRINK, MERGE, SPLIT, APPEAR or DISAPPEAR.** If the data describe users or customers, **the evolution of clusters can reveal how user profiles or behavioural patterns change.** Changes in cluster structure may indicate **the emergence of new patterns, changes in user behaviour, degradation of a system, or even possible FAULTS in the monitored process.**

### 7.1 ⚠️ BIRCH, and why order-dependence flips from bug to feature

> **BIRCH** (Chapter 5) is relevant here because it **processes data INCREMENTALLY and uses a TREE-BASED structure to summarize the data** — a form of **coreset-tree** structure.
>
> **⚠️ But BIRCH is SENSITIVE TO THE ORDER of the incoming instances**: processing data sequentially means **changing the order of the same instances may lead to different clustering results.**
>
> **In a classical stationary dataset this is a LIMITATION** — the final clustering should ideally depend only on the dataset, not on presentation order.
> **In the context of DATA STREAMS this sensitivity is NATURAL — and is exactly WHAT WE WANT.** A stream is **inherently ordered**, and the objective is to **capture the CURRENT and EVOLVING distribution**. *"The fact that the clustering result depends on the sequence of incoming data is not a problem, but it is instead what we want to obtain."*

> **Consequently: stream clustering must NOT be interpreted like clustering on a fixed dataset.** On a fixed dataset the goal is **a stable partition of all available data**; on a stream the goal is **to MAINTAIN AND UPDATE a clustering structure reflecting the most relevant and RECENT characteristics of the distribution.**

## 8. Adaptive Streaming k-Means

*(Puschmann D., Barnaghi P., Tafazolli R., "Adaptive clustering for dynamic IoT data streams", IEEE Internet of Things Journal 4(1):64–74, 2017.)*

**Inputs**: the **data stream $S$** and a parameter **$L$ — the length of the initial sequence of data used for the initialization phase** (the number of instances accumulated in initialization).

> **⚠️ NOTE WHAT IS *NOT* AN INPUT: $k$.** In standard k-means the number of clusters must be chosen in advance; **here $k$ is ESTIMATED AUTOMATICALLY through an analysis of the data distribution, performed FEATURE BY FEATURE.**

### 8.1 Initialization phase — `determineCentroids()`

> **The function has TWO purposes: it estimates a possible RANGE OF VALUES for $k$, and it determines CANDIDATE INITIAL CENTROIDS.**

**Step by step:**
1. **Estimate the PROBABILITY DENSITY FUNCTION (PDF) of the data FOR EACH FEATURE.**
2. **Determine the DIRECTIONAL CHANGES of the PDF curves.** **Each change identifies a new REGION** — *a region is the area between two consecutive directional changes of the PDF curve.*
3. **The NUMBER OF REGIONS is a CANDIDATE $k$, and the CENTERS of these regions are CANDIDATE INITIAL CENTROIDS.**

> **The idea: the SHAPE of the feature distribution provides information about the possible number and position of clusters.**

> **⚠️ But DIFFERENT FEATURES generally show DIFFERENT distributions and different centroids** — one feature may suggest two regions, another six. **For this reason the algorithm does NOT obtain a single $k$, but a RANGE $[k_{\min}, k_{\max}]$.**

4. **For each value of $k$ in the range, the PDF curve is split into EQUIPROBABLE AREAS. The boundaries of these areas are called BETA POINTS.** Since we want the **centre** of each region, **the MIDDLE POINTS between two adjacent betas are computed and saved as initial centroids.**

> **⚠️ What "equiprobable" means** (lecture): **regions containing the SAME AMOUNT OF PROBABILITY**, i.e. approximately the same proportion of data points. **The PDF is NOT divided into intervals of equal WIDTH, but into intervals of equal PROBABILITY MASS.**

5. **Run k-means for each candidate $k$** — from $k_{\min}$ to $k_{\max}$ — on the initialization sequence.
6. **Compare the clustering results using the SILHOUETTE COEFFICIENT** — *a metric measuring how COMPACT the clusters are internally and how well SEPARATED they are from each other; higher is better.* **Select the best $k$ with its corresponding centroids.**

### 8.2 Continuous clustering phase — `changeDetected()`

> Once the best $k$ is selected, **every new instance arriving from the stream is assigned to the cluster whose centroid is CLOSEST** — the usual k-means assignment.
>
> **But the distribution may change.** So the algorithm **continuously monitors whether the current clustering structure is still valid**:
> - **the STANDARD DEVIATION and MEAN of the input data are stored during execution;**
> - **the algorithm TRACKS HOW THESE TWO VALUES CHANGE over time and PREDICTS a concept drift according to the change;**
> - **when a concept drift is predicted, the current centroids are NO LONGER VALID** ⇒ **a RE-INITIALIZATION is triggered.**

> **⚠️ Note what plays the role of the synopsis here: THE CLUSTERS THEMSELVES.** As long as no relevant distributional change is detected, incoming instances are simply assigned to the existing clusters.

### 8.3 Complexity analysis

Let **$L$** be the length of the initial data sequence and **$d$** the data dimensionality.

| Operation | Complexity |
|---|---|
| **Estimating $k$ for a SINGLE dimension** | $O(L)$ |
| **Estimating $k$ for ALL dimensions** | $O(d \cdot L)$ |
| **Running k-means after determining the initial centroids** | $O(L \cdot d \cdot k \cdot cs)$, where **$cs$ is the number of different CENTROID SETS** *(no iterations of the algorithm are needed)* |
| **Assigning a newly received instance to the nearest cluster (online phase)** | $O(k)$ |

$$\textbf{Total worst case} = O(k) + O(d \cdot L) + O(L \cdot d \cdot k \cdot cs) \;=\; O(d \cdot L) + O(L \cdot d \cdot k \cdot cs)$$
*(the initialization terms dominate the simple assignment cost).*

> **⚠️ THE ROLE OF $L$ — a likely exam question.** $L$ **directly affects both the computational cost and the quality of the result:**
> - **too SMALL** ⇒ the initial sample **may not represent the real distribution**, the estimated centroids may be inaccurate, and **the algorithm may need to REINITIALIZE FREQUENTLY**;
> - **too LARGE** ⇒ **the initialization phase becomes more expensive.**
> **A trade-off between efficiency and accuracy.**

> **⚠️ WHEN THE ALGORITHM WORKS, AND WHEN IT DOESN'T.** It is **intuitive and relatively simple to implement**, and **particularly suitable when the distribution is expected to remain STABLE for reasonable periods, with only OCCASIONAL changes.**
> **If the distribution changes CONTINUOUSLY or VERY FREQUENTLY, the algorithm becomes LESS EFFECTIVE** — centroids would be recomputed too often, increasing cost and reducing stability.
> **⇒ The method works best when concept drift EXISTS BUT DOES NOT OCCUR AT EVERY MOMENT.**

## 9. MuDi-Stream

*(Amini A., Saboohi H., Herawan T., Wah T. Y., "MuDi-Stream: A multi density clustering algorithm for evolving data stream", J. Netw. Comput. Appl. 59(C):370–385, 2016.)*

> **MuDi-Stream is a HYBRID algorithm based on BOTH DENSITY-BASED AND GRID-BASED approaches.**
> **Input data instances are CLUSTERED in a density-based approach, and OUTLIERS are DETECTED using grids.**

### 9.1 The synopsis: core mini-clusters

> **For data synopsis, CORE MINI-CLUSTERS are used.**
> **Core mini-clusters are SPECIALIZED FEATURE VECTORS which keep: WEIGHT, CENTER, RADIUS, and the MAXIMUM DISTANCE FROM AN INSTANCE TO THE MEAN.**

> **The intuition** (lecture): the space is **divided into many small regions, or HYPERCUBES.** As new instances arrive they are **assigned to the corresponding hypercube**, and the algorithm **evaluates how DENSE each hypercube is.** **If a hypercube is sufficiently dense it is considered relevant and represented as a MINI-CLUSTER.**
> **⚠️ These mini-clusters are NOT the final clusters — they are the compact SUMMARY of the stream, the SYNOPSIS.**

### 9.2 The two phases

> - **ONLINE phase**: **core mini-clusters are CREATED and KEPT UP TO DATE for each new data instance.** Each incoming point either **updates the corresponding mini-cluster or leads to the creation of a new one.**
> - **OFFLINE phase**: **final clustering is executed OVER THE CORE MINI-CLUSTERS** — not on the original points. Periodically, an offline procedure **analyses the mini-clusters and combines neighbouring or density-connected ones** to obtain the final clusters.

> **⚠️ The key architectural idea: SEPARATE fast ONLINE SUMMARIZATION from periodic OFFLINE CLUSTERING.** The expensive clustering step runs **on the synopsis rather than on the entire set of incoming data.**

### 9.3 The three parameters

| Parameter | Role |
|---|---|
| **Density threshold $\alpha$** | **determines when a region of the grid is DENSE ENOUGH to become a core mini-cluster.** *Not every hypercube becomes a mini-cluster — only those containing a sufficient number of points.* |
| **Decay rate $\lambda$** | **⚠️ NECESSARY because, if data were only ACCUMULATED over time, EVERY grid cell receiving points would EVENTUALLY become dense** — which would fail to reflect changes in the distribution. With decay, **the weight of older points decreases**, so a region **dense in the past that no longer receives points gradually stops being considered dense.** *This is how the algorithm takes concept drift into account.* |
| **Grid granularity** | **determines how the feature space is divided.** A **finer** grid gives a more **detailed** representation but **more cells to manage**; a **coarser** grid **reduces the burden** but gives a **less precise** density description. |

### 9.4 The algorithm

**ONLINE phase, per instance:**
1. **The grid structure is initialized.**
2. For each new instance $x$, **search for the NEAREST core mini-cluster.**
3. **If that mini-cluster can INCLUDE the new point** ⇒ **add it and update the mini-cluster.**
4. **Otherwise, MAP the point into the corresponding GRID CELL.** **If the updated cell becomes dense enough, a NEW core mini-cluster is created from that cell.**
5. **Periodically, during PRUNING, LOW-WEIGHTED grids and LOW-WEIGHTED core mini-clusters are REMOVED** — *essential, because it eliminates regions no longer relevant to the current stream distribution.*

**OFFLINE phase (the final clustering) — a DBSCAN-like expansion over mini-clusters:**
1. **All core mini-clusters are marked UNVISITED.**
2. **An unvisited core mini-cluster is randomly chosen and marked VISITED.**
3. **If it has NO NEIGHBORS, it is marked NOISE.**
4. **If it HAS neighbors, a new final cluster is created with this mini-cluster and its neighbors.**
5. **Each unvisited core mini-cluster in the newly created final cluster is marked visited and ITS neighbors are added to the same final cluster.**
6. **The loop continues until all core mini-clusters are marked visited.**

> **⚠️ On the noise case** (lecture): a mini-cluster with no neighbours **can be interpreted as a COLLECTIVE OUTLIER**, because **the mini-cluster may contain SEVERAL points but is ISOLATED from the other dense regions.**

> **This is conceptually DBSCAN — but applied to MINI-CLUSTERS rather than to individual data points.**

### 9.5 Why the hybrid, and the limitations

> **Why combine the two paradigms:**
> - **GRID-based methods SPEED UP the computation** — the algorithm works on **cells rather than on all individual points**, summarizing the stream more efficiently;
> - **DENSITY-based methods allow clusters of ARBITRARY SHAPE** — unlike k-means, which tends to produce **compact, roughly spherical** clusters.

**⚠️ Three limitations:**
1. **MuDi-Stream is NOT SUITABLE FOR HIGH-DIMENSIONAL DATA**, which makes the processing time longer, **because of the GRID structure** — *when dimensionality increases, the number of possible hypercubes grows very rapidly.*
2. **Clustering quality STRONGLY DEPENDS on the input parameters** — **density threshold, decay rate for the damped window model, and grid granularity.** **These parameters require EXPERT KNOWLEDGE about the data.** *(Exactly as in DBSCAN: powerful when parameters are well chosen, poor or misleading when they are not.)*
3. **The difficulty grows when the data contain regions of DIFFERENT DENSITIES**: a **single** density threshold may not suit the entire space — *one that works for a dense region may miss a sparser but meaningful cluster, while one adapted to sparse regions may incorrectly merge dense areas or include noise.*

### 9.6 Complexity analysis

Let **$c$** = number of core mini-clusters, **$G$** = total density grids for all dimensions (**exponential in the number of dimensions**).

| Operation | Complexity |
|---|---|
| **Linear search on core mini-clusters for each new instance** | $O(c)$ |
| **SPACE complexity of the grid** | $O(\log G)$ — *because scattered grids are PRUNED during execution* |
| **Mapping a data instance to the grid** | $O(\log \log G)$ — *because the list of grids is maintained as a TREE* |
| **Pruning: core mini-clusters** | $O(c)$ |
| **Pruning: grids** | $O(\log G)$ |

$$\textbf{Overall} = O(c) + O(\log\log G) + O(c) + O(\log G) \;=\; \boxed{O(c) + O(\log G)}$$

> **⚠️ WHY THIS RESULT MATTERS** (lecture): **the complexity does NOT directly depend on the TOTAL NUMBER OF DATA INSTANCES received from the stream.** It depends **only on the number of maintained core mini-clusters and on the grid structure.** **⇒ MuDi-Stream does NOT become progressively more expensive simply because more data have arrived over time** — which is exactly the property a stream algorithm needs.

## 10. Comparison of stream clustering algorithms

> **There is NOT a single algorithm that is optimal in every possible context.** The choice depends on the properties of the stream and the objective of the analysis.

| | **Adaptive Streaming k-Means** | **MuDi-Stream** |
|---|---|---|
| **Multi-density clusters** | ✔ | ✔ |
| **High-dimensional data** | ✔ **suitable** | ✘ **NOT suitable** (grid structure) |
| **Adapts to drift** | ✔ (change detection + re-initialization) | ✔ (decay rate) |
| **Outlier detection** | ✘ **not provided** | ✔ (via grids / isolated mini-clusters) |
| **Expert knowledge for parameters** | **NOT strongly required** — relatively simple to use | **REQUIRED** (density threshold, decay rate, grid granularity) |

> **The criteria to weigh when choosing** (lecture): **is the data high-dimensional? Is outlier detection important? Is concept drift expected? May clusters have different densities? Is reliable expert knowledge available for setting parameters?**

---

# PART 3 — CLASSIFICATION

## 11. The problem with classical decision trees

> - **Classic decision tree learners assume ALL TRAINING DATA CAN BE SIMULTANEOUSLY STORED IN MAIN MEMORY.**
> - **Disk-based decision tree learners repeatedly READ TRAINING DATA FROM DISK sequentially** — **PROHIBITIVELY EXPENSIVE when learning complex trees.**
>
> **⇒ GOAL: design decision tree learners that READ EACH EXAMPLE AT MOST ONCE and use a SMALL CONSTANT TIME to process it.**

> **The recap of classical learning** (lecture): starting from the training set, compute a heuristic measure such as **information gain** — based on **entropy**: compute the entropy of the set, then the entropy after splitting on each candidate attribute, and **select the attribute producing the largest reduction.** Repeat recursively. **All of this requires ACCESS TO THE TRAINING SET**, or at least to the subset reaching each node.
>
> **In a stream the complete training set is NOT available from the beginning.** The algorithm **must decide which attribute to use at the root BEFORE having seen all possible examples**, then at each node **whether to split and on which attribute.**

### 11.1 The key observation

> **In order to find the best attribute at a node, IT MAY BE SUFFICIENT TO CONSIDER ONLY A SMALL SUBSET of the training examples that pass through that node.**
> - **Given a stream of examples, use the FIRST ONES to choose the ROOT attribute.**
> - **Once the root attribute is chosen, the successive examples are PASSED DOWN to the corresponding leaves, and used to choose the attribute there, and so on RECURSIVELY.**
> - **Use the HOEFFDING BOUND to decide HOW MANY EXAMPLES ARE ENOUGH at each node.**

> **⚠️ A warning printed on the slide**: *"the approach analysed in the subsequent phases has some **THEORETICAL PROBLEM**, but you can find it implemented in several tools which manage data streams."* — see §12.4.

**The algorithms:**
- **VFDT** — *"Mining High-Speed Data Streams"*, KDD 2000, **Pedro Domingos, Geoff Hulten**;
- **CVFDT** (window approach) — *"Mining Time-Changing Data Streams"*, KDD 2001, **Geoff Hulten, Laurie Spencer, Pedro Domingos**.

## 12. The Hoeffding bound

### 12.1 The statistical statement

> Let **$X$** be a random variable varying in a range **$R$**; assume we have **$n$ observations** of $X$; let **$\bar{x}$** be the average of those observations.
> **The HOEFFDING BOUND states that, with probability $1-\delta$, the true mean $\bar{X}$ of $X$ is AT LEAST $\bar{x} - \varepsilon$, where**
> $$\varepsilon = \sqrt{\frac{R^2 \ln(1/\delta)}{2n}}$$

> **Reading the formula** (lecture): **$\delta$ is the required confidence level — the probability of error — and is a FIXED value. $\varepsilon$ represents the UNCERTAINTY associated with the estimate.**
> **⚠️ AS $n$ INCREASES, $\varepsilon$ DECREASES** — with more observations, **the sample average becomes a more reliable approximation of the true mean.**

> **⚠️ A crucial property: the Hoeffding bound requires NO ASSUMPTION ABOUT THE DISTRIBUTION of the data.** This makes it **particularly useful in stream mining, where the distribution may be unknown or may change.**
> **BUT: because it is DISTRIBUTION-FREE, the bound is often CONSERVATIVE** — **a relatively LARGE number of observations may be required** before a decision can be made with high confidence.

### 12.2 How the bound is used to split a node

> Let **$G(X_i)$** be the heuristic measure used to choose test attributes (e.g. **information gain, Gini index**).
> - **$X_A$**: the attribute with the **HIGHEST** evaluation value after seeing $n$ examples;
> - **$X_B$**: the attribute with the **SECOND HIGHEST** value after seeing $n$ examples.
>
> **Given a desired $\delta$, if**
> $$\Delta\bar{G} = \bar{G}(X_A) - \bar{G}(X_B) > \varepsilon, \qquad \varepsilon = \sqrt{\frac{R^2\ln(1/\delta)}{2n}}, \qquad R = \ln c$$
> **(where $c$ is the number of classes), after seeing $n$ examples at a node, the Hoeffding bound GUARANTEES that the true $\Delta G \ge \Delta\bar{G} - \varepsilon > 0$, with probability $1-\delta$.**
> **⇒ This node can be SPLIT using $X_A$, and the succeeding examples will be passed to the new leaves.**

> **What this buys us** (lecture): **the algorithm does not need to wait for the complete training set** — impossible in a stream. **It waits until the difference between the best and second-best attributes is LARGER THAN THE UNCERTAINTY $\varepsilon$.** At that point the split is **statistically reliable**: $X_A$ is **truly better than $X_B$, and not merely better because of random variation in the observed sample.**

### 12.3 The worked example

> At a node, three attributes $A$, $B$, $C$ are evaluated.
>
> **After $n = 50$ examples:**
> $$IG(A) = 0.30 \;\text{(best)}, \qquad IG(B) = 0.28 \;\text{(second best)}, \qquad IG(C) = 0.10$$
> $$IG_1 - IG_2 = 0.30 - 0.28 = 0.02, \qquad \text{assume } \varepsilon = 0.05$$
> $$\mathbf{0.02 < 0.05 \;\Rightarrow\; DO\ NOT\ SPLIT\ YET — wait\ for\ more\ data}$$
>
> **Now more examples arrive, $n = 500$, and we recompute:**
> $$IG(A) = 0.31, \qquad IG(B) = 0.26$$
> $$IG_1 - IG_2 = 0.31 - 0.26 = 0.05, \qquad \text{with more data } \varepsilon = 0.02$$
> $$\mathbf{0.05 > 0.02 \;\Rightarrow\; SPLIT\ ON\ A}$$

> **The intuition in one sentence**: **with few instances the uncertainty is high and the algorithm must WAIT; with more instances the uncertainty DECREASES and the decision becomes statistically reliable.**
>
> **The lecture's analogy**: comparing **the running times of two runners** to determine who is faster. **With only a few races — or with times of high variance — it is impossible to decide confidently. Confidence increases as the number of runs considered increases.**

### 12.4 ⚠️ The theoretical problem, and the correct formula

> **THE PROBLEM: split measures like INFORMATION GAIN and GINI INDEX CANNOT BE EXPRESSED AS A SUM $S$ of elements $Y_i$** — but **the Hoeffding bound is defined precisely on the computation of an AVERAGE (a sum of independent terms).** So applying it directly to $\Delta G$ is **not theoretically justified.**
>
> **A corrected definition of $\varepsilon$ was introduced in:** *Rutkowski, L., Pietruczuk, L., Duda, P., Jaworski, M., "**Decision trees for mining data streams based on the McDiarmid's bound**", IEEE TKDE, 25(6), pp. 1272–1279, 2013.*
>
> **⚠️ This is the theoretically CORRECT definition and the one that should be implemented in practice — but in many applications THE ORIGINAL DEFINITION IS STILL USED to develop working Hoeffding decision trees.** *(This is what the slide's "warning" refers to: the method is theoretically flawed yet widely implemented in stream-mining tools.)*

## 13. Two practical considerations in VFDT

### 13.1 Pre-pruning via the "null" attribute

> **Pre-pruning is carried out by considering at each node a "NULL" ATTRIBUTE $X_\emptyset$ that consists of NOT SPLITTING the node.** Thus **a split will be performed if, with confidence $1-\delta$, the best split found is BETTER ACCORDING TO $G$ THAN NOT SPLITTING.**
>
> **The information gain of the null attribute is ZERO:**
> $$IG(X_\emptyset) = H(Y) - H(Y \mid X_\emptyset) = H(Y) - H(Y) = 0$$

> **⚠️ THE DECISION PROCESS THEREFORE HAS TWO LEVELS** (lecture):
> 1. **first, check whether SPLITTING IS PREFERABLE TO NOT SPLITTING** (best attribute vs. the null attribute);
> 2. **then, if splitting is justified, check WHICH ATTRIBUTE should be selected** (best vs. second-best, via the Hoeffding bound).
>
> **This prevents the tree from growing unnecessarily and avoids creating new nodes when the evidence does not support a meaningful split.**

### 13.2 $n_{\min}$ — don't recompute $G$ every time

> **The MOST SIGNIFICANT part of the time cost per example is RECOMPUTING $G$.**
> **It is INEFFICIENT to recompute $G$ for every new example, because it is UNLIKELY that the decision to split will be made at that specific point.**
> **⇒ VFDT (Very Fast Decision Tree) allows the user to specify a MINIMUM NUMBER OF NEW EXAMPLES $n_{\min}$ that must be ACCUMULATED AT A LEAF before $G$ is recomputed.**

## 14. The VFDT algorithm

**In words:**
1. **Calculate the information gain for the attributes and determine the BEST TWO attributes.**
2. **Pre-pruning: consider the "null" attribute** (its info gain is 0).
3. **At each node, check the condition** $\Delta\bar{G} = \bar{G}(X_A) - \bar{G}(X_B) > \varepsilon$.
4. **If the condition is satisfied, CREATE CHILD NODES based on the test at the node.**
5. **If not, STREAM IN MORE EXAMPLES and perform the calculations until the condition is satisfied.**

**The sufficient statistics:**
> **$n_{ijk}$ are the SUFFICIENT STATISTICS needed to compute most heuristic measures.**
> *(Lecture: instead of storing all the examples, **the algorithm stores compact COUNTS that are sufficient to evaluate possible splits** — indexed by attribute $i$, value $j$, class $k$.)*

**The parameters to specify:**
> **the SPLIT EVALUATION FUNCTION $G$** (information gain or Gini index) and **the CONFIDENCE PARAMETER $\delta$**.
> **⚠️ $\delta$ is the PROBABILITY OF CHOOSING THE WRONG ATTRIBUTE** (equivalently, $1-\delta$ is the desired probability of choosing the correct attribute at any given node). **A SMALLER $\delta$ ⇒ HIGHER CONFIDENCE, but usually REQUIRES MORE EXAMPLES before a split can be performed.**

### 14.1 Performance analysis

> Let **$p$** be **the probability that an example passed through the decision tree to level $i$ will FALL INTO A LEAF at that point.**
>
> **The EXPECTED DISAGREEMENT between the tree produced by the Hoeffding tree algorithm and that produced using INFINITE examples at each node is NO GREATER THAN $\boldsymbol{\delta/p}$.**
>
> **Required memory: $O(\text{leaves} \times \text{attributes} \times \text{values} \times \text{classes})$.**

> **How to read $\delta/p$** (lecture): **the reliability of the decision also depends on HOW MANY EXAMPLES ACTUALLY REACH each part of the tree.** **Nodes receiving many examples allow more reliable estimates and more confident splits, with lower error. Leaves receiving few examples carry greater uncertainty, because the available statistics are less stable.**
> *(On memory: this is the same as in a classical decision tree, because the algorithm must store the sufficient statistics needed to evaluate possible splits at each leaf.)*

## 15. CVFDT — adapting to concept drift

> **CVFDT (Concept-adapting Very Fast Decision Tree learner):**
> - **extends VFDT;**
> - **maintains VFDT's SPEED and ACCURACY;**
> - **DETECTS AND RESPONDS to changes in the example-generating process.**

**The observations behind it:**
> 1. **With a time-changing concept, the current SPLITTING ATTRIBUTE of some nodes MAY NOT BE THE BEST ANY MORE.**
> 2. **⚠️ An OUTDATED SUBTREE MAY STILL BE BETTER THAN THE BEST SINGLE LEAF, particularly if it is NEAR THE ROOT.** *(So it should not be discarded immediately — near the root it affects many examples.)*
> 3. **GROW AN ALTERNATIVE SUBTREE with the new best attribute at its root, when the old attribute seems out-of-date.**
> 4. **PERIODICALLY use a bunch of samples to EVALUATE THE QUALITIES of the trees.**
> 5. **REPLACE the old subtree when the alternate one becomes MORE ACCURATE.**

> **CVFDT can be interpreted as a mechanism for maintaining PARALLEL CANDIDATE STRUCTURES and selecting the one that better reflects the current data distribution.** **This lets the tree adapt to concept drift WITHOUT REBUILDING THE ENTIRE MODEL FROM SCRATCH.**

---

## Key points / potential exam pitfalls

### Definitions and the framing
- **⚠️ A data stream is UNBOUNDED and ORDERED.** Both halves matter: unbounded ⇒ cannot store; ordered ⇒ must process in arrival order, and the order carries information.
- **⚠️ APPLYING a pre-trained model to a stream is NOT this chapter's problem. LEARNING the model FROM the stream is.** Be ready to state the difference.
- **The simplifying assumption: the NUMBER OF CLASSES is constant.** Letting it vary belongs to **online learning** and adds complexity.
- **The four traditional requirements that all fail on streams: entire dataset present · multiple scans · random access · heavy learning phases.** The **RAM vs. sequential tape** analogy is the professor's own image for random access.
- **The vaccine/Twitter example is the memorable motivating case**: a model built on tweets **became outdated after a few weeks** because the discourse changed with events.

### The computational model
- **⚠️ Three requirements: SINGLE PASS (each record examined at most once) · BOUNDED STORAGE (memory holds the synopsis, not the data) · REAL-TIME (low per-record processing time).**
- **The architecture of every algorithm here: FAST ONLINE SUMMARIZATION + PERIODIC OFFLINE ANALYSIS.** Complex ML is allowed — **just not on the raw stream in real time.**
- **⚠️ DETERMINISTIC bounds = known maximum error. PROBABILISTIC bounds = within a factor $\varepsilon$ with probability at least $1-\delta$.** Know which symbol is which: **$\delta$ = probability of failure, $\varepsilon$ = tolerated error.**
- **Single-pass algorithms also apply to massive (terabyte) DATABASES**, not only to true streams.

### Concept drift
- **⚠️ The four types and their signatures:** **SUDDEN** = abrupt, only new instances after; **GRADUAL** = old and new **coexist**, proportions shift; **INCREMENTAL** = instances **morph through transitional forms belonging to NEITHER** concept; **RECURRING** = concepts **alternate, none disappears permanently.**
- **Gradual = MIXING of two recognisable concepts. Incremental = MORPHING through intermediate forms.** This is the most likely thing to be asked to distinguish.
- **Recurring drift is hardest** because the model may need to **recognise a returning concept**.

### Data structures and windows
- **Four synopsis structures: FEATURE VECTORS (summary of instances) · PROTOTYPE ARRAYS (representative instances) · CORESET TREES (hierarchical summary) · GRIDS (density in the feature space).**
- **⚠️ The three window models on TWO axes.** **Damped = UNEQUAL weights** via a decay function $f(t) = 2^{-\lambda t}$, **higher $\lambda$ ⇒ faster decay**. **Landmark = equal weights, NO overlap.** **Sliding = equal weights, LARGE overlap, FIFO by one instance.**
- **The damped model is the natural choice under concept drift**, because recent instances are more representative.

### Stream clustering in general
- **⚠️ Why more clustering than classification algorithms exist for streams: clustering is UNSUPERVISED, and in real streams LABELS are often unavailable or delayed.** *(Regression is intermediate — you can predict, then later observe the truth.)*
- **A second goal beyond grouping: OBSERVING THE EVOLUTION of clusters** (grow, shrink, merge, split, appear, disappear) — which can signal new patterns, behaviour change, system degradation or faults.
- **⚠️ BIRCH's ORDER-DEPENDENCE is a LIMITATION on a fixed dataset but exactly WHAT WE WANT on a stream**, because a stream is inherently ordered and we want the clustering to track the evolving distribution. A favourite "explain why" question.

### Adaptive Streaming k-Means
- **⚠️ $k$ is NOT an input** — it is estimated from the data. The **only** input besides the stream is **$L$**, the initialization sequence length.
- **`determineCentroids()`: estimate the PDF PER FEATURE → count DIRECTIONAL CHANGES → each change bounds a REGION → number of regions = candidate $k$, region centres = candidate centroids.**
- **⚠️ Different features give different $k$ ⇒ we get a RANGE $[k_{\min}, k_{\max}]$, not a single value.**
- **BETA POINTS are the boundaries of EQUIPROBABLE areas** — **equal PROBABILITY MASS, not equal WIDTH** — and **the initial centroids are the MIDPOINTS BETWEEN ADJACENT BETAS.**
- **The best $k$ is chosen by the SILHOUETTE COEFFICIENT** (compactness + separation; higher is better).
- **`changeDetected()` tracks the MEAN and STANDARD DEVIATION of the input and triggers RE-INITIALIZATION when they change significantly.** **The clusters themselves are the synopsis.**
- **Complexity $O(d\cdot L) + O(L\cdot d\cdot k\cdot cs)$**; the online assignment is only $O(k)$; **$cs$ = number of candidate centroid sets.**
- **⚠️ The $L$ trade-off (too small ⇒ unrepresentative, frequent re-initialization; too large ⇒ expensive initialization) and the regime of validity: the method works when drift EXISTS BUT IS OCCASIONAL, and degrades when the distribution changes continuously.**

### MuDi-Stream
- **HYBRID: DENSITY-based clustering of the data + GRID-based outlier detection.**
- **⚠️ The synopsis is CORE MINI-CLUSTERS — specialized feature vectors storing WEIGHT, CENTER, RADIUS, and MAXIMUM DISTANCE FROM AN INSTANCE TO THE MEAN.** Memorise all four fields.
- **Online = create/update mini-clusters; Offline = final clustering OVER the mini-clusters.**
- **⚠️ Why the DECAY RATE is indispensable: without it, EVERY cell that ever receives points would eventually become dense**, and the algorithm could not track drift.
- **The offline phase is DBSCAN applied to MINI-CLUSTERS, not points**; **a mini-cluster with no neighbours is NOISE — interpretable as a COLLECTIVE OUTLIER**, since it may contain several points yet be isolated.
- **Three limitations: NOT suitable for high dimensions (grid explosion) · strong parameter dependence requiring EXPERT KNOWLEDGE · trouble with regions of DIFFERENT DENSITIES** (one threshold cannot fit the whole space).
- **⚠️ Complexity $O(c) + O(\log G)$ — and the point is that it does NOT depend on the number of instances seen so far**, only on the mini-clusters and the grid. Note the individual pieces: **$O(\log\log G)$ to map an instance (tree-structured grid list), $O(\log G)$ grid space because scattered grids are pruned.**

### VFDT / Hoeffding trees
- **The goal: read each example AT MOST ONCE, in small constant time.**
- **⚠️ The key observation: to pick the best attribute at a node it may SUFFICE to consider a SMALL SUBSET of the examples passing through it.**
- **The bound: $\varepsilon = \sqrt{\dfrac{R^2\ln(1/\delta)}{2n}}$, with $R = \ln c$ ($c$ = number of classes). $\varepsilon$ DECREASES as $n$ grows.**
- **Split when $\Delta\bar{G} = \bar{G}(X_A) - \bar{G}(X_B) > \varepsilon$** — then with probability $1-\delta$ the true difference is positive.
- **⚠️ The bound is DISTRIBUTION-FREE — a strength (no assumptions) that is also a weakness (CONSERVATIVE, so many observations may be needed).**
- **Be able to redo the worked example**: at $n=50$, $0.30-0.28 = 0.02 < \varepsilon = 0.05$ ⇒ **wait**; at $n=500$, $0.31-0.26 = 0.05 > \varepsilon = 0.02$ ⇒ **split on $A$.**
- **⚠️ THE THEORETICAL FLAW: information gain and Gini CANNOT be written as a SUM of independent elements, which is what the Hoeffding bound assumes.** The corrected $\varepsilon$ comes from the **McDiarmid bound** (Rutkowski et al., 2013) — **yet the original formulation is still what most tools implement.** Knowing this caveat is exactly what the slide's "warning" is testing.
- **⚠️ PRE-PRUNING with the NULL attribute ($IG = 0$) makes the decision TWO-LEVEL: (1) is splitting better than not splitting? (2) which attribute?**
- **$n_{\min}$ exists because RECOMPUTING $G$ is the dominant per-example cost**, and one extra instance rarely flips the decision.
- **$n_{ijk}$ = the SUFFICIENT STATISTICS** — compact counts replacing stored examples.
- **$\delta$ = probability of choosing the WRONG attribute; smaller $\delta$ ⇒ more confidence but MORE EXAMPLES needed before splitting.**
- **Expected disagreement with the infinite-example tree $\le \delta/p$**, where **$p$ = probability that an example reaching level $i$ falls into a leaf there.** **Memory $O(\text{leaves} \times \text{attributes} \times \text{values} \times \text{classes})$.**

### CVFDT
- **⚠️ The central insight: an OUTDATED SUBTREE MAY STILL BEAT THE BEST SINGLE LEAF, especially NEAR THE ROOT** — so do not prune it on suspicion.
- **Instead: GROW AN ALTERNATIVE SUBTREE with the new best attribute at its root, evaluate both PERIODICALLY on recent samples, and REPLACE the old one only when the alternative becomes more accurate.**
- **This is "maintain parallel candidate structures", and it avoids rebuilding the model from scratch.**

### Cross-chapter connections
- **BIRCH** (Chapter 5) is the direct ancestor of stream synopses — **incremental, tree-structured (a coreset tree), and order-dependent**; here its order-dependence becomes a virtue. Note also that **BIRCH's CF vectors** are the same "compact sufficient statistics" idea as **core mini-clusters** and as **$n_{ijk}$** in VFDT.
- **DBSCAN** (Chapter 5) is the template for MuDi-Stream's offline phase — **core objects, neighbours, expansion, noise** — and MuDi-Stream inherits its **parameter-sensitivity** and its **trouble with varying densities**, which was exactly DBSCAN's known weakness.
- **k-means / c-means and the SILHOUETTE coefficient** (Chapter 5) underpin Adaptive Streaming k-Means; the **multiple-scan requirement of c-means** is the lecture's own example of why classical algorithms fail on streams.
- **Decision trees, entropy, information gain and Gini** (Chapter 4) are the basis of VFDT; **pre-pruning** also comes from there, here implemented via the **null attribute**.
- **Grid-based clustering** connects to **CLIQUE** (Chapters 5 and 9), and the **grid explosion in high dimensions** is the **curse of dimensionality** again (Chapters 2, 3, 5, 7, 9).
- **Collective outliers** (the isolated mini-cluster) is terminology from **Chapter 7 (Outlier Analysis)**; stream clustering doubles as an outlier/fault-detection mechanism, as Chapter 7's industrial fault-detection motivation anticipated.
- **Concept drift** is the streaming counterpart of the **model-validity problem**, and it is the same phenomenon that **CVFDT** and the **federated non-IID discussion** (Chapter 14 §8.1) both wrestle with — distributions that refuse to stay put.

---

*File auto-generated by merging `15-DataStream.pdf` (professor's slides, 23 pages) and `15 - DataStream sbobine.pdf` (lecture notes, 21 pages — lessons L16–L17 of 28/04 and 05/05/2026). Formulas and pseudo-code that the deck stored as images (the decay function, the landmark/sliding window index formulas, the Hoeffding bound and its McDiarmid correction, the VFDT pseudo-code, the algorithm comparison table) have been reconstructed from the surviving text and the lecture's commentary. For questions about this chapter, refer only to this file.*
