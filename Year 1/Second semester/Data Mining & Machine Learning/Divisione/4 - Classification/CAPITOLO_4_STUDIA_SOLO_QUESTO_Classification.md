# Chapter 4 — Classification and Prediction

> **Course:** Data Mining and Machine Learning / Bioinspired Computational Methods — Biological Data Mining
> **Instructor:** Prof. Francesco Marcelloni (University of Pisa)
> **Sources merged into this file:** professor's slides (`4-Classification.pdf`, 99 pages, based on the Han–Kamber–Pei textbook) + lecture notes covering L07, L08, L09, L11 (14/16/21/28-10-2025) (`4 - Classification Sbobine.pdf`, 81 pages)
>
> **Instructions for Claude:** this is the single reference file for Chapter 4. When the user asks study questions about this chapter, use **only** this file as context (no need to re-read the original PDFs unless a detail is missing). The file merges the official slide content with the lecture's discursive explanations and worked examples, keeping both levels of detail.
>
> **⚠️ Professor's emphasis:** *Model Evaluation and Selection* (§9) is flagged in the lecture notes as **very important for this course**. *Class-imbalanced data* (§10.5) and *Random Forest* (§10.4) are flagged as **important for the project**.

---

## Chapter outline (official slide structure)
1. Classification: Basic Concepts
2. Preparing Data for Classification
3. Comparing Classification Methods
4. Decision Tree Induction
5. Bayes Classification Methods
6. Rule-Based Classification
7. Classification by Using Frequent Patterns
8. Lazy Learners
9. Model Evaluation and Selection
10. Techniques to Improve Classification Accuracy: Ensemble Methods
11. Summary

---

## 1. Classification: Basic Concepts

**The core idea**: we have a number of objects and we want to classify them (associate a class to each of them). To do this we need a **training set** — a set of objects with labels already associated.

### Classification vs Numeric Prediction
- **Classification**: predicts **categorical** class labels (discrete or nominal); constructs a model based on the training set and the values (class labels) of a classifying attribute, then uses it to classify new data.
- **Numeric Prediction**: models **continuous-valued** functions, i.e., predicts unknown or missing values.

**Typical applications:**
- Credit/loan approval: is a loan applicant safe or risky?
- Medical diagnosis: is a tumor cancerous or benign?
- Fraud detection: is a transaction fraudulent?
- Web page categorization: which category does it belong to?

> **Context from the lecture**: this course deals with **discriminative (classical) AI** — the classical ML approaches. In recent years **generative AI** (text, images, …) has also emerged, but the classification problems analyzed here are solved with the discriminative approach. Remember ML is a *subset* of AI.

### Classification — A Two-Step Process

**First Step — Model construction (learning step / training phase)**
- Start from the labeled **training set**, apply a learning algorithm, and generate a **model**. This is the actual ML step.
- Each object/tuple is represented by an **n-dimensional attribute vector (feature vector)** $X = (x_1, x_2, \dots, x_n)$, where each dimension is a feature.
- Each tuple is assumed to belong to a predefined class, determined by the **class label attribute**.
- The model can be represented as: **classification rules**, **decision trees**, or **mathematical formulae**.

**Second Step — Model usage: classifying future/unknown objects**
- Once the model is available, use it to classify new **unlabeled** instances.
- But first: **estimate the accuracy** of the classifier. The known label of a test sample is compared with the result produced by the model.
- **Accuracy rate** = percentage of test set samples correctly classified by the model *(valid only if the dataset is not imbalanced — see §9)*.
- If accuracy is acceptable, use the model to classify tuples whose class labels are unknown.

> **Why a separate test set?** (from the lecture) Once we deploy the model on truly unlabeled objects, we *cannot* compute the accuracy rate — there's no ground truth to compare against. So we need another labeled set, the **test set**.

### The Test Set
- The test set is **independent of the training set**.
- It has the **same format** as the training set (it is also labeled) — the difference is its **purpose of use**.
- Because it's independent, it lets us test whether the model **generalizes**: whether it correctly classifies instances different from those used in training.
- In practice we usually don't get a pre-split training/test pair — we get one labeled dataset and must split it ourselves (done in a smart way, see §9).

### Overtraining (Overfitting)
**Overtraining** happens when the classifier learns *too much* from the training set — it's too tuned to it and fails to generalize.

**How to detect it:**
1. Use the training set to generate the model.
2. Classify **both** the training set and the test set with that model, measuring the accuracy rate of each.
3. If accuracy on the training set is **much higher** than on the test set → indication of overtraining. If the two are approximately similar → the classifier is *not* overtrained.

### Supervised vs Unsupervised Learning
- **Supervised learning (classification)**: training data comes with **labels** indicating the class of each observation. The labels "supervise" the whole process — they are a guide used to generate the model. New data is classified based on the training set.
- **Unsupervised learning (clustering)**: class labels of training data are **unknown**. Given a set of measurements/observations, the aim is to establish the existence of classes or clusters in the data.

### Worked walkthrough (from the lecture)

**Step 1 — Model construction.** A tiny training set of 6 people, described by `NAME` (not usable as a feature — it's just an ID), `RANK` and `YEARS` (years in that rank), with class `TENURED`:

| NAME | RANK | YEARS | TENURED |
|---|---|---|---|
| Mike | Assistant Prof | 3 | no |
| Mary | Assistant Prof | 7 | yes |
| Bill | Professor | 2 | yes |
| Jim | Associate Prof | 7 | yes |
| Dave | Assistant Prof | 6 | no |
| Anne | Associate Prof | 3 | no |

Fixing the model type as *a set of rules*, and reasoning intuitively: class = yes when `rank = professor` **or** `years > 6`. So the classifier is:

```
IF rank = 'professor' OR years > 6 THEN tenured = 'yes'
```

**Step 2 — Model usage.** We discard the training set and use the **test set** (same format):

| NAME | RANK | YEARS | TENURED |
|---|---|---|---|
| Tom | Assistant Prof | 2 | no |
| Merlisa | Associate Prof | 7 | no |
| George | Professor | 5 | yes |
| Joseph | Assistant Prof | 7 | yes |

Accuracy is **not** 100%: Merlisa is classified as tenured (7 > 6) but she is actually not.

---

## 2. Preparing Data for Classification

Before applying any ML algorithm we must prepare the data (this connects directly to Chapter 3).

- **Data cleaning**: preprocessing to remove/reduce noise and treat missing values.
- **Relevance Analysis**: many attributes may be irrelevant for the classification problem.
  - Correlation analysis
  - Attribute subset selection (feature subset selection)
  - **Ideally**, the time spent on relevance analysis should be *less* than the time that would have been spent learning from the original full set of attributes.
- **Data Transformation and Reduction**
  - **Normalization**: scale attribute values into a small specified range.
  - **Reduction** (when there is a large number of instances): generalization to higher-level concepts (e.g., numeric `income` → low/medium/high); several methods (wavelet transform, PCA, discretization via binning, histogram analysis, clustering).
  - **Discretization**: exploit the **ground truth** (class labels) to determine an optimal discretization of continuous-valued attributes.
    - Some learning algorithms in the literature are applicable **only to categorical attributes**.
    - Especially with big data, performing discretization *concurrently* with learning can be very computationally expensive.
    - Typically we use **heuristic approaches based on specific metrics**.

### 2.1 Supervised Discretization: the Fayyad & Irani approach (1992)

**Objective**: find an **optimal partition** for each continuous-valued attribute $A$. Partitions are determined by a set of **cut points**. "Optimal" is relative to the classification problem — we want a partition that helps solve the classification task.

**Key observation (proved in the original paper)**: the optimal cut points for the metrics they use always lie **between two examples of different classes** in the sequence of sorted values of attribute $A$.

> **Why this matters** (from the lecture): when discretizing, we sort the attribute's values in increasing order; each value belongs to an instance which belongs to a class. The observation tells us the optimal cut points sit in the middle between an instance of one class and an instance of another class. **This dramatically limits the search space** — we don't need to test cut points inside a zone where all instances share one class. With many instances, the number of candidate cut points would otherwise be huge.

**Definition (boundary point)**: the potential cut point $T$ is a **boundary point** $b$ **iff** in the sequence of examples sorted by the value of $A$, there exist two examples $e_1, e_2$ having **different classes**, such that $A(e_1) < T < A(e_2)$ and there exists no other example $e'$ such that $A(e_1) < A(e') < A(e_2)$. We assume $b$ is the **midpoint** between $A(e_1)$ and $A(e_2)$.

Let $B_A$ be the set of all candidate boundary points for attribute $A$. The algorithm exploits **entropy** as its quality measure.

**Class entropy of a subset $S$**: let there be $k$ classes $C_1,\dots,C_k$, and let $P(C_i,S)$ be the proportion of examples in $S$ having class $C_i$ (i.e., the probability of class $C_i$):

$$Ent(S) = -\sum_{i=1}^{k} P(C_i,S)\cdot \log\big(P(C_i,S)\big)$$

> **Why entropy = 0 means a pure subset** (lecture explanation): if the subset contains only one class, that class has probability 1, so $\log(1)=0$ and its contribution is 0; all other classes have probability 0, so their products with the log are 0 too. Hence total entropy = 0. **A pure interval is fantastic for classification** — all values in that interval belong (with high probability) to that single class. Our aim is to isolate intervals that are pure sets.

**Class information entropy of the partition induced by $T$** (splitting $S$ into $S_1$ = examples with $A$-values not exceeding $T$, and $S_2 = S - S_1$):

$$EP(A,T;S) = \frac{|S_1|}{N}Ent(S_1) + \frac{|S_2|}{N}Ent(S_2)$$

i.e. the entropy of each side, **weighted by its cardinality** over the total number of points.

**The cut point $T_A$ for which $EP(A,T_A;S)$ is minimal** among all candidate cut points is taken as the **best cut point**.
> A very low $EP$ means both the left and right subsets are very close to being pure — the best situation for classification.

**Procedure**: at the first step, test all possible cut points, compute $EP$ for each, and cut using the one giving the minimum $EP$. Then **repeat recursively** on $S_1$ and on $S_2$, until a stopping condition is met.

**Stopping condition** (the professor explicitly said: *"We don't have to remember this formula, we just have to remember that there is a termination condition in the Fayyad–Irani approach"*):

$$G(A,T_A;S) > \frac{\log_2(N-1)}{N} + \frac{\Delta(A,T_A;S)}{N}$$

where $\Delta(A,T_A;S) = \log_2(3^c - 2) - \big[c\cdot E(S) - c_1E(S_1) - c_2E(S_2)\big]$, with $c$, $c_1$, $c_2$ the numbers of classes in $S$, $S_1$, $S_2$ respectively, and $G(A,T_A;S) = E(S) - EP(A,T_A;S)$.

**Key advantage**: no parameters need to be set beforehand — just apply the approach and at the end you get a discretization of your variable. Each resulting interval gets a label, and all points inside the interval are characterized by that same label value.

> **Conclusion from the lecture**: this approach is very interesting because it genuinely **exploits supervision** — we discretize by exploiting the classes, generating a partition that is well-suited to the classification task, since we're looking for pure intervals.

---

## 3. Comparing Classification Methods

With thousands of models and learning algorithms available, our task is to test different solutions and understand how to compare them to find the optimal one. The criteria:

- **Accuracy**: the classifier should achieve a high percentage of correctly classified instances.
- **Speed** — two phases matter:
  - time to **construct** the model (training time) → *the most important time for us (the developers)*
  - time to **use** the model (classification/prediction time) → *the most important time for the user*
- **Robustness**: handling noise and missing values. What's really interesting is understanding, if a misclassification happens, what the **effect** of that misclassification is. Robustness matters especially in critical applications. *The real problem with ML today is the impossibility of predicting the consequences of possible faults of the ML model.*
- **Scalability**: efficiency in disk-resident databases.
- **Interpretability**: very important nowadays because regulations demand **transparency**, and a transparent model must be interpretable — we must be able to explain **how the model works**. The simple rule model above is highly explainable: just reading the rule explains why that input produced that output. Neural networks, by contrast, are not interpretable — a big problem given current regulations.
- **Other measures**: goodness of rules, decision tree size, compactness of classification rules.

---

## 4. Decision Tree Induction

> *"This model is very important because it is one of the first models proposed and because in the last few years it was re-explored. It is considered one of the most interpretable and most transparent classifiers."*

### 4.1 Structure and Usage

**Decision Tree Induction**: learning decision trees from class-labeled training tuples.

**Decision tree structure** — a flowchart-like tree structure where:
- each **internal node (non-leaf node)** denotes a **test on an attribute**;
- each **branch** represents an **outcome** of the test (e.g., a decision node tests whether age > or < 50, giving two branches);
- each **leaf node (terminal node)** holds a **class label**.

**How it classifies**: given a tuple, its attribute values are tested against the tests at each decision node; following the corresponding branches, we arrive at a leaf where we find the class prediction.

**Why so popular?**
- **No domain knowledge needed** — no input from an expert; just start from data and apply the learning algorithm.
- **No parameter setting** — a genuine exception among ML methods (compare with neural networks, where you must set the number of neurons, layers, etc.).
- **Can manage high-dimensional data.**
- **Representation is intuitive and easy to assimilate.**

**Multi-way vs Binary trees:**
- **Multi-way (multi-split) decision tree**: each node can have two **or more** branches.
- **Binary decision tree**: each decision node can have **only two** branches.

### 4.2 The running example dataset (`buys_computer`)

14 instances, 4 attributes → predict class `buys_computer` ∈ {yes, no}:
- `age` ∈ {youth, middle_aged, senior}
- `income` ∈ {high, medium, low}
- `student` ∈ {yes, no}
- `credit_rating` ∈ {fair, excellent}

The resulting tree:

```
                    age?
       youth      middle_aged     senior
         |             |             |
     student?         Yes      credit_rating?
     no    yes                 excellent  fair
      |     |                      |       |
     No    Yes                    No      Yes
```

**Classifying a new instance** (age=youth, student=yes, credit_rating=excellent): test `age` → follow *youth* → test `student` → follow *yes* → leaf: class = **yes**.

### 4.3 The Basic Algorithm

The basic algorithm is a **greedy algorithm**: it follows the problem-solving heuristic of making the **locally optimal choice at each stage**, hoping to find the global optimum.

> **Important caveat** (lecture): we can generate a very high number of different decision trees from the same data. There is no "best one" in an absolute sense — what we do is **optimize step by step**: at each node select what we consider the best attribute. This does *not* guarantee the optimal decision tree at the end, but we're probably very close to the optimal solution.

The tree is constructed in a **top-down recursive divide-and-conquer** manner:
- At start, **all training examples are at the root**.
- Attributes are **categorical** (if continuous-valued, they're **discretized in advance**).
- Examples are **partitioned recursively** based on selected attributes.
- Test attributes are selected on the basis of a **heuristic or statistical measure** (e.g., information gain).

**Conditions for stopping partitioning:**
1. **All samples for a given node belong to the same class** → pure partition, no need to split again.
2. **There are no remaining attributes for further partitioning** → **majority voting** is employed to label the leaf (we couldn't isolate a pure partition, but we must terminate, so we take the majority class among the remaining samples).
3. **There are no samples left** (e.g., we discover that for `senior` we have no samples at all → stop on that branch).

**Strategy for selecting attributes**: create partitions at each branch that are **as pure as possible**. A partition is *pure* when all tuples in it belong to the same class.

> **Intuition** (lecture): if an attribute splits the training set into pure partitions, that's a fantastic attribute — with it we've solved the classification problem. E.g., in a 3-class problem, if `age=youth` → all C1, `age=middle` → all C2, `age=senior` → all C3, then this single attribute decides the class. Realistically this never happens perfectly, so we look for the attribute that gets us **closest** to pure partitions. That's why we need a **metric**.

### 4.4 Attribute Selection Measure #1: Information Gain (ID3 / C4.5)

Let $D$ be a training set of class-labeled tuples, and suppose the class label attribute has $m$ distinct values defining $m$ classes $C_i$ ($i=1,\dots,m$).

- Select the attribute with the **highest information gain**.
- Let $p_i$ be the probability that an arbitrary tuple in $D$ belongs to class $C_i$, estimated by $|C_{i,D}|/|D|$.
- **Expected information (entropy) needed to classify a tuple in $D$** *(flagged VERY IMPORTANT in the lecture notes)*:

$$Info(D) = -\sum_{i=1}^{m} p_i \log_2(p_i)$$

> **Entropy behaviour in a 2-class problem** (lecture): if $P(C_1)=0$ then $P(C_2)=1$, and entropy = 0. When $P(C_1)\to 0.5$ (so $P(C_2)\to 0.5$), entropy reaches its **maximum** — meaning maximum confusion between classes. This is the worst situation: the two classes are equiprobable.

**After splitting on attribute $A$** into $v$ partitions $\{D_1,\dots,D_v\}$ using $A$'s $v$ distinct values $\{a_1,\dots,a_v\}$:

$$Info_A(D) = \sum_{j=1}^{v} \frac{|D_j|}{|D|}\cdot Info(D_j)$$

i.e., the entropy of each subset $D_j$, **weighted** by the ratio between the number of samples having value $a_j$ for $A$ and the total number of samples in $D$.

> **Why the weights?** (lecture) We want to account for the **number of samples** in each partition. Consider the degenerate case where a split isolates a single instance: that partition is certainly pure, but it doesn't really mean anything. Adding weights makes us consider the **cardinality** of the isolated sets.

**Information Gain** — our metric to evaluate a single attribute:

$$Gain(A) = Info(D) - Info_A(D)$$

**The algorithm's idea**: test all attributes, compute $Gain$ for each, **select the attribute with the highest gain** for the root. Split the training set into subsets accordingly, then **repeat the same process recursively** on each subset.

#### Worked example (full computation)

**Step 1 — Entropy of the whole dataset.** 9 instances with class *yes*, 5 with class *no*, out of 14:

$$Info(D) = I(9,5) = -\frac{9}{14}\log_2\!\left(\frac{9}{14}\right) - \frac{5}{14}\log_2\!\left(\frac{5}{14}\right) = 0.940$$

This is a **very high** value → lots of confusion in terms of classes.

**Step 2 — Test each attribute.** Starting with `age`:

| age | $p_i$ (yes) | $n_i$ (no) | $I(p_i,n_i)$ |
|---|---|---|---|
| youth | 2 | 3 | 0.971 |
| middle_aged | 4 | 0 | **0** |
| senior | 3 | 2 | 0.971 |

> `youth`: 2 yes vs 3 no → highly indeterminate → entropy 0.971. `middle_aged`: **all instances are class yes → pure partition → entropy 0**. `senior`: same situation as youth.

$$Info_{age}(D) = \frac{5}{14}I(2,3) + \frac{4}{14}I(4,0) + \frac{5}{14}I(3,2) = 0.694$$

$$Gain(age) = Info(D) - Info_{age}(D) = 0.940 - 0.694 = \mathbf{0.246}$$

Repeating for the other attributes:
- $Gain(income) = 0.029$
- $Gain(student) = 0.151$
- $Gain(credit\_rating) = 0.048$

**→ `age` has the highest gain → it becomes the root**, splitting the dataset into 3 subsets.

**Step 3 — Check stopping conditions on the three subsets.** Condition 1 (*all samples same class*) is **satisfied for `middle_aged`** (all yes) → that branch becomes a leaf immediately, labeled *yes*, with no further tests. For `youth` and `senior`, none of the three stopping conditions holds (attributes remain, instances remain, classes are mixed) → continue splitting recursively with the same procedure.

### 4.5 Information Gain for Continuous-Valued Attributes

**Best solution**: discretize **before** starting the decision tree algorithm (e.g., using Fayyad & Irani, §2.1).

**Alternative (older) approach** — determine the best split point for a continuous attribute $A$:
- Sort the values of $A$ in increasing order.
- Typically the **midpoint** between each pair of adjacent values is considered as a possible split point: $(a_i + a_{i+1})/2$.
- The point with the **minimum expected information requirement** for $A$ is selected as the split point.
- Split: $D_1$ = tuples with $A \le$ split-point, $D_2$ = tuples with $A >$ split-point.

> **Why this is rarely used today** (lecture): testing all possible midpoints is only feasible with a limited number of instances. With modern (large) training sets it becomes very expensive — so we normally **discretize first** and then apply the tree learning algorithm.

### 4.6 Attribute Selection Measure #2: Gain Ratio (C4.5)

**Problem with Information Gain**: it is **biased towards attributes with a large number of values**. For instance, an attribute acting as a **unique identifier** produces perfectly pure partitions → $Info_{product\_ID}(D) = 0$ → maximum gain.

> **Concrete illustration** (lecture): imagine a training set of people including a national ID/tax-code attribute — every person has a different one. Splitting on it produces one subset per instance, each trivially pure (entropy 0). So it looks like the "best" attribute, but the resulting tree is useless: a **new** person will have an ID different from all the others, so no branch matches and we can't classify them at all.

**Solution — C4.5 (successor of ID3) uses gain ratio**, normalizing information gain by a "**split information**":

$$SplitInfo_A(D) = -\sum_{j=1}^{v}\frac{|D_j|}{|D|}\times \log_2\!\left(\frac{|D_j|}{|D|}\right)$$

This value represents the **potential information generated by splitting** $D$ into $v$ partitions, corresponding to the $v$ outcomes of a test on $A$.

$$GainRatio(A) = \frac{Gain(A)}{SplitInfo(A)}$$

*Example*: $GainRatio(income) = 0.029 / 1.557 = 0.019$.

**The attribute with the maximum gain ratio is selected as the splitting attribute.**

> **Effect**: when an attribute has very many possible values (like an ID), $SplitInfo$ becomes large, so the gain ratio becomes **low** — exactly the correction we wanted.

### 4.7 Attribute Selection Measure #3: Gini Index (CART, IBM IntelligentMiner)

If a dataset $D$ contains examples from $m$ classes, the Gini index is:

$$gini(D) = 1 - \sum_{i=1}^{m}p_i^2$$

where $p_i$ is the relative frequency of class $i$ in $D$.

> **Comparison with entropy** (lecture): plotting both for a 2-class problem, the **shape/trend is the same**, though the Gini index reaches a **lower peak value** than entropy.

**Key characteristic: the Gini index considers a BINARY split for each attribute** — it only generates **binary decision trees**.

**Handling a discrete attribute with $v > 2$ distinct values**: all possible subsets that can be formed from $A$ are examined to determine the optimal binary split. E.g., for {low, medium, high} the possible subsets are: {low,medium}, {low,high}, {medium,high}, {low}, {medium}, {high} — so one branch could be {low,medium} and the other {high}, etc.

**Gini index when splitting $D$ on $A$ into $D_1$ and $D_2$:**

$$gini_A(D) = \frac{|D_1|}{|D|}gini(D_1) + \frac{|D_2|}{|D|}gini(D_2)$$

**Reduction in Impurity:**

$$\Delta gini(A) = gini(D) - gini_A(D)$$

**The attribute that maximizes the reduction of impurity** (equivalently: has the **minimum Gini index**) is selected as the splitting attribute.

#### Worked example
- $D$ has 9 tuples with `buys_computer = yes` and 5 with `no`:
$$gini(D) = 1 - \left(\frac{9}{14}\right)^2 - \left(\frac{5}{14}\right)^2 = 0.459$$
- Start with attribute `income` and consider each possible splitting subset. The subset {low, medium} gives 10 tuples in $D_1$ and 4 in $D_2$.
- Gini values for the remaining subsets: $Gini_{\{low,high\}} = Gini_{\{medium\}} = 0.458$; $Gini_{\{medium,high\}} = Gini_{\{low\}} = 0.450$.
- **Split on {low, medium} (vs {high})** since it has the lowest Gini index.

> Note: the Gini index may need other tools (e.g., clustering) to get possible split values for continuous attributes.

### 4.8 Comparing Attribute Selection Measures

All three generally return good results, **but each has a bias**:

| Measure | Bias / limitation |
|---|---|
| **Information gain** | biased towards **multivalued** attributes |
| **Gain ratio** | tends to prefer **unbalanced splits** in which one partition is much smaller than the others |
| **Gini index** | biased to multivalued attributes; has difficulty when the **number of classes is large**; tends to favor tests resulting in **equal-sized partitions** and purity in both partitions |

> **Bottom line**: a perfect index does not exist — each can give very good results but each has problems. **Normally we use the gain ratio.**

### 4.9 Decision Tree Algorithm (pseudocode)

**Inputs:**
- `D` — data partition: a set of training tuples and their class labels;
- `attribute_list` — the set of candidate attributes;
- `Attribute_selection_method` — a procedure determining the splitting criterion that "best" partitions the tuples into individual classes. This criterion consists of a splitting attribute and possibly a split-point or splitting subset.

> **This is the only choice we have**: decide whether we want a binary or multiway tree, and which metric to use. **No other parameters need setting.**

**Output**: a decision tree. The algorithm is **recursive** (initialization, stop condition, recursion).

```
create a node N;

// two termination conditions tested first
if tuples in D are all of the same class C then
    return N as a leaf node labeled with the class C;          // stopping condition 1
if attribute_list is empty then
    return N as a leaf node labeled with the majority class in D;  // stopping condition 2 (majority voting)

apply Attribute_selection_method(D, attribute_list) to find the "best" splitting_criterion;
label node N with splitting_criterion;

if splitting attribute is discrete-valued AND multiway splits allowed then   // not restricted to binary trees
    attribute_list = attribute_list - splitting_attribute;      // remove the attribute from the list

// create a node for each possible value
for each outcome j of splitting_criterion do                    // partition tuples and grow subtrees
    let D_j be the set of data tuples in D satisfying outcome j;   // a partition
    if D_j is empty then                                        // stopping condition 3
        attach a leaf labeled with the majority class in D to node N;
    else
        attach the node returned by Generate_decision_tree(D_j, attribute_list) to node N;
endfor

return N;
```

> **The problem left at the end**: the tree we obtain suffers from **overtraining**. Because the stopping conditions are generic, the tree can grow very deep; in deep parts it isolates a very small region of the space, over-tuning on the training set and failing to generalize.

### 4.10 Other Attribute Selection Measures *(can be skipped — lecture notes mark these as optional)*

- **CHAID**: popular decision tree algorithm, measure based on the $\chi^2$ test for independence.
- **C-SEP**: performs better than information gain and Gini index in certain cases.
- **G-statistic**: close approximation to the $\chi^2$ distribution.
- **MDL (Minimal Description Length)** principle — the simplest solution is preferred: the best tree is the one requiring the fewest bits to both (1) encode the tree and (2) encode the exceptions to the tree.
- **Multivariate splits** (partition based on multiple variable combinations): **CART** finds multivariate splits based on linear combinations of attributes.

**Which is best?** Most give good results, none is significantly superior; all have some bias. However, the **time complexity of decision tree induction increases exponentially with tree height**, so measures producing **shallower trees** may be preferred — but shallow trees tend to have many leaves and higher error rates.

### 4.11 Overfitting and Tree Pruning

**Overfitting**: an induced tree may overfit the training data — too many branches, some reflecting anomalies due to **noise or outliers** → poor accuracy for unseen samples. **The decision boundary gets distorted by noise points.**

**Detecting overfitting:**
1. Learn the decision model using the training set.
2. Compute accuracy $A_{Training}$ by classifying the training set.
3. Compute accuracy $A_{Test}$ by classifying the test set.
4. If $A_{Test} \ll A_{Training}$ → **overtraining**.

> **The characteristic curve**: as the number of nodes increases, the **error on the training set decreases** steadily; the **error on the test set** stays stable or **increases**.

#### Estimating Generalization Errors
- **Re-substitution errors**: error on the **training** set ($e_{TR}$).
- **Generalization errors**: error on the **test** set ($e_{TS}$).

**Methods for estimating generalization errors:**
- **Optimistic approach**: $e_{TS} = e_{TR}$ — very optimistic, especially under overtraining.
- **Pessimistic approach**: add a penalty of 0.5 per leaf node: $e_{TS} = e_{TR} + 0.5 \times N$ (N = number of leaf nodes).
  - *Example*: a tree with 30 leaf nodes and 10 errors on 1000 training tuples → $e_{TR} = 10/1000 = 1\%$, but $e_{TS} = (10 + 30\times0.5)/1000 = 2.5\%$.
- **Reduced Error Pruning (REP)**: use a **pruning data set** to estimate generalization error. The initial dataset is split into training set and test set, and then the **training set is split again** into a **growing set** (used to learn the tree) and a **pruning set** (used to prune it).

#### Occam's Razor
- Given two models of **similar generalization errors**, prefer the **simpler** model over the more complex one.
- For complex models there is a greater chance that the fit happened **accidentally** because of errors in the data.
- Therefore one should include **model complexity** when evaluating a model.

> Reducing complexity also helps **interpretability**: fewer nodes → easier to understand how the tree works.

#### Prepruning
**Halt tree construction early**, using stopping conditions beyond the three basic ones.
- Typical stopping conditions for a node:
  - Stop if all instances belong to the same class.
  - Stop if all the attribute values are the same.
- **More restrictive conditions:**
  - Stop if the **number of instances is less than a user-specified threshold**.
  - Stop if the **class distribution of instances is independent of the available features** (e.g., using a Chi-square test).
  - Stop if **expanding the current node does not improve impurity measures** (e.g., information gain or Gini index).

> **Professor's experience**: the 1st and 3rd restrictive conditions are the ones commonly used. **But prepruning is not easy**: every restrictive condition requires setting a **threshold**, and we don't know what the optimal threshold should be. **This is why we prefer post-pruning.**

#### Postpruning
**Remove branches from a "fully grown" tree**, obtaining a sequence of progressively pruned trees.
- Remove subtrees from a fully-grown tree, **starting from the leaves**. A subtree at a given node is pruned by removing its branches and **replacing it with a leaf**, labeled with the **most frequent class** among the subtree being replaced.
- Use a set of data **different from the training data** to decide which is the "best pruned tree".
- We expect some **loss in accuracy on the training set**, but a **gain on the pruning/test set**.

#### Reduced Error Pruning (REP) — detailed

Use the **pruning set** to estimate the accuracy of subtrees and of individual nodes. Let $T$ be a subtree rooted at node $v$.

$$\text{Gain from pruning} = \#\text{misclassifications in } T - \#\text{misclassifications at } v$$

**Repeat**: prune at the node with the **largest gain**, until only negative-gain nodes remain.

**"Bottom-up restriction"**: $T$ can only be pruned if it does not contain a subtree with lower error than $T$.

> The pruning set is useful precisely because it is **not** used to generate the tree — so it lets us test generalization capability when deciding whether to prune. **It is like a test set for the pruning decision.**

##### Worked example (REP) — schematized

**Three principles underlying post-pruning:**
1. The **class of a leaf** is decided by the **majority class in the TRAINING SET**.
2. The **error** is computed by counting misclassified elements **in the PRUNING SET**.
3. If a node $v$ roots a subtree $T$, it makes sense to prune $T$ (reducing to just leaf $v$) if the classification error on the pruning set **at $v$ is $\le$** the error at $T$.

**Case 1 — subtree rooted at V2** (children V4, V5):

*Error KEEPING the subtree* → $E(T_{V2}) = 3$:
- **V4**: label is **C1** (training majority: 6 vs 1). Pruning set at V4: 2×C1 and 1×C2. The C2 element is misclassified as C1 → **1 error**.
- **V5**: label is **C2** (training majority: 3 vs 1). Pruning set at V5: 2×C1 and 1×C2. The two C1 elements are misclassified → **2 errors**.
- **Total: 3 errors.**

*Error PRUNING the subtree* → $E(V2) = 2$:
- If V2 becomes a leaf, its label is decided by merging the **training sets** of V4 and V5: 7×C1 (6+1) and 4×C2 (1+3) → **C1 wins** → leaf V2 is labeled **C1**.
- Now merge the **pruning sets** of V4 and V5: 4×C1 and 2×C2 in total. Since the leaf says "everything is C1", the 2×C2 are misclassified → **2 errors**.

**Conclusion: error drops from 3 to 2 → PRUNE.**

**Case 2 — subtree rooted at V3** (children V6, V7):

*Error KEEPING the subtree* → $E(T_{V3}) = 1$:
- **V6**: label is **C2**. Pruning set: 0×C1 and 2×C2 → **0 errors**.
- **V7**: label is **C1**. Pruning set: 3×C1 and 1×C2 → the C2 element is misclassified → **1 error**.
- **Total: 1 error.**

*Error PRUNING the subtree* → $E(V3) = 3$:
- Merging the **training sets** of V6 and V7: 3×C1 and 4×C2 → **C2 wins** → leaf V3 labeled **C2**.
- Merging the **pruning sets**: 3×C1 and 3×C2. Since the leaf says "everything is C2", the 3×C1 are misclassified → **3 errors**.

**Conclusion: error would rise from 1 to 3 → DO NOT PRUNE, keep the subtree.**

#### Postpruning: Cost Complexity (CART)

Used in **CART**. Cost complexity is a function of the **number of leaves** in the tree and the **resubstitution error** of the tree:

$$\text{Cost complexity} = \text{Resubstitution Error} + \beta \cdot \text{Number of leaf nodes}$$

where the resubstitution error is the misclassification rate computed on the **training set**, and $\beta$ (the **complexity parameter**) is a **penalty per additional terminal node**.

> **Why we can't use the error alone**: otherwise we'd never prune. We need to balance the (normally increasing) error from pruning against the **reduction in complexity**. $\beta$ lets us weight complexity more or less in the cost.

**Procedure (search for the right-sized tree):**
- Prune or collapse branches of the largest tree **from the bottom up**, using the cost complexity parameter, and use cross-validation or an independent test sample to measure the predictive accuracy of the pruned tree.
- Use the resubstitution cost to **rank the subtrees**, generating a tree sequence table ordered from the most complex tree at the top to a less complex tree at the bottom.
- Identify the **minimum-cost tree** and pick as optimal the tree **within one standard error** of the minimum-cost tree — specifically the one with the **smallest number of terminal nodes** among those lying within one standard error.

#### Postpruning in C4.5: Pessimistic Pruning
Similar to cost complexity pruning **but does not use a pruning set**. It adjusts the error rates obtained on the training set by **adding a penalty**, computed with a heuristic approach based on statistical theory (hence "pessimistic"). If the error rate **in the node** is lower than the error rate **in the subtree** originating from that node, the subtree is pruned.

#### Remaining problems after pruning
Pruned trees tend to be more compact, but can still suffer from:
- **Repetition** — an attribute is repeatedly tested along a given branch.
- **Replication** — duplicate subtrees exist within the tree.

> **Why these happen** (lecture): with a **multiway** tree we don't have this problem; but when branches are **not mutually exclusive** (binary trees), the same attribute can be tested in different subtrees. This doesn't necessarily hurt results, but it **impacts explainability**. A multiway tree is often used to avoid these problems.

**Repetition and replication can impede both the accuracy and the comprehensibility** of a decision tree. Remedies:
- Use **multivariate splits** (splits based on a combination of attributes).
- Use a **different knowledge representation, such as rules**, instead of trees. Indeed, a rule-based classifier can be constructed by extracting IF-THEN rules from a decision tree (see §6).

### 4.12 Enhancements to Basic Decision Tree Induction

- **Allow for continuous-valued attributes**: dynamically define new discrete-valued attributes partitioning the continuous range into intervals. (In practice we normally discretize before learning — the "during learning" approaches are computationally expensive and no longer used.)
- **Handle missing attribute values**:
  - Assign the most common value of the attribute.
  - Assign a probability to each of the possible values.
- **Attribute construction**: create new attributes based on existing ones that are sparsely represented → reduces fragmentation, repetition, and replication, and improves interpretability.

### 4.13 Classification in Large Databases

**Scalability**: classifying datasets with millions of examples and hundreds of attributes with reasonable speed. There are parallel implementations of decision trees.

**Why decision tree induction remains popular:**
- relatively **faster learning speed** than other classification methods;
- **convertible to simple, easy-to-understand classification rules** — with a classical decision tree, during classification **only one path is activated**, so **only one rule fires**, making it very easy to explain why a given input produced a given output → highly interpretable;
- can use **SQL queries** for accessing databases;
- **comparable classification accuracy** with other methods. *(Honest caveat from the lecture: true in some application domains, not in general — other approaches can perform better; but in many applications we try to balance explainability against accuracy.)*

**Note**: if the training set does not fit in memory, decision tree construction becomes inefficient due to swapping training tuples in and out.

**Scalability frameworks** *(supporting detail from the slides):*
- **RainForest** (VLDB'98): uses special data structures called **AVC-sets**. An **AVC-set** (Attribute, Value, Class_label) of an attribute $X$ is the projection of the training dataset onto $X$ and the class label, with counts of individual class labels aggregated. An **AVC-group** of a node is the set of AVC-sets of all attributes at that node. Adapts to the amount of main memory available and applies to any decision tree induction algorithm.
- **BOAT** (Bootstrapped Optimistic Algorithm for Tree Construction): uses **bootstrapping** to create several smaller in-memory subsets, builds a tree per subset, then examines these trees to construct a new tree $T'$ that turns out to be very close to the tree that would have been generated from the whole dataset. Requires only **two scans** of the DB and is **incremental** (usable for incremental updates).

---

## 5. Bayes Classification Methods

A **statistical classifier**: performs **probabilistic prediction**, i.e., predicts **class membership probabilities**.

> **What's interesting**: the output is a probability associated with **each** class. We naturally select the class with the highest probability as the final output, but we can also exploit the whole probability distribution.

- **Foundation**: Bayes' Theorem.
- **Performance**: the simple **naïve Bayesian classifier** has comparable performance to decision trees and selected neural network classifiers depending on the domain. *(Normally performance is not very high, but it's used as a **benchmark** because it's very easy to implement.)*
- **Incremental**: each training example can incrementally increase/decrease the probability that a hypothesis is correct — prior knowledge can be combined with observed data.
  > **Contrast**: adding instances just changes probabilities without recomputing everything. With a decision tree, updating instances forces you to **rebuild the model**.
- **Standard**: even when Bayesian methods are computationally intractable, they provide a standard of **optimal decision making** against which other methods can be measured.

### 5.1 Bayesian Theorem: Basics

- Let $X$ be a data tuple — our **"evidence"** (class label unknown).
- Let $H$ be the **hypothesis** that $X$ belongs to class $C$.

**Classification consists of determining:**

- **$P(H|X)$ — posterior probability**: the probability that the hypothesis holds given the observed data sample $X$.
  - *Example*: if $H$ = "a customer will buy a computer", and a customer is characterized by age and income, then $P(H|X)$ = the probability that $X$ will buy a computer given his age and income (e.g., $X$ is a 35-year-old with income $40,000).
- **$P(H)$ — prior probability of $H$**: the initial probability. E.g., the probability that a *generic* customer buys a computer, regardless of age/income.
  > **Easy to estimate**: count how many training instances belong to class $C_i$, divide by the total number of instances.
- **$P(X)$ — prior probability of $X$**: probability that the sample data is observed (e.g., that a customer is 35 years old and earns $40,000).
- **$P(X|H)$ — likelihood** (posterior probability of $X$ conditioned on $H$): the probability of observing sample $X$ given that the hypothesis holds. E.g., given that $X$ will buy a computer, the probability that $X$ is aged 31–40 with medium income.

**Note**: $P(H)$, $P(X)$ and $P(X|H)$ can all be **estimated from the given data**; $P(H|X)$ is then computed via Bayes' theorem.

### 5.2 Bayes' Theorem

$$P(H|X) = \frac{P(X|H)\,P(H)}{P(X)}$$

Informally: **posterior = likelihood × prior / evidence**

Predict that $X$ belongs to $C_i$ **iff** the probability $P(C_i|X)$ is the highest among all $P(C_k|X)$ for all $k$ classes.

**Practical difficulty**: requires initial knowledge of many probabilities → significant computational cost.

> **The combinatorial explosion** (lecture): consider just two nominal attributes — $A$ with 3 values, $B$ with 4 values — and 2 classes $C_1, C_2$. To use the classifier we'd need $P(H|X)$ for **every possible input combination**. Even with only two small attributes, the number of combinations is already high. Rewriting it as $P(X|H)$ instead of $P(H|X)$ **doesn't help** — the total number of probabilities to estimate is the same.

Since **$P(X)$ is constant for all classes**, it can be dropped, and we only need to **maximize**:

$$P(C_i|X) \propto P(X|C_i)\,P(C_i)$$

where $P(C_i)$ is estimated by $|C_{i,D}|/|D|$.

### 5.3 Naïve Bayesian Classifier — the conditional independence assumption

**The trick to reduce computation**: assume the attributes are **conditionally independent** (no dependence relation between attributes):

$$P(X|C_i) = \prod_{k=1}^{n}P(x_k|C_i) = P(x_1|C_i)\times P(x_2|C_i)\times\dots\times P(x_n|C_i)$$

*Example*: $P((a_1,b_1)|C_1) = P(a_1|C_1)\cdot P(b_1|C_1)$.

> **The advantage**: we no longer need to compute all possible *combinations* — only the probability of each individual value given each class. **This greatly reduces the computation cost.** The price: we must assume independence of attributes, **which is normally not true**.

**Estimating $P(x_k|C_i)$:**
- If $A_k$ is **categorical**: the number of tuples in $C_i$ having value $x_k$ for $A_k$, divided by $|C_{i,D}|$ (the number of tuples of class $C_i$ in $D$).
- If $A_k$ is **continuous-valued**: usually computed based on a **Gaussian distribution** with mean $\mu$ and standard deviation $\sigma$:

$$g(x,\mu,\sigma) = \frac{1}{\sqrt{2\pi}\,\sigma}e^{-\frac{(x-\mu)^2}{2\sigma^2}}, \qquad P(x_k|C_i) = g(x_k, \mu_{C_i}, \sigma_{C_i})$$

where $\mu_{C_i}$ and $\sigma_{C_i}$ are the mean and standard deviation of the values of attribute $A_k$ for the training tuples of class $C_i$ (so we must estimate these **two parameters**).

### 5.4 Worked Example (full computation)

Classes: $C_1$: `buys_computer = yes`, $C_2$: `buys_computer = no`.
Data sample to classify: **X = (age = youth, income = medium, student = yes, credit_rating = fair)**

**Step 1 — Priors $P(C_i)$:**
$$P(\text{buys\_computer} = \text{yes}) = 9/14 = 0.643$$
$$P(\text{buys\_computer} = \text{no}) = 5/14 = 0.357$$

**Step 2 — Conditional probabilities $P(x_k|C_i)$** *(only the values we care about, not all combinations)*:

| | yes | no |
|---|---|---|
| P(age = "youth" \| C) | 2/9 = 0.222 | 3/5 = 0.6 |
| P(income = "medium" \| C) | 4/9 = 0.444 | 2/5 = 0.4 |
| P(student = "yes" \| C) | 6/9 = 0.667 | 1/5 = 0.2 |
| P(credit_rating = "fair" \| C) | 6/9 = 0.667 | 2/5 = 0.4 |

> *How e.g. $P(age=youth|yes)=2/9$ is obtained*: focus only on instances with class **yes** (9 of them), then count how many have `age = youth` → 2.

**Step 3 — Likelihood $P(X|C_i)$ via the independence product:**
$$P(X|\text{yes}) = 0.222 \times 0.444 \times 0.667 \times 0.667 = 0.044$$
$$P(X|\text{no}) = 0.6 \times 0.4 \times 0.2 \times 0.4 = 0.019$$

**Step 4 — Multiply by the priors:**
$$P(X|\text{yes})\times P(\text{yes}) = 0.028$$
$$P(X|\text{no})\times P(\text{no}) = 0.007$$

**→ Therefore X belongs to class `buys_computer = yes`.**

> Note: the two values don't sum to 1, because we dropped the $P(X)$ normalizer.

### 5.5 Avoiding the Zero-Probability Problem

Naïve Bayesian prediction requires **each conditional probability to be non-zero** — otherwise, since we take a **product**, if **even one** probability is 0 then the whole computed probability becomes **0**.

**Solution: Laplacian correction (Laplacian estimator)** — add 1 to each case.

*Example*: a dataset with 1000 tuples: `income=low` (0 tuples), `income=medium` (990), `income=high` (10):
- $Prob(income=low) = 1/1003 = 0.001$ *(very low, but not 0!)*
- $Prob(income=medium) = 991/1003 = 0.988$
- $Prob(income=high) = 11/1003 = 0.011$

The "corrected" estimates are **close to their uncorrected counterparts** — this very simple trick doesn't meaningfully change the distribution, but it solves the zero-probability problem.

### 5.6 Naïve Bayesian Classifier: Comments

**Advantages:**
- **Easy to implement** — just count instances and apply products/divisions.
- Good results obtained in most cases. *(Partially true — the results are decent but we have much better techniques today.)*

**Disadvantages:**
- **Assumption of class conditional independence** → **loss of accuracy**.
- Practically, **dependencies exist among variables**. E.g., hospitals: patients' Profile (age, family history, …), Symptoms (fever, cough, …), Disease (lung cancer, diabetes, …) — these dependencies **cannot be modeled** by a Naïve Bayesian Classifier.

**How to deal with dependencies? → Bayesian Belief Networks.**

### 5.7 Bayesian Belief Networks

**Bayesian belief networks** (a.k.a. Bayesian networks, probabilistic networks):
- allow the **representation of dependencies among subsets of attributes** (both discrete- and continuous-valued);
- defined by **two components**: (1) a **directed acyclic graph**, and (2) a set of **conditional probability tables (CPTs)**.

**The graph** is a (directed acyclic) graphical model of **causal relationships**:
- **Nodes** = random variables; **Links** = dependency.
- E.g., X and Y are the parents of Z, and Y is the parent of P → **no dependency between Z and P**.
- **Has no loops/cycles.**

**Nodes may correspond to actual attributes in the data OR to "hidden variables"** believed to form a relationship — e.g., in medical data a hidden variable may indicate a **syndrome**, representing a number of symptoms that together characterize a specific disease. So some nodes may be **fictitious attributes**.

**Conditional Probability Table (CPT)**: associated with each node; shows the conditional probability for **each possible combination of the values of its parents**.

*Example*: node `LungCancer (LC)` depends on `FamilyHistory (FH)` and `Smoker (S)` — all binary:

| | (FH, S) | (FH, ¬S) | (¬FH, S) | (¬FH, ¬S) |
|---|---|---|---|---|
| **LC** | 0.8 | 0.5 | 0.7 | 0.1 |
| **¬LC** | 0.2 | 0.5 | 0.3 | 0.9 |

So if FH=yes AND S=yes, then P(LC=yes) = 0.8 and P(LC=no) = 0.2.
> With all **actual** attributes, this table is easy to compute — just estimate from the training set. The situation is different with **fictitious/hidden** nodes, which require an optimization approach.

**Joint probability decomposition**: each variable is **conditionally independent of its non-descendants** in the network graph, **given its parents**. A complete representation of the joint probability distribution:

$$P(x_1,\dots,x_n) = \prod_{i=1}^{n}P\big(x_i \mid \text{Parents}(X_i)\big)$$

*Example with 5 variables A, B, C, D, E:*
- **Without** explicit dependencies (all variables assumed dependent on each other):
$$p(A,B,C,D,E) = p(A|B,C,D,E)\cdot p(B|C,D,E)\cdot p(C|D,E)\cdot p(D|E)\cdot p(E)$$
- **With** dependencies explicitly modeled (much cheaper):
$$p(A,B,C,D,E) = p(A|B)\cdot p(B|C,E)\cdot p(C|D)\cdot p(D)\cdot p(E)$$

**Other capabilities:**
- Any node within the network can be selected as an **"output" node** representing a class label attribute; there may be **more than one** output node.
- Rather than a single class label, classification can return a **probability distribution** over classes.
- Belief networks can answer **probability of evidence** queries (e.g., what is the probability that an individual has LungCancer, given both PositiveXRay and Dyspnea) and **most probable explanation** queries (e.g., which group of the population is most likely to have both PositiveXRay and Dyspnea).

**Training Bayesian Networks — four scenarios:**

| Scenario | Structure | Variables | Approach |
|---|---|---|---|
| **1** | known | all observable | Compute only the **CPT entries** (directly from the training set) → model complete |
| **2** | known | some hidden | **Gradient descent** (greedy hill-climbing): search along the steepest descent of a criterion function |
| **3** | unknown | all observable | Search through the **model space** to reconstruct network topology |
| **4** | unknown | all hidden | **No good algorithms known** for this purpose |

> **In practice, we exploit Scenario 1 and Scenario 2.**

*Scenario 2 detail*: let $w_{ijk}$ be a CPT entry for variable $Y_i = y_{ij}$ having parents $U_i = u_{ik}$, i.e. $w_{ijk} = P(Y_i = y_{ij} | U_i = u_{ik})$. Weights (CPT entries) are initialized to random probability values; at each iteration the method moves towards what appears to be the best solution at that moment (no backtracking), and weights converge to a **local optimum**. The algorithm: compute the gradients for each training tuple, take a small step in the gradient direction (with learning rate $l$), then **renormalize the weights** (they are probabilities: between 0 and 1, summing to 1 for all $i,k$).

---

## 6. Rule-Based Classification

The aim: build a classifier that is simply **a set of rules**.

### 6.1 Using IF-THEN Rules

Represent knowledge in the form of IF-THEN rules:

```
R: IF age = youth AND student = yes THEN buys_computer = yes
```

- Before the THEN → the **rule antecedent / precondition**.
- After the THEN → the **rule consequent**.

**How it classifies**: when an unlabeled instance arrives, the combination of its values activates (hopefully) exactly one rule; the **fired rule predicts the class**.

**Assessment of a rule — coverage and accuracy:**

$$coverage(R) = \frac{n_{covers}}{|D|}, \qquad accuracy(R) = \frac{n_{correct}}{n_{covers}}$$

where $n_{covers}$ = number of tuples covered by rule R (i.e., all instances that **fire** the rule — in the example, all instances with age=youth AND student=yes), $n_{correct}$ = number of tuples **correctly classified** by R, and $D$ is the training data set.

> Note: to be correctly classified, an instance must first *cover* the rule — but not all covered instances are correctly classified. So normally $n_{correct} < n_{covers}$. **A good rule has good coverage, but of course we also want high accuracy.**

### 6.2 The Conflict Problem

If **more than one rule is triggered** by one instance, we have a **conflict** and need conflict resolution.

> If two triggered rules produce the **same** class → no problem. But if they produce **different** classes → we cannot decide the class for the tuple.

**Possible conflict-resolution strategies:**
- **Size ordering**: assign the highest priority to the triggering rule with the **"toughest" requirement** (i.e., with the most attribute tests — largest rule antecedent size).
- **Class-based ordering**: decreasing order of **prevalence** (rules for the most frequent class come first) **or** by **misclassification cost per class** (rules for the class with the highest cost come first).
- **Rule-based ordering (decision list)**: rules are organized into **one long priority list**, ordered by some measure of rule quality (accuracy, coverage, size) or by advice from domain experts. Each rule in a decision list **implies the negation of the rules that come before it** in the list → **difficult to interpret**.

The final aim: examine which rules fired, and assign the class from the rule with the **highest priority**.

### 6.3 Rule Extraction from a Decision Tree

The easiest way to generate a rule-based classifier — a decision tree can be transformed into a set of rules.

- Rules are **easier to understand** than large trees.
- **One rule is created for each path from the root to a leaf** → as many rules as leaves.
- Each attribute-value pair along a path forms a **conjunction**; the leaf holds the class prediction.
- Rules are **mutually exclusive** (no rule conflict) and **exhaustive** (one rule for each possible attribute-value combination).
- **Note: one rule per leaf!** With decision trees that suffer from **repetition and replication**, the extracted rule base can be difficult to interpret.

**Example — rules extracted from the `buys_computer` decision tree:**
```
IF age = youth AND student = no        THEN buys_computer = no
IF age = youth AND student = yes       THEN buys_computer = yes
IF age = middle_aged                   THEN buys_computer = yes
IF age = senior AND credit_rating = excellent THEN buys_computer = no
IF age = senior AND credit_rating = fair      THEN buys_computer = yes
```

**Pruning the rule set:**
- Each condition that does **not improve the estimated accuracy** of the rule can be pruned.
- **C4.5 uses a pessimistic approach** to counteract the bias generated by using the training set.
- Further, **any rule that does not contribute to the overall accuracy is pruned**.
- **⚠️ After pruning, the rules are no longer mutually exclusive and exhaustive** → we again need conflict-resolution solutions:
  - **C4.5 adopts a class-based ordering scheme**: groups the rules per class, then ranks these class rule sets so as to **minimize the number of false-positive errors** (the rule predicts class C but the actual class is not C). The class rule set with the **least number of false positives is examined first**.
  - **Default class**: the class containing the highest number of tuples **not covered by any rule** (the majority class will likely already have many rules for its tuples).

### 6.4 Rule Induction: the Sequential Covering Method

Can we generate rules **directly from the data**, without a decision tree? Yes — heuristic approaches following the **sequential covering** philosophy.

**Typical sequential covering algorithms: FOIL, AQ, CN2, RIPPER.**

**The idea**: rules are learned **sequentially**, focusing on one class at a time. Each rule for a given class $C_i$ should cover **many tuples of $C_i$ but none (or few) of the tuples of other classes**. *(Not an easy task, since there's usually some confusion between points in the training set.)*

**Steps:**
- Rules are learned **one at a time**.
- Each time a rule is learned, the **tuples covered by that rule are removed**.
- The process repeats on the remaining tuples until a **termination condition** — e.g., no more training examples, or the quality of the returned rule falls below a user-specified threshold.

> **Comparison with decision-tree induction**: decision-tree induction learns a set of rules **simultaneously**; sequential covering learns them **one at a time**.

**High-level algorithm:**
```
while (enough target tuples left)
    generate a rule
    remove positive target tuples satisfying this rule
end
```

**Formal pseudocode** — *note: no parameters are needed!*
```
Algorithm: Sequential covering. Learn a set of IF-THEN rules for classification.
Input:  D, a data set of class-labeled tuples;
        Att_vals, the set of all attributes and their possible values.
Output: A set of IF-THEN rules.

(1) Rule_set = {};                                  // initial set of learned rules is empty
(2) for each class c do                             // consider a single class at a time
(3)     repeat
(4)         Rule = Learn_One_Rule(D, Att_vals, c);
(5)         remove tuples covered by Rule from D;
(6)         Rule_set = Rule_set + Rule;             // add new rule to rule set
(7)     until terminating condition;
(8) endfor
(9) return Rule_set;
```

### 6.5 How to Learn-One-Rule? (Greedy depth-first strategy)

**Rule generation loop:**
```
while(true)
    find the best predicate p
    if foil-gain(p) > threshold then add p to current rule
    else break
end
```

**The strategy:**
- **Start with the most general rule possible**: condition = **empty** (a rule with no antecedent → fired by *every* tuple in the training set).
- Then **test all possible one-condition antecedents** — all possible values, for all possible attributes, as a single condition. A **metric** is needed to decide which single condition is best.
- Once the best single condition is fixed, try to improve the rule by **adding a second condition**, repeating the evaluation with the metric.
- Define a **termination condition** for when to stop adding conditions.

Once a rule is generated, remove all instances covered by it and start generating other rules on the remaining instances.

> **The intuition** (lecture): we begin with a simple predicate that most training instances fire — covering both positive and negative examples. Each refinement (extra condition) **narrows** the set of firing instances. The goal is to end up with an antecedent that **maximizes positive examples while minimizing negative ones** — ideally classifying only instances that are positive for the class in the consequent.

### 6.6 Rule-Quality Measure: FOIL_Gain

**Why accuracy alone is not enough:**
Consider two rules: R1 with 95% accuracy but **large coverage**, R2 with 100% accuracy but covering **only two instances**. If we only consider accuracy we'd pick R2; if we also consider coverage we'd pick R1. **We need a metric taking both into account.**

**FOIL_Gain** (used in FOIL & RIPPER; FOIL = First Order Inductive Learner) — assesses information gain from extending a condition:

$$FOIL\_Gain = pos' \times \left(\log_2\frac{pos'}{pos'+neg'} - \log_2\frac{pos}{pos+neg}\right)$$

where $pos$ and $neg$ are the numbers of positive and negative tuples covered by the **current** rule $R$, and $pos'$, $neg'$ are those covered by the **new** (extended) rule $R'$.

> **Reading the formula**: the **accuracy** aspect is implemented by the **ratio** inside the logs; the **coverage** aspect is given by the **multiplying factor $pos'$** in front. **Foil-gain favors rules that have high accuracy AND cover many positive tuples.**

**How it's applied**: start with e.g. `IF ___ THEN loan_decision = accept` (covering every instance); measure $pos$ and $neg$. Then test all possible single conditions in the antecedent, measuring $pos'$ and $neg'$ for each, apply the formula, and **select the condition with the highest FOIL_Gain**. Keep adding conditions as long as FOIL_Gain increases.

### 6.7 Rule Pruning: FOIL_Prune

**Learn_One_Rule does not employ a test set when evaluating rules → the evaluation is optimistic.** So we prune based on an **independent set of test tuples**. Pruning is carried out by **removing a conjunct**.

FOIL uses a simple yet effective method:

$$FOIL\_Prune(R) = \frac{pos - neg}{pos + neg}$$

**If FOIL_Prune is higher for the pruned version of R, prune R.**

- By convention, **RIPPER starts with the most recently added conjunct** when considering pruning.
- Conjuncts are pruned **one at a time**, as long as this results in an improvement.

---

## 7. Classification by Using Frequent Patterns (Associative Classification)

> ⚠️ **The lecture notes mark this section as skippable** — the professor deferred it because frequent patterns hadn't been covered yet. Included here for completeness.

**Associative classification — major steps:**
- Mine data to find **strong associations** between frequent patterns (conjunctions of attribute-value pairs) and class labels.
- Association rules are generated in the form $p_1 \wedge p_2 \wedge \dots \wedge p_l \Rightarrow \text{"}A_{class} = C\text{"}$ (with confidence, support).
- Organize the rules to form a rule-based classifier.

**Why effective?** It explores highly confident associations among **multiple** attributes, overcoming some constraints of decision-tree induction, which considers **only one attribute at a time**. Associative classification is often more accurate than some traditional methods such as C4.5.

**Typical methods:**

| Method | Key ideas |
|---|---|
| **CBA** (Liu, Hsu & Ma, KDD'98) | Mines association rules of form *Cond-set ⇒ class label*. Rules organized by **decreasing precedence based on confidence then support**. If a set of rules shares the same antecedent, the one with the highest confidence represents the set. To classify: the **first rule satisfying the tuple** is used; the classifier also contains a **default rule** with the lowest precedence. |
| **CMAR** (Li, Han, Pei, ICDM'01) | Uses a variant of **FP-growth** to find the complete rule set satisfying min-confidence and min-support. **Rule pruning** on insertion: given $R_1, R_2$, if $R_1$'s antecedent is more general than $R_2$'s and $conf(R_1)\ge conf(R_2)$, then $R_2$ is pruned; also prunes rules where antecedent and class are not positively correlated (via $\chi^2$ test). To classify inconsistent matches: rules grouped by class label, a **weighted $\chi^2$ measure** finds the strongest group, and the tuple gets that group's label. Slightly higher average accuracy and more efficient memory use than CBA. |
| **CPAR** (Yin & Han, SDM'03) | Generates predictive rules using a **FOIL-like** rule-generation algorithm rather than frequent itemset mining. Unlike FOIL, covered tuples **remain under consideration but with reduced weight**. To classify: rules grouped by class label; the **best k rules** (by expected accuracy) of each group predict the class. High efficiency, accuracy similar to CMAR. |

---

## 8. Lazy Learners

### 8.1 Lazy vs Eager Learning

| | **Lazy learning** (instance-based) | **Eager learning** (all methods above) |
|---|---|---|
| Model | **No model generated** — simply stores training data (or does minor processing) and waits for a test tuple | Constructs a classification model **before** receiving new data |
| Time | **Less time training, more time predicting** | More time training, less time predicting |
| Hypothesis space | **Richer** — uses many **local** linear functions forming an implicit global approximation to the target function | Must commit to a **single hypothesis** covering the entire instance space |

> **Practical note** (lecture): we usually prefer to spend more time in **training** rather than in classification, because **classification time is perceived by the user** and makes interaction slower.

**Typical lazy approaches:**
- **k-nearest neighbor**: instances represented as points in a Euclidean space.
- **Locally weighted regression**: constructs local approximations.
- **Case-based reasoning**: uses symbolic representations and knowledge-based inference.

### 8.2 The k-Nearest Neighbor (k-NN) Algorithm

- All instances correspond to points in the $n$-D space.
- Nearest neighbors are defined in terms of **Euclidean distance** $dist(X_1, X_2)$.
- The target function can be **discrete- or real-valued**. For discrete-valued, k-NN returns the **most common value** among the $k$ training examples nearest to $x_q$.
- **Voronoi diagram**: the decision surface induced by 1-NN for a typical set of training examples — lines between instances representing the 1-NN decision boundaries.

**The idea**: to classify an unlabeled instance, consider the distance between it and its nearest neighbour(s) in the training set, and assign the majority class of the $k$ nearest instances.

**Why not just 1-NN?** 1-NN works quite well if the instances of different classes are **well separated**. But if there is an **isolated instance** of one class surrounded by another class (noise), the unlabeled instance gets misclassified. **Solution: consider a higher number of neighbours $k$.**

**Complexity:**
- Computing the Euclidean distance between the unlabeled instance and all others → $O(N)$.
- Sorting the distances to find the lowest → minimum $O(\log N)$.
- **Optimization**: store the minimum distance(s) while computing all distances → complexity of 1-NN is $O(N)$. For general $k$ it remains **$O(N)$** because we just need $k$ variables to store the $k$ lowest distances.
- Other efficiency techniques (from the slides): by **sorting and arranging tuples into search trees**, comparisons can be reduced to $O(\log|D|)$; **parallel implementations**; **partial distance** (use only a subset of the $n$ attributes — if the partial distance already exceeds a threshold, stop computing).

**Practical questions:**
- **Distance for non-numeric (categorical) attributes**: compare the corresponding values — if **identical, the difference is 0**; otherwise **1**.
- **Missing values**: assume the **maximum possible difference** — which is **1** for nominal attributes, and for numeric attributes when the value is missing in **both** tuples.
- **Choosing k**: **experimentally** — use increasing values of $k$, evaluate accuracy, and choose the $k$ giving the best result (with the lowest complexity). Generally, **the larger the number of training instances, the larger the value of $k$**.
- **Choice of distance**: different distances can be used to incorporate attribute weighting and pruning of noisy tuples.

**Advantages/disadvantages**: no training phase at all, but **all the complexity ($O(N)$) moves to classification**; and we need to **keep the entire training set** to classify new instances → large memory requirement. → **Editing methods** address this.

### 8.3 Editing Methods (removing useless / error-inducing training tuples)

- **Wilson editing**: cleans **interclass overlap regions**, leading to **smoother boundaries** between classes.
  - Mechanism: classify each object $o_i$ using the $k$-nearest neighbours; then **eliminate all instances in the training set that are misclassified** by k-NN (these are probably noise).
  - *Illustration*: if a white instance is misclassified as black and a black one as white, Wilson editing removes those 2 instances — they were probably noise.
- **Multi-edit**: **repeatedly applies Wilson editing** to $N$ random subsets of the original dataset **until no more examples are removed** (using one subset as training to classify another subset, like a validation set).
- **Citation Editing**: exploits an analogy — if a paper **cites** another article, the paper is related to that article; similarly, if a paper **is cited by** an article, it is also related. Thus **both citers and references** are considered related to a given paper → the method considers **both directions** of the relation.
- **Supervised Clustering**: replaces the object set $O$ by a subset $O_r$ consisting of **cluster representatives** selected by a supervised clustering algorithm.

**Comparing the methods** — using the **training set compression rate**:

$$\text{compression rate} = \left(1 - \frac{r}{n}\right)\cdot 100$$

where $r$ = number of instances **after** editing, $n$ = number of instances in the **original** training set.

**Findings:**
- **Wilson**: classification accuracy gets better, but **compression is rather low**.
- **SC (Supervised Clustering) editing**: accuracy improves for the UCI dataset but **gets worse** for the 2D dataset, while the **compression rate is very high**.
- **In summary**: we need a good **trade-off between accuracy and reduction** — reducing the number of instances too much can make accuracy worse.

### 8.4 k-NN for Prediction (real-valued targets)

- For a given unknown tuple, return the **mean value** of the $k$ nearest neighbors.
- **Distance-weighted nearest neighbor algorithm**: weight the contribution of each of the $k$ neighbors according to their distance to the query $x_q$, giving **greater weight to closer neighbors**:

$$w \equiv \frac{1}{d(x_q, x_i)^2}$$

- **Robust to noisy data** by averaging over the $k$ nearest neighbors.
- **Curse of dimensionality**: the distance between neighbors could be **dominated by irrelevant attributes** → to overcome it, **eliminate the least relevant attributes**.

### 8.5 Case-Based Reasoning (CBR)

**CBR** is the process of solving new problems based on the solutions of **equal or similar past problems**.

- **Core assumption: similar problems have similar solutions.**
- Uses a **database of problem solutions**, storing **symbolic descriptions** (tuples or cases) — **not** points in a Euclidean space.
- We must implement a **similarity measure** suitable to the case being analyzed, then adapt the differences.

**Everyday examples:**
- A **lawyer** who advocates a particular trial outcome based on legal precedents, or a **judge** who creates case law.
- An **engineer** copying working elements of nature (biomimicry) — treating nature as a database of solutions to problems.

**Methodology:**
- Cases represented by **rich symbolic descriptions** (e.g., function graphs).
- Search for similar cases; **multiple retrieved cases may be combined**.
- When a new successful solution is found, a **new experience** is made and can be stored in the case base to increase its competence → **implementing a learning behavior**.
- **Tight coupling** between case retrieval, knowledge-based reasoning, and problem solving.

**The CBR cycle** (Aamodt and Plaza) — **4 sequential steps** (the "4 R's"):

1. **RETRIEVE**
   - Introduce a **similarity measure**.
   - One or several cases from the case base are selected based on the modeled similarity.
   - The retrieval task = finding a **small number of cases** from the case-base with the **highest similarity** to the query.
   - This is essentially a **k-nearest-neighbor retrieval task** with a specific similarity function.
   - As the case base grows, retrieval efficiency **decreases** → use methods that improve efficiency, e.g. **kd-trees**, case-retrieval nets, or discrimination networks.

2. **REUSE**
   - Reusing a retrieved solution can be simple if the solution is returned **unchanged** as the proposed solution.
   - Otherwise, **adaptation** via **transformational adaptation** or **generative adaptation** (e.g., for synthetic tasks).
   - Most practical CBR applications today try to **avoid extensive adaptation** for pragmatic reasons.

3. **REVISE**
   - **Feedback** related to the solution constructed so far is obtained.
   - Feedback can be a **correctness rating** of the result, or a **manually corrected revised case**.
   - The revised case (or any other feedback) enters the CBR system for use in the subsequent retain phase.

4. **RETAIN**
   - The **learning phase** of a CBR system (adding a revised case to the case base).
   - Explicit **competence models** have been developed to enable **selective retention** of cases (because the case base grows continuously).

**Application areas**: help-desk and customer service; recommender systems in e-commerce; knowledge and experience management; medical applications and image processing; law, technical diagnosis, design, planning; computer games and music.

**Challenges**: (1) find a good **similarity metric**; (2) **indexing** based on syntactic similarity measure, and when it fails, backtracking and adapting to additional cases (as the database grows, retrieval takes more time).

---

## 9. Model Evaluation and Selection ⭐

> **Flagged in the lecture notes as VERY IMPORTANT for this course.**

### 9.1 Why accuracy alone is not enough

**Accuracy** = number of correctly classified instances / total instances in the **test set**. (Whenever we talk about accuracy, we're considering the test set.)

**The class-imbalance problem:**
> If we have classes C1 (95% of instances) and C2 (5%), and we build a classifier that **always** assigns class C1, it would achieve **95% accuracy** — but this is obviously a terrible solution.

**Conclusion**: the desired level of accuracy **depends on the distribution of the data**. Performance should be good on **both** the majority and minority class, but accuracy gives us no measure of this → we need a more detailed metric.

One class may be rare (e.g., fraud, HIV-positive, cancer): significant majority of the negative class and minority of the positive class.

### 9.2 Confusion Matrix

Rows = **actual** class (in the test set), columns = **predicted** class.

| Actual \ Predicted | $C_1$ | $\neg C_1$ | Total |
|---|---|---|---|
| **$C_1$** | True Positives (TP) | False Negatives (FN) | P |
| **$\neg C_1$** | False Positives (FP) | True Negatives (TN) | N |
| **Total** | P' | N' | All |

Given $m$ classes, an entry $CM_{i,j}$ indicates the **number of tuples in class $i$ that were labeled by the classifier as class $j$**. **In the ideal case the confusion matrix is completely diagonal.**

*Example:*

| Actual \ Predicted | buy_computer = yes | buy_computer = no | Total |
|---|---|---|---|
| buy_computer = yes | 6954 | 46 | 7000 |
| buy_computer = no | 412 | 2588 | 3000 |
| Total | 7366 | 2634 | 10000 |

### 9.3 Metrics derived from the Confusion Matrix

**Accuracy / error rate:**
$$Accuracy = \frac{TP+TN}{All}, \qquad \text{Error rate} = 1 - Accuracy = \frac{FP+FN}{All}$$

**Sensitivity and Specificity:**
$$Sensitivity = \frac{TP}{P} \quad \text{(true positive recognition rate)}, \qquad Specificity = \frac{TN}{N} \quad \text{(true negative recognition rate)}$$
> These let us easily recognize if one of the classes is being classified in a completely wrong way (in the "always predict C1" example above, specificity would be **0**).

**Precision and Recall:**
$$Precision = \frac{TP}{TP+FP} \quad \text{(exactness)}, \qquad Recall = \frac{TP}{TP+FN} = \frac{TP}{P} \quad \text{(completeness)}$$
- **Precision** — of the tuples the classifier **labeled positive**, what % are actually positive?
- **Recall** — of the **actually positive** tuples, what % did the classifier label positive?
- **Perfect precision (1.0)**: every tuple labeled as class C does indeed belong to C — **but it says nothing about how many class-C tuples the classifier mislabeled**.
- **Perfect recall (1.0)**: every item from class C was labeled as such — **but it says nothing about how many other tuples were incorrectly labeled as C**.
- **Conclusion: these metrics must be used in pairs** — one alone doesn't give enough information.
- There is an **inverse relationship** between precision and recall: a medical classifier may achieve high precision by labeling only very obvious cancer cases as cancer, but have low recall by missing many other cancer cases.

**F-measure (F₁ / F-score)** — combining precision and recall into a **single index** (harmonic mean):

$$F_1 = \frac{2 \times Precision \times Recall}{Precision + Recall}$$

**The higher F, the better the classifier.**

**$F_\beta$ — weighted measure** (assigns $\beta$ times as much weight to recall as to precision):

$$F_\beta = \frac{(1+\beta^2)\times Precision \times Recall}{\beta^2 \times Precision + Recall}$$

Commonly used: $F_2$ (weights **recall** more than precision) and $F_{0.5}$ (weights **precision** more than recall).

> **$F_1$ takes into account possible imbalances in the classification, so it is a better measure than accuracy.**

#### Worked example (cancer detection — imbalanced)

| Actual \ Predicted | cancer = yes | cancer = no | Total | Recognition (%) |
|---|---|---|---|---|
| cancer = yes | 90 | 210 | 300 | **30.00 (sensitivity)** |
| cancer = no | 140 | 9560 | 9700 | 98.56 (specificity) |
| Total | 230 | 9770 | 10000 | **96.40 (accuracy)** |

- **Accuracy = 96.40%** → looks rather high!
- But the **minority class** (which is also the **most important** to identify — failing to detect cancer carries significant risk) performs terribly:
  - $Precision = 90/230 = \mathbf{39.13\%}$
  - $Recall = 90/300 = \mathbf{30.00\%}$

> **Convention**: as a standard, we take the **minority class as the positive class**.

#### Multi-class problems
Compute $TP_i$, $TN_i$, $FP_i$, $FN_i$ taking a reference class $C_i$ as **positive** and considering **all the other classes as negative**. Consequently precision, recall, etc. are defined **per class**; metrics for the whole classifier are obtained by **averaging** the per-class metrics.

### 9.4 Evaluating Classifier Accuracy: resampling methods

**The problem**: a **single execution** (one training set / one test set) is **not enough** to evaluate a classifier's performance — by chance one classifier could be simply very well-suited to one particular train/test split, giving unreliably high results. We need **different trials**, but we only have one labeled dataset.

**a) Holdout method**
- Data randomly partitioned into two independent sets: **training set (e.g. 2/3)** for model construction, **test set (e.g. 1/3)** for accuracy estimation.
- **Random subsampling**: a variation — repeat holdout $k$ times; accuracy = average of the accuracies obtained.
- Repeating it gives us a **distribution of accuracy values** for each classifier, which we can compare.

> **⚠️ Is the average a good measure?** Consider two Gaussian distributions: the red one *appears* to have a higher average than the black one, so we'd take it as best. But if the distributions are **heavily overlapping**, we must **statistically demonstrate** that they are two genuinely distinct distributions → **hence the need for a statistical test** (§9.5).

**b) Cross-validation (k-fold, k = 10 most popular)** — *the most popular method*
- Randomly partition the data into $k$ **mutually exclusive subsets ("folds")**, each of approximately equal size.
- At the $i$-th iteration, use $D_i$ as the **test set** and the others as the **training set**.
- At the end we again obtain a **distribution of accuracy measures**.
- **Leave-one-out**: $k$ folds where $k$ = number of tuples — only one sample is left out at a time for the test set. Suitable for **small** datasets.
- **Stratified cross-validation**: folds are **stratified** so the class distribution in each fold is approximately the same as in the initial data. *(Recommended for accuracy estimation.)*

**c) Bootstrap**
- Samples the given training tuples **uniformly with replacement** — each time a tuple is selected it is equally likely to be selected again and re-added to the training set.
- **Works well with small datasets.**
- Commonly used variant: **.632 bootstrap**:
  - Sample a dataset of $d$ tuples $d$ times **with replacement**, producing a training set of $d$ samples. The tuples that **didn't** make it into the training set form the **test set**.
  - About **63.2%** of the original data ends up in the bootstrap sample, and the remaining **36.8%** forms the test set — since the probability of *not* being chosen is $(1-1/d)^d \approx e^{-1} = 0.368$ for large $d$.
  - Repeat the sampling procedure $k$ times. The overall accuracy is a **weighted sum** of the accuracy obtained on the test set and on the training set:

$$Acc(M) = \sum_{i=1}^{k}\big(0.632\times Acc(M_i)_{test\_set} + 0.368\times Acc(M_i)_{train\_set}\big)$$

### 9.5 Statistical tests: comparing classifier models $M_1$ vs $M_2$

Suppose we have 2 classifiers $M_1$, $M_2$ — which is better?
- Use **10-fold cross-validation** to obtain $err(M_1)$ and $err(M_2)$.
- These mean error rates are just **estimates** of the error on the true population of future data.
- **What if the difference between the 2 error rates is just attributed to chance?**
  - → Use a **test of statistical significance**.
  - → Obtain **confidence limits** for our error estimates: e.g., "one model is better than the other by a margin of error of ±4%."

> **Note**: 10 folds is the **minimum** number for the statistical evidence to be relevant. And everything said about accuracy applies identically to the **$F_1$ measure** or **AUC**.

#### a) Parametric test: the t-test (Student's t-test)

**Two assumptions:**
1. Classification accuracies are **normally distributed** within each group.
2. The **variances** of the two populations are **not reliably different**.

**Null Hypothesis $H_0$**: the two distributions of accuracy for $M_1$ and $M_2$ are **the same**. If we can **reject** $H_0$, we conclude the difference between $M_1$ and $M_2$ is **statistically significant** → choose the model with the lower error rate.

**Pairwise comparison**: the **same test set must be used** for $M_1$ and $M_2$. For the $i$-th round of 10-fold cross-validation, the **same cross-partitioning** is used to obtain $err(M_1)_i$ and $err(M_2)_i$. Average over 10 rounds to get $err(M_1)$ and $err(M_2)$.

The t-test computes the **t-statistic with $k-1$ degrees of freedom**:

$$t = \frac{\overline{err(M_1)} - \overline{err(M_2)}}{\sqrt{var(M_1-M_2)/k}}$$

**Pipeline:**
1. Compute $t$. Select a **significance level** (e.g. sig = 5%).
2. Consult the **t-distribution table**: find the $t$ value corresponding to $k-1$ degrees of freedom (here, **9**).
3. The t-distribution is **symmetric**, so look up the value for confidence limit $z = sig/2$ (here, **0.025**).
4. **If $t > z$ or $t < -z$**, the $t$ value lies in the **rejection region**:
   - → Reject $H_0$ → **statistically significant difference** between $M_1$ and $M_2$.
   - Otherwise → conclude that any difference is due to **chance**.
- **If two test sets are available**: use the **non-paired t-test**; the number of degrees of freedom is taken as the **minimum** of the degrees of freedom of the two models.

> **On degrees of freedom** (lecture): DoF expresses **how many samples in the distribution you have to fix in order to automatically fix the remaining one**. For an average, knowing $k-1$ samples lets you derive the last one — hence $DoF = k-1$. The **confidence value** expresses the probability that $H_0$ is true; if it's low, it's highly unlikely $H_0$ holds. If the computed $t$ exceeds the table $t$ (for the corresponding DoF and confidence), then our probability is lower than the one fixed by the confidence value, so we can reject $H_0$.

**Summary of the procedure**: generate multiple experiments for both classifiers → obtain two distributions → **check the distributions are Gaussian** (necessary assumption) → apply the t-test to verify they are not the same distribution.

#### b) Non-parametric test: the Wilcoxon signed rank sum test

An example of a **non-parametric / distribution-free** test — it needs **no assumption about the kind of distribution** of the F-measure or accuracy rate.

**Null hypothesis**: the **medians** of the two samples are equal (we test whether two populations have the same distribution with the same median).
> In the t-test we assumed Gaussian distributions so we could talk about **means**; here, assuming nothing about the distribution, we talk about **medians**.

**Used as**: a non-parametric alternative to the one-sample t-test or paired t-test; or for ordered (ranked) categorical variables without a numerical scale.

**Assumptions** (same as t-test **except** normality):
1. The two samples are **independent** of one another.
2. The two populations have **equal variance or spread**.

**Procedure (paired data):**
1. Calculate each **paired difference** $d_i = x_i - y_i$, where $x_i, y_i$ are the pairs of observations. (For each fold: the difference between the accuracy/error of the first and second classifier.)
2. **Rank** the differences $d_i$ **ignoring the signs** (rank 1 to the smallest $|d_i|$, rank 2 to the next, etc.) — i.e. sort by absolute value.
3. **Label each rank with its sign**, according to the sign of $d_i$.
4. Calculate **$W^+$** = sum of the ranks of the **positive** differences, and **$W^-$** = sum of the ranks of the **negative** differences. *(Check: $W^+ + W^- = \frac{n(n+1)}{2}$, where $n$ is the number of pairs.)*
5. Choose $W = \min(W^+, W^-)$.
6. Use tables of critical values for the Wilcoxon signed rank sum test to find the probability of observing a value of $W$ **or more extreme**. Most tables give both one-sided and two-sided p-values; if not, **double the one-sided p-value** to obtain the two-sided p-value.

> **Under $H_0$** we'd expect the distribution of the differences to be approximately **symmetric around zero**, with positives and negatives distributed at random among the ranks.

**Normal approximation**: if $\frac{n(n+1)}{2}$ is large enough (**> 20**), a normal approximation can be used with:

$$\mu_W = \frac{n(n+1)}{4}, \qquad \sigma_W = \sqrt{\frac{n(n+1)(2n+1)}{24}}$$

**Dealing with ties:**
- Observations exactly equal to the median value (i.e. 0 in the case of paired differences): **ignore** such observations and **adjust $n$** accordingly.
- Two or more observations/differences equal: **average the ranks** across the tied observations, and **reduce the variance by $\frac{t^3-t}{48}$** for each group of $t$ tied ranks.

##### Worked example (12-fold cross-validation)

| Fold | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **M1** | 2.0 | 3.6 | 2.6 | 2.6 | 7.3 | 3.4 | 14.9 | 6.6 | 2.3 | 2.0 | 6.8 | 8.5 |
| **M2** | 3.5 | 5.7 | 2.9 | 2.4 | 9.9 | 3.3 | 16.7 | 6.0 | 3.8 | 4.0 | 9.1 | 20.9 |
| **Diff** | +1.5 | +2.1 | +0.3 | −0.2 | +2.6 | −0.1 | +1.8 | −0.6 | +1.5 | +2.0 | +2.3 | +12.4 |

**Ranking the differences (by absolute value):**

| Diff | 0.1 | 0.2 | 0.3 | 0.6 | 1.5 | 1.5 | 1.8 | 2.0 | 2.1 | 2.3 | 2.6 | 12.4 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **Rank** | 1 | 2 | 3 | 4 | **5.5** | **5.5** | 7 | 8 | 9 | 10 | 11 | 12 |
| **Sign** | − | − | + | − | + | + | + | + | + | + | + | + |

> Note the **tie**: two values equal to 1.5 → they'd occupy ranks 5 and 6, so both get the average rank **5.5**, and we must adjust the variance.

**Computing $W^+$ and $W^-$:**
$$W^- = 1 + 2 + 4 = 7$$
$$W^+ = 3 + 5.5 + 5.5 + 7 + 8 + 9 + 10 + 11 + 12 = 71$$

**Check the normal approximation applies:** $\frac{n(n+1)}{2} = \frac{12\cdot13}{2} = 78 > 20$ ✓

$$W = \min(W^-, W^+) = 7$$

**Variance adjustment for the tie** (one group of $t=2$ tied ranks): reduce by $\frac{2^3-2}{48} = \frac{6}{48} = 0.125$.

**Compute the z-score** (a measure of how many standard deviations below/above the population mean a raw score is):

$$z = \frac{W - \mu_W}{\sigma_W} = \frac{7 - \frac{12\cdot13}{4}}{\sqrt{\frac{12\cdot13\cdot25}{24} - 0.125}} = \frac{7-39}{\sqrt{162.5-0.125}} = 2.511$$

**Look up this score in the z-table** → area of 0.9880 → **two-tailed p-value of 0.012**. This is a **tiny p-value**, a strong indication that the **medians are significantly different**.

> This way we compared two classifiers **without assuming** the distributions are normal.

### 9.6 Model Selection: ROC Curves

If the dataset is imbalanced we use the F-measure instead of accuracy; a **third alternative is the area under the ROC curve (AUC)**.

**The ROC plane:**
- **Vertical axis**: **True Positive Rate**, $TPR = TP/P$
- **Horizontal axis**: **False Positive Rate**, $FPR = FP/N$
- Both axes go from 0 to 1.
- One **point** on this plot represents the performance of a classifier on a specific test set (one execution → one TPR and one FPR).

**Optimal classifier**: $TPR = 1$ (all real positives classified as positive) and $FPR = 0$ (no false positives). **The closer (FPR, TPR) is to (0,1), the better.**

**ROC** = **Receiver Operating Characteristics** — the name comes from **signal detection theory** (communication).

- ROC curves are useful for **visual comparison** of classification models, showing the **trade-off** between TPR and FPR.
- Every time we change the classifier's **parameters** we get a new point, until we get the complete **curve**.
- The plot also shows a **diagonal line**: the closer the curve is to the diagonal (i.e., the closer the area is to **0.5**), the **less accurate** the model. A model with **perfect accuracy has an area of 1.0**.
- Ranking method: rank the test tuples in **decreasing order** — the one most likely to belong to the positive class appears at the top of the list.

**Plotting a ROC curve for a probabilistic classifier**: if the probability is higher than a **threshold**, the test instance is classified as positive. Start by taking the **highest probability as the threshold** and then decrease it — the true positives (but also the false positives) gradually increase. Drawing the corresponding points yields the **convex hull**: the minimal convex set containing the points of the ROC curve.

### 9.7 Model Selection: Cost of a Classifier

Sometimes — especially in medicine — committing an error on the **minority class** (disease present) has much worse consequences than on the majority class (healthy). So we need to express the **cost of the error**.

- $P(n)$ and $P(p)$: a-priori probabilities of a negative and a positive example.
- $C(Y,n)$ and $C(N,p)$: **false positive cost** and **false negative cost**.

**Total cost** = (negative instances misclassified as positive × their cost) + (positive instances misclassified as negative × their cost). FNR and TNR are obtained from the confusion matrix. **The misclassification costs must be given by an expert in the specific field.**

Once $P(n)$, $C(Y,n)$, $P(p)$, $C(N,p)$ are fixed, we obtain a family of parallel lines called **iso-cost lines**, with slope:

$$\text{slope} = \frac{P(n)\,C(Y,n)}{P(p)\,C(N,p)}$$

- Points belonging to the **same line have the same cost**, and the cost **decreases** as we move to parallel lines closer to the point **(0,1)** (i.e., more north-west).
- **The point that minimizes the classification cost = the tangent point between the ROC curve (convex hull) and the family of parallel iso-cost lines.**
- Practically: plot the convex hull, compute the coefficient of the cost line, and move it until you find the tangent one.

> This technique finds the best **parameter combination for a single classifier** — assuming we know the cost of the error.

### 9.8 Comparing different classifiers with AUC

1. Classify the test set with **both** classifiers → obtain a confusion matrix and one point $(FPR, TPR)$ for each classifier.
2. **Connect that point to (0,0) and to (1,1)**, and compare the resulting **area under the curve** for each classifier.
3. The **highest area corresponds to the point closest to the upper-left corner (0,1)** → **AUC is a metric for the quality of the classifier**.
4. By computing the AUC for **each trial/fold** (as in cross-validation), we generate a **distribution of AUCs** instead of a distribution of accuracies.
5. Then **perform a statistical test** on the distributions of AUCs, and pick the classifier with the higher AUC.

> **To sum up: the three ways to compare classifiers are ACCURACY, F-MEASURE, and AREA UNDER THE CURVE (AUC).**

---

## 10. Ensemble Methods: Improving Classification Accuracy

**How can we improve the performance of a classifier in general? By using ensembles**: use multiple classifiers and combine their outputs. Combine a series of $k$ learned models $M_1, M_2, \dots, M_k$ with the aim of creating an **improved model $M^*$**.

**Popular ensemble methods:**
- **Bagging**: averaging the prediction over a collection of classifiers.
- **Boosting**: weighted vote with a collection of classifiers.
- **Ensemble**: combining a set of **heterogeneous** classifiers.

> **Distinction** (lecture): in **bagging and boosting** we typically use the **same model type** and just combine multiple classifiers; in **ensembles** we also use **different models**.

**The core problem**: we only have **one training set**. If we execute a **deterministic** algorithm (like a decision tree) we obtain the **same classifier every time**. We must find a way to generate a different classifier each time.

### 10.1 Bagging (Bootstrap Aggregation)

*Analogy: diagnosis based on multiple doctors' majority vote.*

**Solution to the "same classifier" problem: generate different training sets via sampling with replacement.**

**Training**: given a set $D$ of $d$ tuples, at each iteration $i$, a training set $D_i$ of $d$ tuples is **sampled with replacement** from $D$ (i.e., a **bootstrap** sample). A classifier model $M_i$ is learned for each training set $D_i$.

**Classification** (classifying an unknown sample X):
- Each classifier $M_i$ returns its class prediction.
- The **bagged classifier $M^*$ counts the votes** and assigns the class with the **most votes** to X.

**Prediction**: bagging can be applied to the prediction of **continuous values** by taking the **average** of each prediction for a given test tuple.

**Accuracy:**
- **Often significantly better** than a single classifier derived from $D$ — because we are using more "perspectives".
- **For noisy data**: not considerably worse, **more robust**.
- **Proved improved accuracy** in prediction.

> **The trade-off**: we generally expect ensembles to perform better than a single classifier, but we **lose something in explainability** (more steps between input and output).

**Illustration of the power of an ensemble**: consider a simple two-class problem described by two attributes $x_1, x_2$ with a **linear decision boundary**. A single decision tree classifier produces a **staircase-like** boundary; an **ensemble** of decision tree classifiers produces a boundary that models the true linear separation much better. **With ensembles we can better model the boundary between classes** — this is why we expect better performance.

### 10.2 Boosting

*Analogy: consult several doctors, based on a combination of **weighted** diagnoses — weight assigned based on previous diagnosis accuracy.*

**How it works:**
- **Weights are assigned to each training tuple.**
- A series of $k$ classifiers is **iteratively** learned.
- After a classifier $M_i$ is learned, the weights are updated so the subsequent classifier $M_{i+1}$ **pays more attention to the training tuples that were misclassified by $M_i$**.
- The final $M^*$ **combines the votes** of each individual classifier, where the weight of each classifier's vote is a **function of its accuracy**.

> **The mechanism** (lecture): we generate the first classifier, which classifies some instances correctly and misclassifies others. We then **weight the misclassified instances more** in the training set used to generate the second classifier — so their probability of being selected is higher. Repeating this, we **hyper-focus on the misclassified instances** and reduce misclassification at each iteration.

- Boosting can be extended for **numeric prediction**.
- **Comparison with bagging**: boosting tends to have **greater accuracy**, but it also **risks overfitting** the model to misclassified data.

### 10.3 AdaBoost (Freund and Schapire, 1997)

The most famous and used boosting technique.

**Setup**: given a set of $d$ class-labeled tuples $(X_1,y_1),\dots,(X_d,y_d)$. **Initially all tuple weights are set to $1/d$** (equal probability of being selected). Generate $k$ classifiers in $k$ rounds.

**At round $i$:**
- Tuples from $D$ are **sampled with replacement** to form a training set $D_i$ of the **same size** (used to learn model $M_i$).
- **Each tuple's chance of being selected is based on its weight.**
- A classification model $M_i$ is derived from $D_i$.
- Its **error rate is calculated using $D_i$ as a test set**. The error rate is the **sum of the weights of the misclassified tuples**:

$$error(M_i) = \sum_{j=1}^{d} w_j \times err(X_j)$$

where $err(X_j)$ is the misclassification error of tuple $X_j$ (1 if misclassified, 0 otherwise).

**Weight update:**
- If a tuple $(X_j, y_j)$ is **correctly classified**, its weight is updated as:

$$w_j = w_j \cdot \frac{error(M_i)}{1 - error(M_i)}$$

 *(this **decreases** the weights of correctly classified instances)*
- Then the weights of **all** tuples (both correctly and incorrectly classified) are **normalized** so their sum remains the same as before. Normalization = multiply the weight by the sum of the **old** weights and divide by the sum of the **new** weights. **The net effect: the weights of misclassified tuples are increased!**

**Using the ensemble to predict** — boosting assigns a **weight to each classifier's vote**, based on how well the classifier performed. *(The idea: "believe more" the classifier that performs better.)*
- The **lower** a classifier's error rate, the **more accurate** it is, so the **higher** its voting weight should be.
- The weight of classifier $M_i$'s vote is:

$$\log\frac{1 - error(M_i)}{error(M_i)}$$

- For each class $c$, **sum the weights of every classifier that assigned class $c$ to X**. The class with the **highest sum** is the winner and is returned as the class prediction for tuple X.

### 10.4 Random Forest (Breiman 2001) ⭐

> **Flagged in the lecture notes as "a very good classifier for the project!"**

A **Random Forest (RF)** is an **ensemble of decision trees**.

**The problem RF solves**: starting from the same training set always gives the same decision tree (deterministic approach). **Breiman observed that using bootstrapping alone doesn't bring much variance between the decision trees.** So he introduced additional **randomness**:
- Each classifier is a decision tree classifier generated using a **random selection of attributes at each node** to determine the split.
- **Each tree votes** and the **most popular class** is returned.

**Algorithm** — let $N$ = number of training instances, $M$ = number of attributes, $m$ = number of attributes to be used for choosing the decision attribute at any node:

1. **Choose a training set** by randomly extracting $N$ samples **with replacement** from all available training instances (i.e., take a **bootstrap sample**). Use the **rest of the instances to estimate the error** of the tree.
2. **For each node** of the tree, **randomly choose $m$ attributes** on which to base the decision at that node. Calculate the best split based on **only these $m$ variables**.
   > *Contrast with the classical approach*: normally we test **all** attributes and select the best; here we introduce randomness by testing only a **randomly selected subset** and then choosing the best among them.
3. **Each tree is fully grown and NOT pruned** (unlike a normal tree classifier).
   > *Why no pruning is needed*: by randomly selecting attributes we already generate enough differences between trees.

**Classification**: each tree assigns a class to the unlabeled instance; the **most popular class** is returned.

**Advantages:**
- RF is **one of the most accurate learning algorithms available** — for many datasets it produces a highly accurate classifier.
- Runs **efficiently on large databases** (it can work **in parallel**).
- Can handle **thousands of input variables** without variable deletion.
- Gives **estimates of which variables are important** in the classification.
- Generates an **internal unbiased estimate of the generalization error** as the forest building progresses.
- Has an effective method for **estimating missing data** and maintains accuracy when a large proportion of the data are missing.

> **Practical note**: usually **more than 100 decision trees** are used — so we need a sufficient number of samples in the training set.

**Disadvantages:**
- RFs have been observed to **overfit** for some datasets with **noisy** classification/regression tasks.
- For data including **categorical variables with different numbers of levels**, RFs are **biased in favor of attributes with more levels** → the **variable importance scores are not reliable** for this type of data.
- Decision trees are interpretable classifiers, but **the complexity of random forests loses this characteristic** → **less explainability**.

### 10.5 Classification of Class-Imbalanced Data Sets ⭐

> **Flagged in the lecture notes as "important for the project!"**

**Class-imbalance problem**: rare positive examples but numerous negative ones — e.g., medical diagnosis, fraud, oil-spill, fault detection. **Traditional methods assume a balanced distribution of classes and equal error costs → not suitable for class-imbalanced data.**

> **The learning-side problem** (lecture): up to now we approached imbalance from an **evaluation** point of view (how to measure performance). But there are also effects on the **learning process**: if learning just aims to minimize error / maximize accuracy, the classifier suffers a **bias towards the majority class**. This problem is **intrinsic to the training set**.

**The solution: don't change the learning algorithm — REBALANCE THE TRAINING SET (NOT the overall dataset).**

> ### ⚠️⚠️ THE CRITICAL RULE
> **We can NEVER rebalance the test set** — that would falsify the results. **During the learning process the test set is never involved.** Since rebalancing is part of the **learning phase**, it must never touch the test set.
>
> **Applying 10-fold cross-validation correctly**: for each fold → select the training set → **rebalance it** → learn → measure F-measure/AUC **on the original (still imbalanced) test set**. Then the next iteration: again rebalance the new training set, and again measure on the **original** test set.
>
> Of course the test set stays imbalanced — **which is exactly why we must avoid using accuracy** and use F-measure/AUC instead.

**Typical methods for imbalanced data in 2-class classification:**
- **Oversampling**: re-sampling of data from the **positive (minority) class** — generating synthetic minority-class instances that should emulate the true minority class.
- **Under-sampling**: randomly **eliminate tuples from the negative (majority) class** to decrease its number of samples.
- **SMOTE** (Synthetic Minority Over-sampling Technique): an oversampling technique applicable to **numerical data** (detailed below).
- *(Other techniques not covered in class)*: **Threshold-moving** — move the decision threshold $t$ so rare-class tuples are easier to classify, reducing the chance of costly false-negative errors; and **Ensemble techniques**.
- **Still difficult** for the class imbalance problem on **multiclass** tasks.

#### SMOTE — Synthetic Minority Over-sampling Technique

Over-samples the minority class by creating **"synthetic" examples** rather than by over-sampling with replacement.

**Method:**
- The minority class is over-sampled by taking **each minority class sample** and introducing synthetic examples **along the line segments joining any/all of the $k$ minority class nearest neighbors**.
  > So: start from an instance, look at its $k$ nearest neighbours belonging to the same class, and randomly generate new instances **along the line** between our instance and the chosen neighbour.
- Depending on the amount of over-sampling required, **neighbors from the $k$ nearest neighbors are randomly chosen**.
  - *Example*: if the amount of over-sampling needed is **200%**, only **two** neighbors from the five nearest are chosen, and **one sample is generated in the direction of each**.

**How synthetic samples are generated:**
1. Take the **difference** between the feature vector (sample) under consideration and its nearest neighbor.
2. **Multiply this difference by a random number between 0 and 1**, and add it to the feature vector under consideration.
3. This causes the selection of a **random point along the line segment** between two specific features. **This approach effectively forces the decision region of the minority class to become more general.**

---

## 11. Summary (from the final slides)

- **Classification** is a form of data analysis that extracts models describing important data classes.
- **Effective and scalable methods** have been developed for decision tree induction, Naïve Bayesian classification, rule-based classification, and many other classification methods.
- **Evaluation metrics** include: accuracy, sensitivity, specificity, precision, recall, $F$ measure, and $F_\beta$ measure.
- **Stratified k-fold cross-validation** is recommended for accuracy estimation. **Bagging and boosting** can be used to increase overall accuracy by learning and combining a series of individual models.
- **Significance tests and ROC curves** are useful for model selection.
- There have been numerous comparisons of the different classification methods; the matter **remains a research topic**.
- **No single method has been found to be superior over all others for all datasets.**
- Issues such as **accuracy, training time, robustness, scalability, and interpretability** must be considered and can involve **trade-offs**, further complicating the quest for an overall superior method.

---

## 12. Key points / potential exam pitfalls

### Basic concepts
- **Why we need a test set at all**: once the model runs on genuinely unlabeled data, accuracy is uncomputable — there's no ground truth. The test set is *the same format* as the training set (labeled); only its **purpose** differs.
- **Overtraining detection is a comparison, not an absolute number**: it's not "training accuracy is high", it's "**training accuracy is much higher than test accuracy**". Similar accuracies on both = no overtraining.

### Discretization
- **Fayyad & Irani's key insight** is a *search-space reduction*: optimal cut points always lie **between examples of different classes**, so you never need to test cut points inside a homogeneous run of one class.
- **Entropy = 0 ⟺ pure partition** — the single most reused fact in this chapter (discretization, decision trees, and rule quality all build on it).
- The professor explicitly said the **stopping-condition formula need not be memorized** — only that *a* termination condition exists, which is why Fayyad–Irani needs **no parameters**.

### Decision trees
- **Information Gain is biased toward multivalued attributes** — the canonical counterexample is an **ID/unique-identifier attribute**: it yields perfectly pure partitions (gain maximal) but a **useless tree**, since a new instance's ID matches no branch at all. **Gain Ratio (C4.5) fixes this** by dividing by **SplitInfo**, which grows with the number of partitions.
- **Gini index only produces BINARY trees** — for an attribute with $v>2$ values you must test **all possible subset splits**. This is the structural difference from Information Gain / Gain Ratio, which naturally produce multiway splits.
- **The weights in $Info_A(D)$ are not decoration**: without them, isolating a single instance would look "perfectly pure" and win every time. Weighting by $|D_j|/|D|$ makes cardinality count.
- **No attribute selection measure is best** — all have bias; **in practice we use gain ratio**.
- **Three (and only three) basic stopping conditions**: same class / no attributes left (→ majority voting) / no samples left. Everything beyond these is **prepruning**.
- **Prepruning vs postpruning — why we prefer postpruning**: every prepruning restrictive condition requires choosing a **threshold**, and we don't know the optimal one. Postpruning avoids this by growing fully first, then cutting back using data.
- **REP's two-set logic is the classic exam trap**: the **leaf's label** comes from the **TRAINING set majority**, but the **error count** comes from the **PRUNING set**. Mixing these up gives the wrong answer every time. And **prune only if the error at $v$ is ≤ the error of the subtree $T$** — pruning can *increase* error (see Case 2 of the worked example), in which case you keep the subtree.
- **The pruning set is carved out of the training set**, not the test set: dataset → {training, test}, then training → {growing set, pruning set}.
- **Cost complexity needs the $\beta$ penalty**: using resubstitution error alone would mean **never pruning**, since error only ever grows when you prune.
- **C4.5's pessimistic pruning uses NO pruning set** — it adds a statistical penalty to training-set error instead. This is the distinguishing feature vs REP and cost-complexity pruning.
- **Repetition vs Replication**: *repetition* = same attribute tested repeatedly **along one branch**; *replication* = **duplicate subtrees** elsewhere in the tree. Both arise when branches aren't mutually exclusive (binary trees); both hurt **explainability** more than accuracy. Multiway trees / multivariate splits / rule representation avoid them.

### Bayes
- **$P(X)$ is dropped because it's constant across classes** — that's why the final numbers don't sum to 1. This is expected, not an error.
- **The naïve independence assumption is what makes it tractable** — it converts an exponential number of joint probabilities into a linear number of per-attribute probabilities. But **the assumption is normally false**, which is the classifier's main accuracy cost.
- **Zero-probability problem**: because $P(X|C_i)$ is a **product**, a single zero factor zeroes the whole thing. **Laplacian correction** (add 1 to each count) fixes it while barely perturbing the estimates.
- **Naïve Bayes is incremental; decision trees are not** — adding training instances just updates counts, whereas a tree must be rebuilt.
- **Bayesian Belief Networks** = **DAG + CPTs**. Their whole point is to model the **dependencies** naïve Bayes ignores. Nodes can be **hidden/fictitious variables** (e.g., a syndrome). Scenario 1 (structure known, all observable) is trivial — just count CPT entries; Scenario 4 (unknown structure, all hidden) has **no good algorithm**. **In practice only scenarios 1 and 2 are used.**

### Rules
- **Rules extracted from an unpruned decision tree are mutually exclusive and exhaustive → no conflicts.** But **after pruning that guarantee is lost**, and you need conflict resolution (C4.5 uses class-based ordering minimizing false positives, plus a **default class** = the majority class among tuples covered by no rule).
- **Accuracy alone is a bad rule-quality metric**: a rule covering 2 instances at 100% accuracy beats a rule covering hundreds at 95%, which is clearly wrong. **FOIL_Gain encodes both**: the **log-ratio** captures accuracy, the **leading $pos'$ factor** captures coverage.
- **Sequential covering learns rules ONE AT A TIME** (removing covered tuples each time); **decision-tree induction learns all rules simultaneously**. This is the standard comparison question.
- **Learn_One_Rule is optimistic** because it never uses a test set — hence the separate **FOIL_Prune** step on independent tuples; RIPPER prunes starting from the **most recently added** conjunct.

### Lazy learners
- **Lazy = no training time, expensive classification**; eager = the opposite. We usually **prefer eager**, because classification time is what the **user perceives**.
- **k-NN complexity stays $O(N)$ even for $k>1$** — you just keep $k$ running minima; you don't need to fully sort.
- **Missing values in k-NN → assume the MAXIMUM possible distance** (1), not zero.
- **Editing methods trade accuracy against compression**: Wilson improves accuracy but compresses little; supervised clustering compresses enormously but can hurt accuracy. **A trade-off is required — over-reduction damages accuracy.**
- **CBR stores symbolic descriptions, NOT Euclidean points** — that's the fundamental difference from k-NN, even though its *Retrieve* step is essentially a k-NN retrieval task.
- **The 4 R's in order: Retrieve → Reuse → Revise → Retain.** Retain is the **learning** phase.

### Model evaluation ⭐
- **Accuracy is meaningless on imbalanced data** — the "always predict the majority class" classifier scores 95% while having **specificity 0**. This is the single most emphasized point of the chapter.
- **Convention: the minority class is the positive class.**
- **Precision and recall must be quoted in pairs** — either alone is uninformative (perfect precision says nothing about missed positives; perfect recall says nothing about false alarms). $F_1$ merges them into a single comparable index.
- **$F_\beta$ direction**: $\beta > 1$ (e.g. $F_2$) weights **recall** more; $\beta < 1$ (e.g. $F_{0.5}$) weights **precision** more.
- **One train/test split proves nothing** — by chance a classifier can look better. You need **multiple trials → a distribution**, and then a **statistical test**. Comparing **averages alone is invalid** when distributions overlap.
- **t-test = parametric** (assumes **normally distributed** accuracies + comparable variances; tests **means**); **Wilcoxon = non-parametric** (assumes **no distribution**; tests **medians**). Choose Wilcoxon when you can't justify normality.
- **The paired t-test requires the SAME cross-validation partitioning** for both classifiers. With two different test sets you must use the **non-paired** version, with DoF = the **minimum** of the two.
- **DoF = k−1** because fixing $k-1$ samples plus the mean determines the last one.
- **Bootstrap .632**: the 63.2%/36.8% split isn't arbitrary — it comes from $(1-1/d)^d \to e^{-1} = 0.368$.
- **ROC axes**: y = TPR, x = FPR. **Best point = (0,1)**. **AUC = 0.5 (the diagonal) = worthless model; AUC = 1.0 = perfect.**
- **Iso-cost lines**: cost decreases moving **north-west**; the optimum is the **tangent point** between the iso-cost family and the ROC convex hull. Requires expert-supplied misclassification costs.
- **Three ways to compare classifiers: accuracy, F-measure, AUC** — and for imbalanced data, only the last two are trustworthy.

### Ensembles
- **The fundamental obstacle is determinism**: the same training set + the same deterministic algorithm = the same classifier. **Bagging** breaks this with **bootstrap sampling**; **AdaBoost** breaks it with **instance weights**; **Random Forest** breaks it with **both bootstrap sampling AND random attribute subsets per node**.
- **Bagging = equal votes; Boosting = weighted votes** (weight $=\log\frac{1-error}{error}$, so more accurate classifiers count more).
- **Boosting usually beats bagging on accuracy but risks overfitting** the misclassified data; bagging is **more robust to noise**.
- **AdaBoost's weight update is written for CORRECTLY classified tuples** (multiply by $\frac{error}{1-error}$, i.e. *decrease* them); misclassified tuples' weights rise **through the renormalization step**, not through a separate formula. This is a common point of confusion.
- **Random Forest deliberately does NOT prune** — randomness across trees already supplies the needed diversity/regularization.
- **RF's bias**: with categorical variables having **different numbers of levels**, RF favors those with **more levels** → its **variable-importance scores become unreliable** for such data.
- **Ensembles cost explainability** — a decision tree is interpretable; a forest of 100+ trees is not.
- **⚠️ Rebalance ONLY the training set, never the test set** — and inside cross-validation, rebalance **per fold**, always evaluating on the **original imbalanced** test fold. Rebalancing the test set falsifies results.
- **SMOTE generates synthetic points along line segments** to same-class nearest neighbours (difference × random number in [0,1]), rather than duplicating existing minority instances — this **generalizes the minority class decision region** instead of just reinforcing existing points.

---

*File auto-generated by merging `4-Classification.pdf` (professor's slides, 99 pages) and `4 - Classification Sbobine.pdf` (lecture notes for L07/L08/L09/L11, 81 pages). For questions about this chapter, refer only to this file.*
