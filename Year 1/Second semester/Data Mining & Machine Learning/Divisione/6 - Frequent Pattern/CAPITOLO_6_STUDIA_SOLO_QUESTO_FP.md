# Chapter 6 — Mining Frequent Patterns, Associations and Correlations

> **Course:** Data Mining and Machine Learning / Bioinspired Computational Methods — Biological Data Mining
> **Instructor:** Prof. Francesco Marcelloni (University of Pisa)
> **Source:** `6 - Associative Rule.pdf` / `6 - Associative rule sbobine.pdf` (25 pages)
>
> **⚠️ Note on sources:** unlike Chapters 2–5, the two PDFs in this folder are **the same document** (identical text content, only the file metadata differs). So this chapter has a single source — lecture-notes-style prose covering the whole topic — rather than a separate slide deck + notes pair. If a separate slide deck for this chapter turns up later, it can be merged in.
>
> **Instructions for Claude:** this is the single reference file for Chapter 6. When the user asks study questions about this chapter, use **only** this file as context.

---

## Chapter outline
1. Basic Concepts: patterns, support, confidence
2. Compact representations: closed and maximal itemsets
3. The Apriori algorithm
4. Generating association rules from frequent itemsets
5. FP-Growth (frequent pattern growth)
6. ECLAT (vertical data format)
7. Pattern evaluation methods (lift, null-invariant measures, Kulczynski + IR)

---

## 1. Basic Concepts

### What is a pattern?

The simplest definition of a **pattern** is **a set of items**. This concept originates from **market basket analysis**, where the goal is to identify **groups of products that are frequently purchased together** — for example, *milk* and *bread* often co-occur in transaction datasets.

> **⚠️ In this context, only the PRESENCE of items is considered, not their quantities.**

**Other types of patterns** (beyond a simple set of items):
- A **subsequence** — e.g., first buying a PC, then a digital camera, then a memory card. Here we look not only at *what* is purchased together, but **in which sequence over time** the purchases occur.
- A **substructure** — different structural forms, such as **subgraphs** or **subtrees**.

### Frequent patterns

**We are not interested in all possible patterns.** Instead, we focus on **frequent patterns**, i.e., patterns that occur frequently in a dataset. Typically we define a **minimum threshold** and consider a pattern frequent only if its **support** (the number of transactions in which the items appear together) exceeds this threshold.

The discovery of frequent patterns plays a fundamental role in mining **associations, correlations**, and other meaningful relationships.

> **⚠️ This process belongs to UNSUPERVISED learning**: we first identify frequent patterns and then derive association rules from them.
>
> **⚠️⚠️ CORRELATION DOES NOT IMPLY CAUSALITY — it only indicates that two or more items tend to appear TOGETHER.**

**Applications**: basket data analysis (containers of supermarket products), cross-marketing, catalog design, sale campaign analysis, Web log (click stream) analysis, and **DNA sequence analysis**.

### Definitions

- **Itemset $I$**: a set of one or more items $I_1, I_2,\dots,I_m$. This set corresponds to a **transaction** (in the supermarket case, the receipt of purchased products).
  > **The order is not important**, because it depends on how randomly items are passed to the reader. If we impose an order, it is **only for computational reasons**.
- **$k$-itemset** $X = \{I_1, I_2, \dots, I_k\}$ — e.g., a 1-itemset contains only one element, a 2-itemset contains 2 elements.
- A **transaction $T$** is a set of items such that $T \subseteq I$.
- An **association rule** is an implication of the form
$$X \Rightarrow Y$$
where $X \subseteq I$, $Y \subseteq I$, and **$X \cap Y = \emptyset$**.

**Support:**
- **(Absolute) support**, or **support count** of $X$: the **frequency or occurrence** of the itemset — the **number of transactions** in which the itemset is present. *(E.g., the support count of "beer" and "diaper" appearing together is 3.)*
- **(Relative) support $s$**: a **normalized measure (a probability)** obtained by dividing the support count by the **total number of transactions**. The support of $X\Rightarrow Y$ in the transaction database $D$ (i.e., the **union** of the sets $X$ and $Y$) is:

$$supp(X\Rightarrow Y) = P(X\cup Y)$$

i.e., **the probability that a transaction contains BOTH $X$ and $Y$** (the percentage of transactions in $D$ containing both).

- An itemset $X$ is **frequent** if its support is **no less than a `minsup` threshold**. *(Another parameter to fix — it depends on the application domain.)*

**Confidence:**

$$conf(X\Rightarrow Y) = P(Y\mid X) = \frac{supp(X\cup Y)}{supp(X)}$$

which represents the **conditional probability** that a transaction containing $X$ also contains $Y$.

> **Support vs Confidence — the key contrast:**
> - **Support** expresses the **relevance** of the rule in our domain → it reflects **USEFULNESS**.
> - **Confidence** is the conditional probability that a transaction containing $X$ (the antecedent) also contains $Y$ → it reflects **CERTAINTY**. It expresses the probability of observing the **consequent** given the **antecedent**.
>
> Confidence ranges between **0 and 1**:
> - **= 0** when the numerator is zero → $X$ and $Y$ **never** appear together in any transaction.
> - **= 1** when **every time $X$ appears, $Y$ also appears**.
>
> **⚠️ NB: having confidence = 1 does NOT imply that $X$ and $Y$ are correlated.** If the consequent part of the rule is present in **all** transactions (also in those where $X$ does not appear), we **cannot** say that $X$ and $Y$ are correlated. *(This is the core motivation for the lift measure — see §7.)*

**NB (Boolean representation)**: each item has a **boolean variable** representing the **presence or absence** of that item — we are **not** considering the numerosity of items. Each basket can therefore be represented by a **Boolean vector** of values assigned to these variables.

**Our aim** is to find all the rules $X\Rightarrow Y$ (with empty intersection) having:
- support higher than **minsup**, and
- a certain value of **confidence**.

We use algorithms that impose the minimum values for support and confidence, and generate **all possible association rules satisfying the constraints**.

### Worked example (beer & diaper)

With **minsup = 50%** and **minconf = 50%**, on a 5-transaction database:

**Frequent itemsets:**
- 1-itemsets: `Beer:3`, `Nuts:3`, `Diaper:4`, `Eggs:3`
- 2-itemset: `{Beer, Diaper}:3`

From the 2-itemset we can generate **2 association rules**, with — obviously — the **same support**:

| Rule | Support | Confidence |
|---|---|---|
| **Beer ⇒ Diaper** | 60% | **100%** |
| **Diaper ⇒ Beer** | 60% | **75%** |

**Why the confidences differ:**
- In the first case, confidence is **100%** because **every time we have beer we also have diaper** — so the support of {beer, diaper} together equals the support of beer alone ($3/3$).
- In the second case, **diaper is present in another transaction in which beer isn't present** ($3/4 = 75\%$).

> **⚠️ This shows that if we SWAP consequent and antecedent, the CONFIDENCE CAN CHANGE** (while the support stays the same).

### Association rule mining = a two-step process

1. **Find all frequent itemsets**: by definition, each of these itemsets occurs at least as frequently as the predetermined minimum support count, `min_sup`.
2. **Generate strong association rules from the frequent itemsets**: by definition these rules must satisfy **minimum confidence** and minimum support.
   > *(The support specification is not strictly necessary here, because when we talk about frequent patterns we are already sure we are above the minimum support.)*

### The combinatorial explosion problem

Suppose the item set contains **100 items**. In principle we could generate **all possible combinations** of products to obtain all candidate itemsets that could potentially be frequent. The total number of possible sub-itemsets generated from 100 items is:

$$\binom{100}{1} + \binom{100}{2} + \dots + \binom{100}{100} = 2^{100} - 1 \approx 1.27\times10^{30}$$

where the binomial coefficient is

$$\binom{n}{k} = \frac{n!}{k!\,(n-k)!}$$

> **Generating all possible itemsets is computationally INFEASIBLE**, since the number of combinations grows **exponentially** with the number of items. For this reason, specific algorithms have been developed to efficiently find frequent patterns — **they exploit properties of the SUPPORT measure to prune large portions of the search space.**

**A second, distinct issue — the size of the OUTPUT:** when the `minsup` threshold is set to a very **low** value, the number of frequent itemsets discovered may still be **extremely large**, possibly reaching **millions of patterns**. Presenting such a large output to a user becomes **impractical**.

> **Therefore an important problem is: how to SUMMARIZE the set of frequent itemsets in a compact way WITHOUT LOSING INFORMATION.** Several approaches address this: **closed frequent itemsets** and **maximal frequent itemsets**.

---

## 2. Compact Representations: Closed and Maximal Itemsets

### Closed itemsets

An itemset $X$ is **closed** in a dataset $D$ if $X$ is **frequent** and there exists **no proper super-itemset $Y$** such that $X\subset Y$ **and** $supp(Y) = supp(X)$.

> **A closed itemset provides a LOSSLESS compression of frequent patterns.**

**Illustration** — let $I' = \{I_1,I_2\}$ and $I'' = \{I_1,I_2,I_3\}$, with $supp(I') = 20$:

| Case | Consequence |
|---|---|
| $supp(I'')=20$ (equal) | **$I'$ is NOT closed**, since a proper superset has the same support. It is sufficient to report only $I''$ as closed — all its subsets share the same support, so the complete information can be **recovered from $I''$**. |
| $supp(I'')=15$ (lower) | **$I'$ IS closed**, since no superset of $I'$ has the same support. |
| $supp(I'')=25$ (higher) | **IMPOSSIBLE** — the support of a superset **cannot exceed** the support of its subset: $supp(I'')\le supp(I')$. |

### Maximal itemsets

An itemset $X$ is a **maximal itemset** in $D$ if $X$ is **frequent** and there exists **no frequent super-itemset $Y\supset X$**.

> **⚠️ Unlike closed itemsets, maximal itemsets do NOT preserve complete support information.** If we return only maximal itemsets, **some information about the supports of their subsets is LOST.**

### The decisive example

Let $C$ be the set of closed frequent itemsets and $M$ the set of maximal frequent itemsets for a database $DB$ satisfying `minsup`.

$$DB = \{\ \langle a_1,\dots,a_{100}\rangle,\ \ \langle a_1,\dots,a_{50}\rangle\ \}, \qquad min\_sup = 1$$

**What is the set of CLOSED itemsets?**
$$\langle a_1,\dots,a_{100}\rangle : 1 \qquad\text{and}\qquad \langle a_1,\dots,a_{50}\rangle : 2$$

**What is the set of MAXIMAL itemsets?**
$$\langle a_1,\dots,a_{100}\rangle : 1$$

**The observation that matters:**
> **The set of closed frequent itemsets contains COMPLETE information regarding the frequent itemsets.**
>
> From the set of closed itemsets $C$ we can correctly derive $\{a_2,a_{45}\}:2$ — because $\{a_2,a_{45}\}\subseteq\langle a_1,\dots,a_{50}\rangle$, which has support 2.
>
> By contrast, if we return only the **maximal** itemset $\langle a_1,\dots,a_{100}\rangle$ (it has no frequent supersets), **the user CANNOT infer that $\{a_2,a_{45}\}$ appears in BOTH transactions** and therefore has support 2 — from the maximal representation alone one would only infer support 1.

> **⚠️ CONCLUSION: in order NOT to lose any information, we can return only CLOSED itemsets, and NOT maximal ones.**

---

## 3. The Apriori Algorithm

### Sensitivity to minsup and the search space

In any frequent-pattern mining algorithm, **the number of frequent itemsets to be generated is sensitive to the `minsup` threshold**. When `minsup` is low, there potentially exists an **exponential** number of frequent itemsets.

The **worst-case** number of possible itemsets is $M^N$, where $M$ = number of distinct items in the database and $N$ = maximum length of the transactions.

> **However**, this worst-case complexity should be evaluated considering the **expected probability of occurrence** of the itemsets. In practice, **itemsets containing a small number of items are more likely to appear frequently**. As the number of items in an itemset increases, the probability that all those items appear together in the same transaction **decreases** — their support tends to **fall below the threshold**, preventing the algorithm from exploring larger itemsets. **This mechanism effectively reduces the search space.**
>
> *Example*: suppose Walmart has $10^4$ kinds of products. The chance of picking one particular product is $10^{-4}$, and the chance of picking a particular set of 10 products is $10^{-40}$ (assuming all products equally probable).

### The Apriori property

The most important algorithm proposed to mine frequent patterns is the **Apriori algorithm**. It exploits the **Apriori property**:

> **ANY NON-EMPTY SUBSET OF A FREQUENT ITEMSET MUST ALSO BE FREQUENT.** *(This is a necessary condition.)*

*Example*: if {beer, diaper, nuts} is frequent, so is {beer, diaper} — every transaction having beer, diaper, nuts also contains beer, diaper.

**This property belongs to a special category called ANTIMONOTONICITY**: *if a set cannot pass a test, all of its supersets will fail the same test as well.* (It is called antimonotonicity because the property is monotonic in the context of **failing** a test.)

> **The practical consequence: it's useless to generate itemsets starting from itemsets that are NOT frequent.** This limits the generation of itemsets and speeds up the process.

**Later algorithms** (after Apriori, 1994):
- **Frequent pattern growth (FP-Growth)** (2000) — generates **exactly the same patterns** as Apriori, but in a **very efficient** way.
- **Vertical data format approach** (2002) — ECLAT.

### Method of the Apriori algorithm

1. **Initially, scan the database once** to get the frequent **1-itemsets** (check whether all the single items are frequent).
2. **Generate length-$(k+1)$ candidate itemsets from length-$k$ frequent itemsets** (e.g., combine 1-itemsets to generate 2-itemsets). **It is not guaranteed** that the new candidate itemsets are frequent — but in any case we **reduce the number of itemsets generated**, because we start only from itemsets satisfying the necessary condition.
3. **Test the candidates against the database** — count in how many transactions these itemsets are present.
4. **Terminate when no frequent or candidate set can be generated.** *(With the increase of items in an itemset, the probability of the itemset decreases.)*

**Notation**: $L_k$ = the set of **frequent** itemsets of size $k$; $C_k$ = the set of **candidate** itemsets of size $k$.

> **Pruning in action**: suppose item $D$ is found to be **infrequent**. According to the Apriori property, **any itemset containing $D$ cannot be frequent**. Therefore it is **unnecessary to generate candidate 2-itemsets that include $D$**.
>
> After generating the candidate 2-itemsets, we must **verify whether they are frequent** by scanning the transaction database and counting the transactions in which each candidate appears. **Only the itemsets whose support ≥ minsup are retained** in $L$.

### Pseudocode

```
C_k : candidate itemsets of size k
L_k : frequent itemsets of size k

L_1 = {frequent items};
for (k = 1; L_k ≠ ∅; k++) do
    C_{k+1} ← candidates generated from L_k;
    for each transaction t in the database do
        increment the count of all candidates in C_{k+1} that are contained in t;
    L_{k+1} ← candidates in C_{k+1} with min_support;
end
return ∪_k L_k;
```

**Input**: the database and `minsup`. **Output**: frequent patterns. It is a **two-step algorithm** consisting of **JOIN** and **PRUNE** actions.

### The Join step

To find $L_k$, a set of candidate $k$-itemsets **$C_k$ is generated by joining $L_{k-1}$ with itself**.

**Notation**: $l_i[j]$ refers to the $j$-th item in $l_i$ (e.g., $l_1[k-2]$ is the second-to-last item in $l_1$).

> **For efficient implementation, Apriori assumes that items within a transaction or itemset are sorted in LEXICOGRAPHIC ORDER — in this way we AVOID DUPLICATES.** For a $(k-1)$-itemset $l_i$ the items are ordered such that $l_i[1] < l_i[2] < \dots < l_i[k-1]$.

The join $L_{k-1}\bowtie L_{k-1}$ is performed. **Members of $L_{k-1}$ are joinable if their first $(k-2)$ items are identical** — they differ **only in the last element**. Let $l_1,l_2\in L_{k-1}$. They can be joined if:

$$l_1[1]=l_2[1],\quad l_1[2]=l_2[2],\ \dots,\ l_1[k-2]=l_2[k-2] \qquad\text{and}\qquad l_1[k-1] < l_2[k-1]$$

The resulting candidate $k$-itemset is:

$$c = \{\,l_1[1],\ l_1[2],\ \dots,\ l_1[k-2],\ l_1[k-1],\ l_2[k-1]\,\}$$

### The Prune step

**Any $(k-1)$-itemset that is not frequent cannot be a subset of a frequent $k$-itemset** (the Apriori property). Therefore:

> **If ANY $(k-1)$-subset of a candidate $k$-itemset $c$ is NOT contained in $L_{k-1}$, then $c$ cannot be frequent and must be REMOVED from $C_k$.**

**Worked termination example**: the algorithm uses $L_3\bowtie L_3$ to generate a candidate set of 4-itemsets, $C_4$. Although the join results in $\{\{I_1,I_2,I_3,I_5\}\}$, the itemset $\{I_1,I_2,I_3,I_5\}$ is **pruned because its subset $\{I_2,I_3,I_5\}$ is not frequent**. Thus $C_4 = \emptyset$ and **the algorithm terminates**, having found all the frequent itemsets.

---

## 4. Generating Association Rules from Frequent Itemsets

Association rules can be generated from the frequent itemsets discovered by algorithms such as Apriori.

**For each frequent itemset $I$:**
1. Generate **all the non-empty subsets $s \subset I$**.
2. For each subset, construct the rule:
$$s \Rightarrow (I - s)$$
where $s$ is the **antecedent** and $I-s$ is the **consequent**.
3. **A rule is accepted if its confidence is greater than the minimum threshold `min_conf`.**

$$confidence(A\Rightarrow B) = P(B\mid A) = \frac{support\_count(A\cup B)}{support\_count(A)}$$

### Limitations of these interest measures

- **The Rare Item Problem**: the **support** measure may remove **rare items that could still produce interesting rules**. If an item appears infrequently in the dataset, rules containing that item may be pruned **even if they are meaningful**.
- **Confidence is sensitive to the frequency of the CONSEQUENT**: if the consequent appears very frequently in the dataset, the confidence of a rule may be **high even when there is no real association** between the items.

> **Therefore, support and confidence ALONE are not always sufficient to evaluate the true dependency between items.** *(This motivates §7 — pattern evaluation methods.)*
>
> Moreover, Apriori is **very expensive from a computational point of view**: a very low percentage of the generated itemsets is really frequent. **To improve on Apriori, the FP-Growth approach was introduced.**

---

## 5. FP-Growth: A Frequent Pattern-Growth Approach

**FP-Growth** = *Mining Frequent Patterns Without Candidate Generation*. It allows frequent itemset discovery **without candidate itemset generation** — we don't need to scan the transaction database several times; instead we exploit the **FP-tree**, a data structure that **summarizes** the transaction database.

**Input**: the complete dataset $D$ and `minsup`. **Output**: the complete set of frequent patterns. **Two-step approach:**

- **Step 1 — Construction of the FP-tree**: a compact data structure built by scanning the database **twice**. The FP-tree **compresses the transactions**, **keeps only the frequent items**, and **exploits the common PREFIXES shared by transactions**. As a result, **many transactions can be represented by a single path**, significantly reducing the size of the data.
- **Step 2 — Extract frequent itemsets**: once the tree is built, frequent itemsets are extracted **directly from the FP-tree**. **Unlike Apriori, it is not necessary to generate candidate itemsets.** The algorithm explores the FP-tree using a **depth-first search**; the main idea is to **grow longer patterns from shorter ones**.

### Step 1 in detail — building the FP-tree (2 passes)

**Pass 1**: Scan the data, find the support for each item and **discard infrequent items** (following the Apriori property). Then **sort the frequent items in DECREASING order of support**, and use this order when building the FP-tree **so that common prefixes can be shared**.
> **This is the first difference from Apriori** — thanks to this ordered dataset we simplify the complexity of the algorithm.

**Pass 2**: In the FP-tree **each node represents an item** and contains a **counter** indicating **how many transactions pass through that node**. The algorithm reads **one transaction at a time**, and each transaction is **mapped to a path** in the tree.

*Example*: the transaction $\{bread, milk, butter\}$ is represented by the path
$$bread \to milk \to butter$$

Items in each transaction are sorted according to a **global order** (usually by decreasing support). **This ordering ensures that similar transactions share the same prefix in the tree.** If two transactions share the same prefix, **their paths overlap**:

- $T_1 = \{bread, milk, butter\}$
- $T_2 = \{bread, milk, cheese\}$

$$bread(2) \to milk(2) \to \begin{cases}butter(1)\\ cheese(1)\end{cases}$$

**The counters of `bread` and `milk` become 2**, because **if a transaction passes through an existing node, the node counter is incremented** — indicating that multiple transactions share the same prefix.

**Node links**: **all nodes corresponding to the same item are connected using pointers.** These pointers create a **singly linked list for each item**, allowing the algorithm to **quickly find all occurrences of that item** in the tree.

> **If many transactions share common prefixes, the FP-tree becomes very compact. The more paths overlap, the higher the compression of the dataset.**

**Worked walkthrough (from the notes):** We start from the root with a **null** node, then follow the path `ab`, corresponding to the transaction being observed — the order of items follows their **frequency** ($a$ most frequent, $b$ second). We repeat for the other transactions: since the items differ, from the second transaction we generate a **new path** `bcd`. **The counter represents the number of transactions in which a specific item appears along a given path**; **the links represent the presence of the same item in different paths.** When we analyze the **third** transaction, the first item $a$ is **common** with the first transaction → **we reuse the existing node and increase the counter of $a$ to 2**, then create a **new branch** because the following items are not common. **This is why the items are ordered by decreasing support**: there is a higher probability that transactions share common items **near the top** of the FP-tree, so we don't need a completely new path for each transaction — **we compress the number of paths.**

**Best and worst case:**
- **Best case**: all transactions contain **the same set of items** → only **1 path** in the FP-tree.
- **Worst case**: every transaction has a **unique** set of items (no items in common) → the FP-tree is **at least as large as the original data**, and the **storage requirements are HIGHER** — we also need to store the **pointers** between nodes and the **counters**.

> **The size of the FP-tree depends on how the items are ordered.** This is why ordering by decreasing support is typically used. **However, it does not always lead to the smallest tree** — **it's a HEURISTIC**, we can't guarantee it produces the smallest FP-tree. **This is not important**, because all we need is that the FP-tree **fits in memory**.

### Step 2 in detail — extracting frequent itemsets

FP-Growth extracts frequent itemsets from the FP-tree using a **BOTTOM-UP** approach — **starting from the leaves and moving towards the root**.

The algorithm follows a **divide-and-conquer** strategy: it first searches for frequent itemsets **ending with the item having the LOWEST support** (e.g. $e$), then continues with combinations such as $de$, $d$, $cd$, and so on.

**The idea is to extract PREFIX PATHS ending in a given itemset.** Starting from the complete FP-tree, we identify all occurrences of the selected item (e.g. $e$) and reconstruct all the paths in which that item appears. The resulting structure is a **conditional FP-tree**, obtained by **projecting** the original FP-tree onto the item $e$ — i.e., **we only consider the transactions that contain $e$**.

> **⚠️ NB — a subtle but important point about counters in the PREFIX PATHS:**
> In the extracted prefix paths, **the value of the counter associated with each node remains UNCHANGED** (as in the original tree). Considering the actual meaning of these paths, along each of them the itemset containing $e$ appears **only once**, so each path represents exactly one occurrence of an itemset containing $e$.
> **However**, in some branches the counters of nodes (e.g. $b$ and $c$) may be **equal to 2** — this happens because these nodes are **shared with another branch of the original FP-tree in which the item $e$ does NOT appear**. Consequently, their counters reflect the **total** number of transactions passing through those nodes in the **entire** FP-tree, **not only** the transactions containing $e$.

**Each prefix-path sub-tree is processed RECURSIVELY** to extract the frequent itemsets, and solutions are then **merged**. For example: the prefix-path sub-tree for $e$ is used to extract frequent itemsets ending in $e$, then in $de$, $ce$, $be$, $ae$, then in $cde$, $bde$, etc.

**Worked example with `minSup = 2`:**
1. **Extract all frequent itemsets containing $e$** and obtain the prefix-path sub-tree for $e$.
2. **Check if $e$ is a frequent item** by **adding the counts along the linked list** (the dotted line). If so, extract it. In our case the count equals **3**, so $e$ **is** extracted as a frequent itemset.
3. **As $e$ is frequent, use the conditional FP-tree for $e$** to find frequent itemsets **ending in $e$**: $de$, $ce$, $be$, $ae$. *(Note: $be$ is not considered as $b$ is not in the conditional FP-tree for $e$.)* For each of them (e.g. $de$), find the prefix paths from the conditional tree for $e$, extract frequent itemsets, generate a conditional FP-tree, etc. — **recursively.**

> **⚠️ How counters CHANGE in the CONDITIONAL FP-tree** (the key mechanic):
> In the **original** FP-tree, counters represent the number of transactions passing through each node. But when we construct the **conditional** FP-tree for $e$, **we are interested ONLY in the transactions that contain $e$**.
> - The support of item $a$ in the conditional FP-tree for $e$ is **2**, because only two transactions containing $a$ also contain $e$.
> - Considering the branch starting from $b$: only **one** transaction containing $b$ also contains $e$. Since this is **below minsup**, **$b$ is not frequent in the conditional FP-tree for $e$ → that branch can be REMOVED.**
>
> We then proceed the same way to build the conditional FP-tree for $de$, starting from the conditional FP-tree for $e$. There is only one transaction in which item $c$ appears in the conditional pattern base for $de$ → $c$ is not frequent → that branch is removed. After removing infrequent items we are left with two paths containing the itemset $da$, so **the conditional FP-tree for $de$ consists of a single node $a$ with support 2**.
>
> If instead we try to construct the conditional FP-tree for $ce$, the support of each node is **1** → **we can't construct a conditional FP-tree, because there are no frequent itemsets.**

### The algorithm — summary

**STEP 1: FP-tree construction**
- Compute the support of each item (**first scan** of the database) and **remove infrequent items**.
- **Sort the frequent items in descending order of support.**
- **Insert the transactions into the FP-tree following this order.**
- **Update the node counters when paths overlap** (they indicate how many transactions pass through each node).

**STEP 2: Extraction of frequent itemsets**
- For each item (**from the least frequent to the most frequent**), find all the paths leading to that item.
- Build the **conditional pattern base**.
- Construct the **conditional FP-tree** using those paths.
- **Repeat recursively until the tree becomes empty.**

> **⚠️ NB — the counter rule in one line:**
> - **Original FP-tree**: the counters remain **UNCHANGED** when extracting the paths.
> - **Conditional FP-tree**: the counters are **RECOMPUTED** according to the frequencies of the paths.

### Advantages and disadvantages

**Advantages of FP-Growth:**
- **Only 2 scans over the dataset**, just to generate the FP-tree — a "compressed" version of the original dataset. **If it fits in memory, we're done.**
- **No candidate generation.**
- **Much faster than Apriori**: studies show FP-growth is efficient and scalable for mining both long and short frequent patterns, and is **about an ORDER OF MAGNITUDE faster** than Apriori.
- **Benefits of the structure:**
  - **Completeness**: preserves **complete information** for frequent pattern mining.
  - **Compactness**: reduces irrelevant info (infrequent items) and is **never larger than the original database**.

**Disadvantages of FP-Growth:**
- **The FP-tree may not fit in memory.**
- **The FP-tree is expensive to build.**

**Performance comparison**: when the `minsup` threshold is **lowered** in Apriori, the computational time **increases EXPONENTIALLY**. With the FP-tree approach, the impact of lowering `minsup` is **significantly smaller**. This shows that `minsup` obviously influences the result, but **it is not a crucial element for FP-Growth as it is for Apriori**.

> **⚠️ NB: FP-growth is NOT an approximation** in terms of the frequent items it extracts — **it produces EXACTLY the same result as the Apriori algorithm. The only advantage is the faster way.**

---

## 6. ECLAT: Mining by Exploring Vertical Data Format

Another way to represent transactions and itemsets. **However, it is not necessarily more efficient than FP-Growth.**

In this representation, the dataset is described by considering, **for each item or itemset, the SET OF TRANSACTIONS in which it appears**.

> **Instead of storing:** transaction → items
> **we store:** **item → list of transactions**

**Vertical format:**
$$t(AB) = \{T_{11}, T_{25}, \dots\}$$
- **tid-list**: the list of **transaction IDs** containing an itemset.

**Deriving frequent patterns based on vertical intersections:**
- When we want to derive frequent patterns, we compute the **INTERSECTION** between the transaction sets of different itemsets. **If the cardinality of the resulting transaction set is ≥ minsup, the corresponding itemset is frequent.**
- $t(X) = t(Y)$ → **$X$ and $Y$ always appear together.**
- $t(X) \subset t(Y)$ → **transactions having $X$ always contain $Y$.**

**Using `diffset` to accelerate mining** — only keep track of the **differences** of tids:
- $t(X) = \{T_1,T_2,T_3\}$, $t(XY) = \{T_1,T_3\}$
- $\text{Diffset}(XY, X) = \{T_2\}$

**Algorithms**: **Eclat** (Zaki et al., KDD 1997); for mining **closed** patterns using vertical format: **CHARM**.

**How it works with Apriori-style generation:** we can also use the Apriori approach here — start from 1-itemsets and combine single items to generate 2-itemsets, intersecting their tid-lists.

> **Worked structure**: the vertical representation of the **1-itemsets** lists, for each item $I_1,\dots,I_5$, the corresponding set of transaction IDs containing it. **2-itemsets** are derived by computing the **intersection of the TID-sets of the individual items**. **3-itemsets** are obtained by intersecting the TID-sets of the corresponding **2-itemsets**. **The support of each itemset is the CARDINALITY of its TID-set.** Only itemsets satisfying `minsup` are kept.

> **Why it helps: this method avoids scanning the entire database repeatedly — it relies only on SET INTERSECTION operations between tid-lists to compute the support of itemsets.**

---

## 7. Pattern Evaluation Methods

### 7.1 Why confidence is not enough — the computer games example

10,000 customer transactions analyzed:
- **6,000** include **computer games**
- **7,500** include **videos**
- **4,000** include **both** computer games and videos

$$\text{buys}(X, \text{"computer games"}) \Rightarrow \text{buys}(X, \text{"videos"}) \qquad [\,support = 40\%,\ confidence = 66\%\,]$$

- **Support** = number of customers who bought computer games and videos together = $4000/10000 = 40\%$.
- **Confidence** = support of the rule / support of the antecedent = $4000/6000 = 66\%$.

> **⚠️ THE RULE IS MISLEADING**: the probability of purchasing videos **on its own is 75%** ($7500/10000$), which is **even LARGER than 66%**.
> **Computer games and videos are NEGATIVELY associated** — the purchase of one of these items actually **DECREASES** the likelihood of purchasing the other.

**The diagnosis**: the level of confidence is high, but it **doesn't really express that antecedent and consequent are correlated**. Confidence is appropriate just for verifying the conditional probability, **but it doesn't express CORRELATION**. It suffers from a structural problem: **if the consequent appears in every transaction, the confidence is always 1 — but that doesn't mean antecedent and consequent are really correlated.**

### 7.2 Lift

To solve this problem, the **lift** was introduced. It measures **how many times more often $X$ and $Y$ occur together than expected if they were statistically INDEPENDENT.**

$$lift(A\to B) = lift(B\to A) = \frac{conf(A\to B)}{supp(B)} = \frac{conf(B\to A)}{supp(A)} = \frac{P(A\cap B)}{P(A)\,P(B)}$$

**Interpretation** — assume $A$ and $B$ are **independent**. Then the joint probability equals the product of the probabilities, $P(A\cap B) = P(A)P(B)$, and therefore

$$lift(A\to B) = \frac{P(A\cap B)}{P(A)P(B)} = 1$$

> **The higher the value of the lift, the STRONGER the correlation between $A$ and $B$.** In particular, if the numerator $P(A\cap B)$ increases, the correlation increases — meaning the probability of observing $A$ and $B$ **together** is higher than what we would expect assuming independence.
>
> **Note the symmetry**: $lift(A\to B) = lift(B\to A)$ — unlike confidence, lift does **not** depend on rule direction.

**Properties**: lift is **not downward closed** and **does not suffer from the rare item problem**.
> **⚠️ But it has its own weakness**: rare itemsets with low counts (low probability) which **by chance** occur a few times (or only once) together **can produce ENORMOUS lift values.**

### 7.3 The basketball / cereal example

- $\text{play basketball} \Rightarrow \text{eat cereal}\ [40\%,\ 66.7\%]$ **is MISLEADING**.
- The **overall percentage of students eating cereal is 75% > 66.7%**.
- $\text{play basketball} \Rightarrow \text{NOT eat cereal}\ [20\%,\ 33.3\%]$ **is MORE ACCURATE**, although it has **lower support and confidence**.

### 7.4 Other correlation measures

Another metric already introduced is the **correlation via the $\chi^2$ (chi-square) test** — see **Chapter 3 §3.1** for the full treatment. There are many metrics for assessing correlation between antecedent and consequent, alternative to lift or $\chi^2$:

**All_confidence** — range $[0,1]$:
$$all\_conf(A,B) = \frac{sup(A\cup B)}{\max\{sup(A),\ sup(B)\}} = \min\{P(A\mid B),\ P(B\mid A)\}$$

**Max_confidence** — range $[0,1]$:
$$max\_conf(A,B) = \max\{P(A\mid B),\ P(B\mid A)\}$$

**Kulczynski** — range $[0,1]$:
$$Kulc(A,B) = \frac{1}{2}\big(P(A\mid B) + P(B\mid A)\big)$$

**Cosine** — range $[0,1]$:
$$cosine(A,B) = \frac{P(A\cup B)}{\sqrt{P(A)\times P(B)}} = \frac{sup(A\cup B)}{\sqrt{sup(A)\times sup(B)}} = \sqrt{P(A\mid B)\times P(B\mid A)}$$

> **The values of these metrics can be significantly influenced by the underlying DISTRIBUTION of the dataset.** Different data distributions may lead to **different interpretations** of the relationship between two items. To study this, we consider multiple datasets derived from a **contingency table** involving two items, varying the values to analyze how the metrics behave and assess their **robustness**.

### 7.5 Null-invariance — the crucial property

> **A measure is NULL-INVARIANT if its value is free from the influence of NULL-TRANSACTIONS** (transactions in which **neither** item appears). **This is an important property for measuring association patterns in large transaction databases.**
>
> - **$\chi^2$ and LIFT are NOT null-invariant**, because they **depend on the total number of transactions**, which includes null transactions. Consequently their values can be **significantly influenced** by changes in the number of such transactions.
> - **all_conf, max_conf, Kulczynski and cosine ARE null-invariant** — they have **no dependency on the total number of transactions**.

### 7.6 The six-dataset comparison

Contingency table notation: **$mc$** = both milk and coffee; **$m\bar c$** = milk only; **$\bar mc$** = coffee only; **$\bar m\bar c$** = **null transactions** (neither).

| Data | $mc$ | $m\bar{c}$ | $\bar{m}c$ | $\bar{m}\bar{c}$ | all_conf. | max_conf. | Kulc. | cosine | IR |
|---|---|---|---|---|---|---|---|---|---|
| **$D_1$** | 10000 | 1000 | 1000 | 100000 | 0.91 | 0.91 | 0.91 | 0.91 | 0.0 |
| **$D_2$** | 10000 | 1000 | 1000 | 100 | 0.91 | 0.91 | 0.91 | 0.91 | 0.0 |
| **$D_3$** | 100 | 1000 | 1000 | 100000 | 0.09 | 0.09 | 0.09 | 0.09 | 0.0 |
| **$D_4$** | 1000 | 1000 | 1000 | 100000 | 0.5 | 0.5 | 0.5 | 0.5 | 0.0 |
| **$D_5$** | 1000 | 100 | 10000 | 100000 | 0.09 | 0.91 | 0.5 | 0.29 | 0.89 |
| **$D_6$** | 1000 | 10 | 100000 | 100000 | 0.01 | 0.99 | 0.5 | 0.10 | 0.99 |

**Analysis of each dataset:**

**$D_1$** — Coffee and milk appear together 10,000 times out of the 11,000 transactions in which coffee appears, and similarly milk appears 10,000 times out of the 11,000 in which it occurs. A **higher $\chi^2$** indicates a stronger likelihood of correlation; the **lift ≈ 9.26**, suggesting a **strong positive correlation**. The other metrics (0 = independence → 1 = strong correlation) also take **high values (0.91)**, consistently indicating a **strong relationship**.

**$D_2$** — The first three columns are **unchanged** (same number of co-occurrences); **the ONLY difference is the number of NULL transactions** (100 instead of 100,000). Now **$\chi^2 = 0$ and lift = 1**, both suggesting **independence**! **But the other metrics still indicate strong correlation (0.91).**
> **⚠️ This is the decisive demonstration**: $\chi^2$ and lift are **affected by null transactions**, whereas the other metrics are **null-invariant**.

**$D_3$** — The number of transactions in which the two items appear together **also changes** (only 100). We can conclude **there is NO correlation**, since the co-occurrence count is **low compared to** the individual appearance counts. Here too, $\chi^2$ and lift are affected by the null transactions, which are more numerous than the co-occurrences. **The other metrics also indicate absence of correlation (0.09).**

**$D_4$** — Same value for co-occurrences and for individual appearances (1000/1000/1000). **Lift and $\chi^2$ express high correlation because of the null transactions.** Using the other metrics we find an **intermediate situation (0.5)** — a genuinely balanced, neutral case.

**$D_5$** — A **highly UNBALANCED** situation: 1000 transactions with coffee and milk together, 100 with only coffee, 10,000 with only milk, plus ~100,000 null transactions. Overall coffee appears 1,100 times.
> **From the perspective of COFFEE there is a strong correlation with milk; from the perspective of MILK there is NO significant correlation with coffee.**
>
> $\chi^2$ and lift take high values, mainly due to the large number of null transactions. The remaining metrics **disagree with each other**: **all_conf indicates independence (0.09)**, **max_conf suggests correlation (0.91)**, **Kulczynski indicates an intermediate situation (0.5)**, and **cosine suggests something closer to independence (0.29)**.
> **→ Kulczynski appears the MOST INFORMATIVE here, because it highlights that a DEFINITIVE CONCLUSION CANNOT BE DRAWN — the interpretation depends on the perspective adopted (coffee's or milk's).**

**$D_6$** — The number of transactions with only coffee **decreases** (10) while those with only milk **increases** (100,000). The **lift value decreases**, since the transactions containing only milk become significantly larger. Again: **all_conf → independence (0.01)**, **max_conf → correlation (0.99)**, **Kulczynski → intermediate (0.5)**, **cosine → independence (0.10)**. As before, **Kulczynski is the most reliable**, reflecting the ambiguity and highlighting that the conclusion depends on the chosen perspective.

### 7.7 The Imbalance Ratio (IR)

> **⚠️ The remaining problem**: **the Kulczynski measure takes THE SAME VALUE (0.5) in two fundamentally DIFFERENT scenarios.** In $D_4$ that value corresponds to a **balanced** situation (co-occurrences comparable to individual occurrences). In $D_5$ and $D_6$ the same value arises in **highly unbalanced** situations.
>
> **For this reason we introduce an additional metric that explicitly captures IMBALANCE**, so Kulczynski can be used **in combination** with an imbalance indicator to distinguish balanced from unbalanced cases.

**IR (Imbalance Ratio)** — measures the imbalance of two itemsets $A$ and $B$ in rule implications:

$$IR(A,B) = \frac{|sup(A) - sup(B)|}{sup(A) + sup(B) - sup(A\cup B)}$$

> **⚠️ Why Kulczynski can be 0.5 despite imbalance**: depending on which item is more frequent, the conditional probability $P(A\mid B)$ can be **close to 1** while $P(B\mid A)$ is **close to 0** (or vice versa). The joint probability stays the same, but when it is **normalized by a SMALL marginal probability** (that of the **less frequent** item), the resulting conditional probability can be close to 1. Averaging a near-1 with a near-0 gives ≈ 0.5.

**Kulczynski and IR together give a clear picture for $D_4$–$D_6$:**
- **$D_4$ is BALANCED and neutral** (IR = 0.0)
- **$D_5$ is IMBALANCED and neutral** (IR = 0.89)
- **$D_6$ is VERY IMBALANCED and neutral** (IR = 0.99)

> **⚠️⚠️ TO SUM UP: if we want to properly evaluate whether there is a correlation between antecedent and consequent in an association rule, we should use the KULCZYNSKI measure TOGETHER WITH the IMBALANCE RATIO (IR). The LIFT measure is affected by NULL TRANSACTIONS and may provide misleading results that differ from the expected interpretation.**

---

## 8. Summary

**Scalable frequent pattern mining methods:**
- **Apriori** — candidate generation and test
- **Projection-based** — FP-Growth, CLOSET+, …
- **Vertical format approach** — ECLAT, CHARM, …

**Which patterns are interesting?** → **Pattern evaluation methods** (§7).

---

## 9. Key points / potential exam pitfalls

### Basics
- **Support = usefulness/relevance; Confidence = certainty.** Keep these two roles separate — it's the framing question the whole chapter builds on.
- **Support is SYMMETRIC, confidence is NOT.** Swapping antecedent and consequent leaves support unchanged but **can change the confidence** (Beer⇒Diaper 100% vs Diaper⇒Beer 75%). Lift, by contrast, is symmetric again.
- **⚠️ Confidence = 1 does NOT mean correlation.** If the consequent appears in essentially every transaction, confidence is trivially high — this is *the* recurring trap of the chapter and the reason lift exists.
- **Only the PRESENCE of items matters, not quantities** — each item is a Boolean variable, each basket a Boolean vector.
- **Association rule mining is a TWO-step process**: (1) find frequent itemsets, (2) generate strong rules from them. Almost all the computational cost sits in step 1.

### Closed vs maximal
- **Both compress the output, but only CLOSED is LOSSLESS.** Maximal itemsets lose the support information of subsets.
- **The definitions differ in one word**: *closed* = no proper superset with the **SAME SUPPORT**; *maximal* = no **FREQUENT** proper superset. Every maximal itemset is closed, but not vice versa.
- **$supp(\text{superset}) \le supp(\text{subset})$ always** — a superset can never have higher support. Spotting the "impossible" case is a classic exam check.
- **The $\langle a_1..a_{100}\rangle$ / $\langle a_1..a_{50}\rangle$ example is the canonical illustration**: closed gives you $\{a_2,a_{45}\}:2$; maximal only lets you infer support 1.

### Apriori
- **The Apriori property is a NECESSARY condition**, stated as: every non-empty subset of a frequent itemset is frequent. Its useful contrapositive is the pruning rule: **if a subset is infrequent, every superset is infrequent.**
- **"Antimonotonicity" is named from the FAILING direction** — if a set fails the test, all its supersets fail too.
- **The join rule is precise**: two $(k-1)$-itemsets are joinable **iff their first $(k-2)$ items are identical AND $l_1[k-1] < l_2[k-1]$**. The lexicographic ordering exists **specifically to avoid generating duplicates**.
- **Join and prune are two distinct steps.** The join produces candidates; the prune removes any candidate having an infrequent $(k-1)$-subset — *before* the (expensive) database scan.
- **Termination happens when $C_k = \emptyset$**, as in the worked example where $\{I_1,I_2,I_3,I_5\}$ is pruned because $\{I_2,I_3,I_5\}$ isn't frequent.
- **Apriori's cost is the repeated database scanning plus candidate generation** — one scan per level $k$.

### FP-Growth
- **Exactly TWO database scans, and NO candidate generation** — these are the two headline differences from Apriori.
- **⚠️ FP-Growth is NOT an approximation.** It returns **exactly the same** frequent patterns as Apriori — only faster (about an order of magnitude).
- **Items are ordered by DECREASING support so that transactions share common PREFIXES near the root** → path overlap → compression. This ordering is a **heuristic**: it does **not** guarantee the smallest possible tree, and that's acceptable as long as the tree fits in memory.
- **Best case = 1 path (all transactions identical). Worst case = every transaction unique**, where the FP-tree is **larger** than the original data (counters + pointers add overhead).
- **⚠️ The counter rule is the single most error-prone detail**: in the **original** FP-tree and in extracted **prefix paths**, counters stay **unchanged**; in the **conditional** FP-tree, counters are **recomputed** to reflect only transactions containing the conditioning item. A node showing count 2 in a prefix path for $e$ may be shared with a branch where $e$ never appears.
- **Extraction is BOTTOM-UP, starting from the LEAST frequent item** (divide and conquer, depth-first), building conditional pattern bases → conditional FP-trees → recursion.
- **Lowering minsup blows up Apriori exponentially, but affects FP-Growth much less.**

### ECLAT / vertical format
- **The representation flips**: instead of *transaction → items*, store *item → tid-list*.
- **Support = CARDINALITY of the tid-list**; candidate support is computed by **intersecting** tid-lists — **no repeated database scans**.
- $t(X)=t(Y)$ → always co-occur; $t(X)\subset t(Y)$ → transactions with $X$ always contain $Y$.
- **Diffsets** store only the *differences* between tid-lists to accelerate mining.
- **ECLAT is not necessarily faster than FP-Growth** — it's an alternative, not a strict improvement.

### Pattern evaluation ⭐
- **The computer-games example is the canonical "confidence lies" case**: confidence 66% looks decent until you notice the baseline probability of the consequent is **75%** — so the association is actually **NEGATIVE**.
- **Lift = 1 → independence; > 1 → positive correlation; < 1 → negative.** Lift is **symmetric** and immune to the rare-item problem — **but** rare itemsets co-occurring by chance can produce **enormous, spurious lift values**.
- **⚠️ NULL-INVARIANCE is the central concept of §7.** $\chi^2$ and **lift are NOT null-invariant** (they depend on the total transaction count, which includes null transactions); **all_conf, max_conf, Kulczynski and cosine ARE.** $D_1$ vs $D_2$ is the proof: the co-occurrence counts are identical, only the null transactions change, and lift swings from 9.26 to 1 while the null-invariant measures stay at 0.91.
- **all_conf = min of the two conditional probabilities; max_conf = max; Kulczynski = their average; cosine = their geometric mean.** Knowing this makes the $D_5$/$D_6$ disagreement obvious: with one conditional near 1 and the other near 0, min→0, max→1, mean→0.5, geometric mean→small.
- **⚠️ Kulczynski alone is insufficient**: it returns **0.5 both for a genuinely BALANCED neutral case ($D_4$) and for highly IMBALANCED ones ($D_5$, $D_6$)**. That is exactly why **IR** is added.
- **FINAL RECOMMENDATION: use KULCZYNSKI TOGETHER WITH the IMBALANCE RATIO.** This is the chapter's concluding prescription and a very likely exam question.
- **Support suffers from the RARE ITEM PROBLEM** (meaningful rules about infrequent items get pruned); **confidence suffers from sensitivity to the CONSEQUENT's frequency.** Different weaknesses, different fixes.

### Cross-chapter connections
- **The $\chi^2$ test** used here as a correlation measure is fully developed in **Chapter 3 §3.1** (contingency tables, expected counts, degrees of freedom, critical values). Its **lack of null-invariance** is a new criticism introduced only here.
- **"Correlation does not imply causality"** was first stated in **Chapter 3** (hospitals vs. car thefts) and is restated here as a headline warning.
- **Associative classification** (**Chapter 4 §7** — CBA, CMAR, CPAR) is the direct application of this chapter's machinery to classification; the professor deferred it there precisely because frequent patterns hadn't yet been covered.
- **CLIQUE** (**Chapter 5 §6.2**) uses the **same Apriori principle** for pruning candidate dense units across dimensions — including the same caveat that the property is **necessary but not sufficient**, so candidates must still be verified by counting.
- **Clustering as a preprocessing tool for association analysis** was listed in **Chapter 5 §1**.

---

*File auto-generated from `6 - Associative Rule.pdf` (the two PDFs in this folder have identical content). For questions about this chapter, refer only to this file.*
