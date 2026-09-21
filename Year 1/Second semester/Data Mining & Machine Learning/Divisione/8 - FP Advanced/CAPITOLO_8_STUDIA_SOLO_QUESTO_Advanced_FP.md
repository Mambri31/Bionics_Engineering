# Chapter 8 — Advanced Frequent Pattern Analysis

> **Course:** Data Mining and Machine Learning / Bioinspired Computational Methods — Biological Data Mining
> **Instructor:** Prof. Francesco Marcelloni (University of Pisa)
> **Sources merged into this file:** professor's slides (`8-FPAdvanced.pdf`, 32 pages — Chapter 7 of Han–Kamber–Pei) + lecture notes of 10/03 and the first part of 12/03 (`8 - FPAdvanced sbobine.pdf`, 27 pages)
>
> **Instructions for Claude:** this is the single reference file for Chapter 8. When the user asks study questions about this chapter, use **only** this file as context (no need to re-read the original PDFs unless a detail is missing). The file merges the official slide content with the lecture's discursive explanations and worked examples, keeping both levels of detail.

---

## Chapter outline (official slide structure)
1. Pattern Mining in Multi-Level, Multi-Dimensional Space
   - Mining Multi-Level Association
   - Mining Multi-Dimensional Association
   - Mining Quantitative Association Rules
   - Mining Rare Patterns and Negative Patterns
2. Constraint-Based Frequent Pattern Mining
3. Mining High-Dimensional Data and Colossal Patterns
4. Mining Compressed or Approximate Patterns
5. Summary

> **What the lecture emphasised**: the two blocks that matter most in this chapter are **Constraint-Based Frequent Pattern Mining** and **Mining High-Dimensional Data and Colossal Patterns**. The rest is important context, but those two are where the real algorithmic ideas live. The general theme of the chapter: extend basic mining to **complex datasets**, and above all **reduce computational complexity** — the point is not "apply another algorithm", it is **how you set up the problem so that the search space collapses**.

---

## 0. Prerequisite recap (from the lecture opening)

Frequent pattern discovery rests on two foundational algorithms that use **different methodologies but produce identical results**:

| | Apriori | FP-Growth |
|---|---|---|
| **Idea** | Level-wise (breadth-first) candidate generate-and-test | Compressed prefix tree (FP-tree) + recursive projection (depth-first) |
| **Pros** | Highly intuitive, easy to implement | Significantly faster; **only two database scans** |
| **Cons** | Computationally expensive; **many full scans of the DB** | More complex to implement; **more memory** (the tree must be stored) |

**The critical parameter is `min_sup`** — the minimum percentage (or absolute count) of transactions in which an itemset must appear to be called *frequent*.

> **The `min_sup` trade-off** (lecture): a **low `min_sup` causes combinatorial explosion** — a huge number of patterns and a long processing time. A **high `min_sup`** makes the process efficient **but can miss interesting insights**. **The ideal threshold is not fixed: it must be calibrated on the application domain** (retail vs. bioinformatics behave completely differently). This tension is the thread running through the whole chapter.

---

## 1. Pattern Mining in Multi-Level, Multi-Dimensional Space

### 1.1 Mining Multiple-Level Association Rules

**Items often form hierarchies** (concept hierarchies): `Milk → 2% Milk → 2% Foremost Milk`. Mining can therefore be done at several **levels of abstraction**.

Two key facts:
- **Items at the lower level are expected to have LOWER support** (they are more specific).
- Therefore we need **flexible support settings** across levels (shared multi-level mining: Agrawal & Srikant @ VLDB'95, Han & Fu @ VLDB'95).

#### Uniform support vs. reduced support — the worked example

```
                        UNIFORM SUPPORT                     REDUCED SUPPORT
Level 1                 min_sup = 5%                        min_sup = 5%
                        Milk [support = 10%]  -> frequent   Milk [support = 10%]  -> frequent

Level 2                 min_sup = 5%                        min_sup = 3%
                        2% Milk   [support = 6%] -> frequent    2% Milk   [6%] -> frequent
                        Skim Milk [support = 4%] -> LOST!       Skim Milk [4%] -> frequent
```

> **Why uniform support is problematic** (lecture): with a single threshold, a general category like *Milk* is easily frequent, but a specific sub-category like *Skim Milk* (4%) **fails to reach the threshold** and disappears. To explore specialised items we must adopt **reduced (varying) support at lower levels**. This lets us keep the valuable high-level abstractions **without losing the details** of less frequent but significant sub-items.
>
> **NB**: some items are **intrinsically more valuable despite being less frequent**. A fixed threshold across all levels leads to **"over-pruning" of significant specialised data**.

#### Flexible min-support thresholds (group-based)

- Some items are **more valuable but less frequent** → use **non-uniform, group-based min-support**.
- E.g. `{diamond, watch, camera}: 0.05%` while `{bread, milk}: 5%`.

> **The implementation trick the professor gave**: if you lower the *global* threshold to catch rare items you get combinatorial explosion. Instead of scanning the whole dataset with a very low threshold, **isolate the transactions containing the specific items of interest and run Apriori / FP-Growth only on that subset**. Same result for those items, a fraction of the cost.

#### Redundancy filtering

Some rules are **redundant because of "ancestor" relationships** between items:

```
milk     => wheat bread   [support = 8%, confidence = 70%]      <-- ancestor rule
2% milk  => wheat bread   [support = 2%, confidence = 72%]      <-- descendant rule
```

**Definition**: a rule is **redundant if its support and confidence are close to the "expected" value based on the rule's ancestor.**

> **How to read this** (lecture): 2% milk is roughly a quarter of all milk sales, and indeed $8\% \times \tfrac{1}{4} \approx 2\%$; the confidence barely moves (70% → 72%). The specific rule adds nothing **surprising**, so it can be filtered out. Only descendant rules that **deviate** from the ancestor's expectation are worth reporting.
>
> **Reminder on confidence**: $\text{conf}(A \Rightarrow B) = P(B \mid A)$, the conditional probability of the consequent given the antecedent. High confidence matters because it signals a **strong predictive relationship**.

### 1.2 Mining Multi-Dimensional Association Rules

- **Single-dimensional rules** — one predicate, repeated:
  $$\text{buys}(X, \text{“milk”}) \Rightarrow \text{buys}(X, \text{“bread”})$$
- **Multi-dimensional rules** — $\ge 2$ dimensions/predicates. Two sub-types:

| Type | Definition | Example |
|---|---|---|
| **Inter-dimensional** | **no repeated predicates** | $\text{age}(X, \text{“19-25”}) \wedge \text{occupation}(X, \text{“student”}) \Rightarrow \text{buys}(X, \text{“coke”})$ |
| **Hybrid-dimensional** | **repeated predicates** | $\text{age}(X, \text{“19-25”}) \wedge \text{buys}(X, \text{“popcorn”}) \Rightarrow \text{buys}(X, \text{“coke”})$ |

> **Processing method** (lecture — this is the practical point): treat **each attribute–value pair as a single item** (e.g. `age=19-25` becomes one item). Once the data is transformed into this item-based format, **standard Apriori / FP-Growth apply unchanged**. No new algorithm is needed, only a re-encoding of the data.

### 1.3 Mining Quantitative Associations

Attributes come in two flavours:

- **Categorical attributes**: finite number of values, **no ordering** (colour, brand) → handled with the **data-cube approach**.
- **Quantitative attributes**: numeric, **implicit ordering** among values (age, salary) → must be **discretized** before mining.

**Four families of techniques** (classified by how numeric attributes are treated):

1. **Static discretization** based on **predefined concept hierarchies** (data-cube methods) — e.g. group ages into "Young", "Adult", "Senior".
2. **Dynamic discretization** based on the **data distribution** (quantitative association rules — Agrawal & Srikant @ SIGMOD'96): binning.
3. **Clustering: distance-based association** (Yang & Miller @ SIGMOD'97) — **one-dimensional clustering first, association afterwards**.
4. **Deviation** (Aumann & Lindell @ KDD'99) — report a rule when the outcome deviates significantly from the norm:
   $$\text{Sex} = \text{female} \Rightarrow \text{Wage: mean} = \$7/\text{hr} \quad (\text{overall mean} = \$9/\text{hr})$$

### 1.4 Mining Rare Patterns and Negative Patterns

#### Rare patterns
- **Very low support but interesting** — e.g. buying Rolex watches.
- **Mining strategy**: set **individual-based or group-based support thresholds** for valuable items.

> **Why not simply lower `min_sup` globally** (lecture): a globally low threshold produces a **massive increase of irrelevant patterns**. Assign the low threshold **only to the specific high-value item categories**.

#### Negative (correlated) patterns
- Since it is unlikely that someone buys a **Ford Expedition** (large SUV) and a **Toyota Prius** (hybrid) together, those two are likely **negatively correlated**.
- **⚠️ Negatively correlated patterns that are INFREQUENT tend to be MORE interesting than those that are frequent.**

#### Definition 1 (support-based) — and why it FAILS

> If itemsets $X$ and $Y$ are both frequent but rarely occur together, i.e.
> $$\text{sup}(X \cup Y) < \text{sup}(X) \times \text{sup}(Y)$$
> then $X$ and $Y$ are negatively correlated.

(In probability terms: the joint probability is lower than it would be under independence.)

**The counter-example (needle packages), worked fully:**

A store sold two "needle 100" packages $A$ and $B$; **only one transaction contains both** $A$ and $B$.

- **Case A — 200 transactions total** ($A$ in 100 of them, $B$ in 100 of them):
  $$s(A \cup B) = \frac{1}{200} = 0.005, \qquad s(A)\cdot s(B) = 0.5 \times 0.5 = 0.25$$
  $$0.005 < 0.25 \;\Rightarrow\; \textbf{negatively correlated} \;\;\checkmark$$
- **Case B — $10^5$ transactions total** (same 100 occurrences of $A$, 100 of $B$, same single co-occurrence):
  $$s(A \cup B) = \frac{1}{10^5} = 10^{-5}, \qquad s(A)\cdot s(B) = \frac{1}{10^3}\cdot\frac{1}{10^3} = 10^{-6}$$
  $$10^{-5} > 10^{-6} \;\Rightarrow\; \textbf{positively correlated} \;\;\text{(!!)}$$

> **⚠️ Where is the problem? NULL TRANSACTIONS.** Nothing changed about $A$ and $B$ themselves — only the number of transactions containing **neither** of them. The support-based definition is **not null-invariant**, so its verdict flips from "negative" to "positive" purely because the database grew with irrelevant transactions.

#### Definition 2 (Kulczynski measure-based) — the fix

> If itemsets $X$ and $Y$ are frequent, but
> $$\frac{P(X \mid Y) + P(Y \mid X)}{2} < \varepsilon$$
> where $\varepsilon$ is a **negative pattern threshold**, then $X$ and $Y$ are **negatively correlated**.

**Why it works**: the measure is built from **conditional probabilities only**, so it is **null-invariant** — adding transactions that contain neither $X$ nor $Y$ does not change $P(X\mid Y)$ or $P(Y\mid X)$.

**Same needle example, with `min_sup = 0.01%` and $\varepsilon = 0.02$:**
$$P(A \mid B) = \frac{1}{100} = 0.01, \qquad P(B \mid A) = \frac{1}{100} = 0.01 \;\Rightarrow\; \frac{0.01+0.01}{2} = 0.01 < 0.02$$
**Negatively correlated in BOTH cases** — whether the database has 200 or $10^5$ transactions. The verdict is now stable.

---

## 2. Constraint-Based (Query-Directed) Frequent Pattern Mining

### 2.1 Motivation

- **Finding all patterns in a database autonomously is unrealistic** — the patterns are **too many and not focused**.
- **Data mining should be an INTERACTIVE process**: the user directs what is mined through a **data mining query language or a GUI**.

**Two benefits of constraint-based mining:**
- **User flexibility**: the user provides constraints on what to mine.
- **Optimization**: the system **exploits the constraints for efficient mining**.

> **⚠️ Crucial precision from the slides**: constraint-based mining still finds **ALL the answers satisfying the constraints** — it is **not** a heuristic search that returns *some* answers. Completeness is preserved; only the search space shrinks.

### 2.2 The five types of constraints

| # | Constraint type | Meaning | Example |
|---|---|---|---|
| 1 | **Knowledge type** | which task | classification, association, … |
| 2 | **Data constraint** | SQL-like selection | *find product pairs sold together in stores in **Chicago** this **year*** |
| 3 | **Dimension/level** | relevance of dimensions / abstraction levels | region, price, brand, customer category |
| 4 | **Rule (pattern) constraint** | structural requirement on the rule | *small sales (price < \$10) triggers big sales (sum > \$200)* |
| 5 | **Interestingness** | statistical thresholds | strong rules: `min_support ≥ 3%`, `min_confidence ≥ 60%` |

### 2.3 Meta-rule guided mining

A **metarule** forms a **hypothesis** about the relationships the user wants to probe or confirm. It is a rule with **partially instantiated predicates and constants**:

$$P_1(X, Y) \wedge P_2(X, W) \Rightarrow \text{buys}(X, \text{“iPad”})$$

*"We are interested in determining which type of customer buys an iPad."* The resulting instantiated rule could be:

$$\text{age}(X, \text{“15-25”}) \wedge \text{profession}(X, \text{“student”}) \Rightarrow \text{buys}(X, \text{“iPad”})$$

In general: $P_1 \wedge P_2 \wedge \dots \wedge P_l \Rightarrow Q_1 \wedge Q_2 \wedge \dots \wedge Q_r$.

### 2.4 Post-processing vs. constraint pushing

> **The central question of the whole section** (slides): *"How can we use rule constraints to prune the search space? More specifically, **what kind of rule constraints can be 'pushed' deep into the mining process and still ensure the completeness** of the answer returned for a mining query?"*

- **Post-processing**: run Apriori/FP-Growth to completion, then filter. Dimension/level and interestingness constraints **can** be applied this way, **but it is generally more efficient and less expensive to use them DURING mining** — post-processing is often unfeasible because of **pattern explosion** (too many intermediate patterns get generated anyway).
- **Constraint pushing (optimized)**: integrate the constraints **inside** the mining process, at the transaction or candidate-generation level.

**Two orthogonal kinds of pruning:**

| | What it checks | Decision |
|---|---|---|
| **Pattern search space pruning** | candidate **patterns** | can this pattern (and its descendants) be pruned? |
| **Data search space pruning** | the **data set / transactions** | can this piece of data still contribute to generating satisfiable patterns? |

**Taxonomy of the properties exploited:**

- *Pattern space pruning constraints*
  - **Anti-monotonic**: if $c$ is violated, further mining of that branch **can be terminated**.
  - **Monotonic**: if $c$ is satisfied, **no need to check $c$ again**.
  - **Succinct**: $c$ must be satisfied, so one can **start directly from the data sets satisfying $c$**.
  - **Convertible**: $c$ is neither monotonic nor anti-monotonic, **but becomes one of the two if the items are properly ordered**.
- *Data space pruning constraints*
  - **Data succinct**: the data space can be pruned **at the very start** of mining.
  - **Data anti-monotonic**: if a transaction $t$ does not satisfy $c$, **$t$ can be pruned from further mining**.

### 2.5 The reference TDB used in all the constraint examples

```
TDB (min_sup = 2)                Item profits
TID   Transaction                a: 40    b:   0
10    a, b, c, d, f              c: -20   d:  10
20    b, c, d, f, g, h           e: -30   f:  30
30    a, c, d, e, f              g: 20    h: -10
40    c, e, f, g
```

### 2.6 Anti-monotonicity

> **Definition**: a constraint $C$ is **anti-monotone** if, whenever a **super-pattern satisfies $C$, all of its sub-patterns satisfy it too**. Equivalently: **if an itemset $S$ violates $C$, so does every superset of $S$.**

**Optimization**: if an itemset violates the constraint, **terminate mining of all its descendants**. *(If a pattern is "bad", making it bigger only makes it worse.)*

**Examples:**
1. $\text{sum}(I.\text{price}) \le v$ — **anti-monotone** (adding items can only increase a sum of non-negative prices).
2. $\text{range}(I.\text{profit}) \le 15$ — **anti-monotone**. Itemset $ab$ has profits $\{40, 0\}$, so $\text{range} = 40 > 15$ → **violates $C$**; every superset still contains $a$ and $b$, so its range is **at least 40** → **every superset violates $C$ too**.
3. $\text{sum}(I.\text{price}) \ge v$ — **NOT anti-monotone**: adding further items can push the sum above $v$, so a violating itemset may have a satisfying superset.
4. **Support count is anti-monotone** — **this is the core property used by Apriori**. If an itemset has support below `min_sup`, no superset of it can be frequent. *(If milk is not frequent, `{milk, bread}` cannot be frequent.)*

### 2.7 Monotonicity

> **Definition**: a constraint $C$ is **monotone** if, whenever a pattern satisfies $C$, **all of its supersets satisfy it too** — so **we do not need to check $C$ again** in subsequent mining.

**Examples:**
1. $\text{sum}(I.\text{price}) \ge v$ — **monotone**. If the price of a diamond alone is already over $v$, **any set containing that diamond satisfies the constraint**.
2. $\text{min}(I.\text{price}) \le v$ — **monotone**. Adding an item **cannot increase** the minimum price of the set, so once satisfied, always satisfied.
3. $C: \text{range}(I.\text{profit}) \ge 15$ — **monotone**. Itemset $ab$ (range 40) satisfies $C$, and so does every superset of $ab$.

> **The mental picture**: anti-monotone = *"once broken, always broken"* → prune downward. Monotone = *"once satisfied, always satisfied"* → **stop testing** (saves computation, but does not prune the space).

### 2.8 Succinctness

> **Definition**: a constraint is **succinct** if we can **enumerate all and only those sets guaranteed to satisfy it** — i.e. we can **generate precisely the satisfying sets even before support counting begins**.

Formally: given $A_1$, the set of **items** satisfying a succinctness constraint $C$, **any set $I$ satisfying $C$ is based on $A_1$** — $I$ contains a subset belonging to $A_1$.

**The idea**: **without looking at the transaction database**, whether an itemset $I$ satisfies $C$ can be decided from the **selection of items alone**.

**Optimization**: **if $C$ is succinct, $C$ is PRE-COUNTING PUSHABLE** — this avoids the substantial overhead of the generate-and-test paradigm (constraints are *pre-counting prunable*).

**Examples:**
- $\text{min}(I.\text{price}) \ge v$ — **succinct**: there is a precise formula for the satisfying sets (all itemsets built only from items with price $\ge v$). Simply **remove every item with price $< v$ before starting the algorithm**.
- $\text{sum}(I.\text{price}) \ge v$ — **NOT succinct**: you cannot decide membership item-by-item; you must check the actual combination.

#### Worked example: standard Apriori vs. Apriori with a succinct constraint pushed deep

**Database D**

| TID | Items |
|---|---|
| 100 | 1, 3, 4 |
| 200 | 2, 3, 5 |
| 300 | 1, 2, 3, 5 |
| 400 | 2, 5 |

**Item prices**: `1: 0.5`, `2: 1`, `3: 4`, `4: 0.3`, `5: 4.5`. `min_sup = 2`.

**(a) Plain Apriori**

- **Scan 1 ($C_1 \to L_1$)**: supports `{1}:2, {2}:3, {3}:3, {4}:1, {5}:3`. **Item 4 is dropped** (support 1 < 2) — since it is not frequent, it **cannot be part of any larger frequent itemset**. → $L_1 = \{1\},\{2\},\{3\},\{5\}$.
- **Scan 2 ($C_2 \to L_2$)**: candidate pairs from $L_1$, counted against the DB: `{1,2}:1, {1,3}:2, {1,5}:1, {2,3}:2, {2,5}:3, {3,5}:2`. → $L_2 = \{1,3\},\{2,3\},\{2,5\},\{3,5\}$.
- **Scan 3 ($C_3 \to L_3$)**: joining $\{2,3\}$ and $\{2,5\}$ gives the candidate $\{2,3,5\}$; its support is **2** (it appears in TID 200 and 300) → **frequent**. $L_3 = \{2,3,5\}$.

**(b) Apriori with the succinct constraint $\text{min}\{I.\text{price}\} \ge 3$ pushed in**

Only items with price $\ge 3$ can appear: **items 3 (4.0) and 5 (4.5)**. Items **1, 2, 4 are eliminated before the first support count**.

> **The saving** (lecture): instead of generating all frequent items and then filtering, we **"push" the constraint into the data-selection phase**. The algorithm starts only with the items **guaranteed to satisfy the structural requirement**, so the candidate sets it must evaluate collapse to $\{3\}, \{5\}, \{3,5\}$. **This pruning happens BEFORE any counting** — that is exactly what "pre-counting pushable" means.

### 2.9 Convertible constraints

Some constraints — the classic example being the **average** — are **neither monotone nor anti-monotone**. But they **become one of the two if the items are properly ordered**.

**Example: $C: \text{avg}(S.\text{profit}) \ge 25$.**

Order items in **value-descending order** $R$:
$$\langle a(40),\; f(30),\; g(20),\; d(10),\; b(0),\; h(-10),\; c(-20),\; e(-30) \rangle$$

- Appending any later item can **only decrease** the average.
- So if an itemset $afb$ **violates** $C$, so do $afbh$, $afb*$, … → **$C$ becomes ANTI-MONOTONE with respect to $R$**.
- With the **ascending** order $R^{-1} = \langle e, c, h, b, d, g, f, a \rangle$, appending later items can only **increase** the average → **$C$ becomes MONOTONE**.

Since $\text{avg}(X) \ge 25$ is convertible **both** anti-monotone (w.r.t. $R$) **and** monotone (w.r.t. $R^{-1}$), it is called **strongly convertible**.

> **Precisely**: "$\text{avg}(X) \ge 25$ is convertible anti-monotone w.r.t. $R$" means *if an itemset $af$ violates $C$, so does every itemset having $af$ as a prefix* (e.g. $afd$). "Convertible monotone w.r.t. $R^{-1}$" means *if an itemset satisfies $C$, so does every itemset having it as a prefix* (e.g. from $f$ (30) we get $fa$ (avg 35), still satisfying). **The prefix is what matters** — that is why this only works inside a projection-based (prefix-path) framework.

#### ⚠️ Why Apriori CANNOT handle convertible constraints

> **The lecture's explanation, in full**: Apriori is **level-wise** — it builds itemsets of size $k$ by **joining frequent itemsets of size $k-1$**, relying on *"if a subset is not frequent, the superset cannot be frequent"*. For a constraint like $\text{avg}(X) \ge 25$ this logic **does not hold**.
>
> Take the itemset $\{d, f\}$: profits 10 and 30, average 20 → it **violates** $C$. But add $a$ (profit 40): $\{a, d, f\}$ has average $\tfrac{40+10+30}{3} \approx 26.7$ → it **satisfies** $C$. **If Apriori pruned $\{d,f\}$ at level 2 because it violated the constraint, it could never assemble $\{a,d,f\}$ at level 3.** Within the level-wise framework, **no direct pruning based on such a constraint can be made** without risking the loss of valid answers.

#### …but FP-Growth CAN

FP-Growth does **not** use the level-wise join; it builds a tree and explores patterns through **prefix paths**. Order the items in the FP-tree by **descending profit** and the "average" behaves like an **anti-monotone** constraint: since any subsequent item from the header table can only **decrease** the average, **if the current average already violates the threshold, the whole branch can be safely pruned**.

**Projection-based mining, worked on the reference TDB:**

- $C: \text{avg}(X) \ge 25$, `min_sup = 2`.
- List items in every transaction in value-descending order $R = \langle a, f, g, d, b, h, c, e\rangle$; $C$ is convertible anti-monotone w.r.t. $R$.
- **Scan the TDB once and remove infrequent items** → **item $h$ is dropped** (support 1).
- Itemsets $a$ and $f$ are "good" starting points, …

```
TDB after projection & ordering (min_sup = 2)
TID   Transaction
10    a, f, d, b, c
20    f, g, d, b, c
30    a, f, d, c, e
40    f, g, c, e
```

> *(The slide prints `40  f, g, h, c, e`, but $h$ was just dropped as infrequent and does not occur in T40 in the original TDB — treat it as a typo in the deck.)*

**Take-away**: *many tough constraints can be converted into (anti-)monotone ones by imposing an appropriate order on item projection.*

### 2.10 Handling multiple constraints

Different constraints may require **different or even conflicting item orderings** (e.g. $C_1: \text{avg}(\text{profit}) \ge 25$ wants descending, $C_2: \text{sum}(\text{price}) \le 100$ may want ascending).

- **Compatible case**: if there exists **one order $R$ such that both $C_1$ and $C_2$ are convertible w.r.t. $R$**, there is **no conflict** — push both into the same FP-tree simultaneously.
- **Conflicting case** — the strategy from the lecture:
  1. **Priority mining**: satisfy **one constraint first** (typically the most restrictive one, or the one that is strictly anti-monotone/succinct).
  2. **Projected-database mining**: then **switch to the ordering required by the second constraint** and mine frequent itemsets in the **corresponding projected databases**.
  3. **Post-filtering**: if a constraint really cannot be pushed because of the conflict, apply it as a **filter during the recursive steps** of FP-Growth.

### 2.11 What constraints are convertible?

| Constraint | Convertible anti-monotone | Convertible monotone | Strongly convertible |
|---|---|---|---|
| $\text{avg}(S) \le v$ , $\ge v$ | Yes | Yes | **Yes** |
| $\text{median}(S) \le v$ , $\ge v$ | Yes | Yes | **Yes** |
| $\text{sum}(S) \le v$ (items of any value, $v \ge 0$) | Yes | No | No |
| $\text{sum}(S) \le v$ (items of any value, $v \le 0$) | No | Yes | No |
| $\text{sum}(S) \ge v$ (items of any value, $v \ge 0$) | No | Yes | No |
| $\text{sum}(S) \ge v$ (items of any value, $v \le 0$) | Yes | No | No |

> **Note the subtlety**: `sum` is plain anti-monotone/monotone **only when all item values are non-negative**. As soon as values can be **negative**, `sum` degrades to merely *convertible*, and **the sign of $v$ decides which of the two it converts to**.

### 2.12 Data space pruning with data anti-monotonicity

Data-space pruning acts on **transactions**, not on itemsets: it asks whether a transaction $t$ **can still contribute** to generating satisfiable patterns.

- **Data succinct**: the data space can be pruned **at the initial mining step** — remove all data that cannot meet the criteria, shrinking the input before the algorithm starts.
- **Data anti-monotone**: *a constraint $c$ is data anti-monotone if, for a pattern $p$ that cannot be satisfied by a transaction $t$ under $c$, **$p$'s supersets cannot be satisfied by $t$ either**.*

> **The key mechanism is RECURSIVE DATA REDUCTION** — the pruning is re-applied inside every projected database, so the data keeps shrinking as the recursion deepens.

**Examples**: $\text{sum}(I.\text{price}) \ge v$ is data anti-monotone; $\text{min}(I.\text{price}) \le v$ is data anti-monotone.

**Worked example — $C: \text{range}(I.\text{profit}) \ge 25$**, with this TDB and profits (`min_sup = 2`):

```
TID   Transaction                 Item profits
10    a, b, c, d, f, h            a: 40    b:  0    c: -20   d: -15
20    b, c, d, f, g, h            e: -30   f: -10   g: 20    h:  -5
30    b, c, d, f, g
40    c, e, f, g
```

Projected database of the itemset $\{b, c\}$:
- $T_{10}' = \{d, f, h\}$ → profits $\{-15, -10, -5\}$ → $\text{range} = -5 - (-15) = 10 < 25$ → **$C$ can never be satisfied by $T_{10}'$, so $T_{10}'$ is PRUNED.**
- $T_{20}' = \{d, f, g, h\}$ → profits $\{-15,-10,20,-5\}$ → $\text{range} = 35 \ge 25$ → kept.
- $T_{30}' = \{d, f, g\}$ → $\text{range} = 35 \ge 25$ → kept.

### 2.13 Constraint-based mining — the general picture

| Constraint | Anti-monotone | Monotone | Succinct |
|---|---|---|---|
| $v \in S$ | no | yes | yes |
| $S \supseteq V$ | no | yes | yes |
| $S \subseteq V$ | yes | no | yes |
| $\min(S) \le v$ | no | yes | yes |
| $\min(S) \ge v$ | yes | no | yes |
| $\max(S) \le v$ | yes | no | yes |
| $\max(S) \ge v$ | no | yes | yes |
| $\text{count}(S) \le v$ | yes | no | weakly |
| $\text{count}(S) \ge v$ | no | yes | weakly |
| $\text{sum}(S) \le v$ ($\forall a \in S,\, a \ge 0$) | yes | no | no |
| $\text{sum}(S) \ge v$ ($\forall a \in S,\, a \ge 0$) | no | yes | no |
| $\text{range}(S) \le v$ | yes | no | no |
| $\text{range}(S) \ge v$ | no | yes | no |
| $\text{avg}(S)\, \theta\, v$, $\theta \in \{=, \le, \ge\}$ | **convertible** | **convertible** | no |
| $\text{support}(S) \ge \xi$ | yes | no | no |
| $\text{support}(S) \le \xi$ | no | yes | no |

**A classification of constraints** (the Venn-diagram slide): the universe of constraints splits into **Antimonotone**, **Monotone**, **Succinct** (these three overlap), plus **Convertible anti-monotone** and **Convertible monotone**, whose intersection is **Strongly convertible**; everything left over is **Inconvertible**.

> **How to read the table under exam pressure**: reverse the inequality and you usually **swap** anti-monotone ↔ monotone. `min`/`max`/`∈`/`⊆` constraints are **always succinct** (decidable item-by-item); `sum`, `range`, `avg`, `support` are **never succinct**. `count` is only **weakly** succinct.

---

## 3. Mining High-Dimensional Data and Colossal Patterns

> Reference: F. Zhu, X. Yan, J. Han, P. S. Yu, H. Cheng, *"Mining Colossal Frequent Patterns by Core Pattern Fusion"*, ICDE'07.

### 3.1 The problem: downward closure becomes a liability

*We have many algorithms, but can we mine **large (colossal) patterns** — of size around 50 to 100? **Unfortunately, not.***

**Why not? The curse of "downward closure".**

- **Downward closure (a.k.a. the Apriori property)**: *any sub-pattern of a frequent pattern is frequent.*
- If $(a_1, a_2, \dots, a_{100})$ is frequent, then **every one of its $2^{100}-1$ non-empty subsets is frequent too** — $a_1$, $(a_1,a_2)$, $(a_1,a_3)$, …, $(a_1,a_2,a_3)$, …
- **No matter whether we use breadth-first search (Apriori) or depth-first search (FP-Growth)**, we must examine that astronomical number of patterns.

> **The property that makes mining feasible for small patterns is exactly the property that makes it impossible for colossal ones.** The search leads to explosion, i.e. a system crash or effectively infinite execution time.

**Do closed/maximal patterns save us? No — only partially.**

- **Closed pattern**: a frequent pattern $X$ is **closed** if there is **no super-pattern $Y$ with the same support** as $X$.
- **Maximal pattern**: a frequent pattern $X$ is **maximal** if there is **no frequent super-pattern $Y$**. *(It is the largest pattern that still clears the support threshold; beyond that boundary we become too specific and frequency drops below `min_sup`.)*

> **⚠️ These concepts reduce the SIZE OF THE OUTPUT, not the COST OF THE SEARCH.** To decide that a pattern is maximal or closed, traditional algorithms **still have to traverse the space** and meet the explosion of sub-patterns. And, as the next example shows, sometimes they do not even reduce the output.

### 3.2 The motivating (diagonal) example — closed/maximal do not help

Build a database of **40 transactions**, each initially containing **all 40 items**; then **delete the diagonal**: remove item 1 from $T_1$, item 2 from $T_2$, …, item 40 from $T_{40}$.

```
T1  = 2 3 4 ..... 39 40
T2  = 1 3 4 ..... 39 40
:
T40 = 1 2 3 4 .... 39
```

Result: each transaction has exactly **39 items**, and **each item appears in 39 of the 40 transactions**.

Set $\sigma = 20$ (minimum support). **Any combination of 20 items out of 40 appears in enough transactions to be frequent.** The number of frequent patterns of size 20 is

$$\binom{40}{20} \approx 1.37 \times 10^{11} \quad (\approx 137 \text{ billion patterns})$$

and, in general, $\binom{n}{n/2} \approx \dfrac{2^n}{\sqrt{n}}$ — **exponential in $n$**.

> **⚠️ And here is the sting: EVERY one of those size-20 patterns is BOTH CLOSED AND MAXIMAL.** Adding a 21st item changes the support (so the pattern was closed) or drops it below the threshold (so it was maximal). **Even restricting the output to "essential" patterns leaves you with billions of results.**

### 3.3 The colossal pattern hiding in the noise

Extend the same database with 20 more transactions:

```
T41 = 41 42 43 ..... 79
T42 = 41 42 43 ..... 79
:
T60 = 41 42 43 ..... 79
```

With $\sigma = 20$ there is **exactly one pattern of size greater than 20** — the **colossal** pattern
$$\alpha = \{41, 42, \dots, 79\}, \quad |\alpha| = 39, \quad \text{supported by } T_{41} \dots T_{60}.$$

> **The existing fastest mining algorithms (FPClose, LCM) FAIL to complete**: they are trapped exploring the $1.37 \times 10^{11}$ mid-sized patterns of the first 40 transactions. **Pattern-Fusion outputs the colossal pattern in seconds.**

### 3.4 Why we care, and the change of philosophy

**Colossal patterns are few but precious**: in high-dimensional datasets only a **negligible fraction** of the frequent patterns are colossal, yet they carry **substantially higher analytical and domain-specific value** than the small ones.

**Motivating applications**:
- **Micro-array analysis in bioinformatics** (especially when support is low),
- **biological sequence patterns**,
- **biological / sociological / information graph pattern mining**.

**The three principles of the new philosophy:**
1. **No hope for completeness** — if mining mid-sized patterns already explodes, there is **no hope** of reaching colossal patterns under a "complete set" mining philosophy.
2. **Jump out of the swamp of mid-sized results** — develop a philosophy that **skips the explosive middle** and lands directly on colossal patterns.
3. **Strive for mining almost-complete colossal patterns** — build a mechanism that **quickly reaches colossal patterns and discovers most of them**.

### 3.5 Core patterns and the robustness of colossal patterns

**Observation**: a colossal pattern $\alpha$ is surrounded by a **dense population of sub-patterns $\alpha_1, \dots, \alpha_k$ that cluster tightly around it by sharing a similar support set**. These are called **core patterns of $\alpha$**.

> **Intuition** (lecture): since $\alpha$ is frequent, and each $\alpha_i \subseteq \alpha$ appears at least wherever $\alpha$ appears, the sub-patterns of a colossal pattern tend to have **support very close to $\alpha$'s own support**.

**Definition (τ-core pattern)**: for a frequent pattern $\alpha$, a sub-pattern $\beta$ is a **$\tau$-core pattern of $\alpha$** if it shares a similar support set with $\alpha$:

$$\frac{|D_\alpha|}{|D_\beta|} \ge \tau, \qquad 0 < \tau \le 1$$

where $D_\alpha$ is the set of transactions containing $\alpha$ and $\tau$ is the **core ratio**. *(Since $\beta \subseteq \alpha$, necessarily $D_\alpha \subseteq D_\beta$, hence the ratio is $\le 1$.)*

**Definition ($(d,\tau)$-robustness)**: a pattern $\alpha$ is **$(d, \tau)$-robust** if $d$ is the **maximum number of items that can be removed from $\alpha$** for the resulting pattern to **remain a $\tau$-core pattern of $\alpha$**.

- **For a $(d,\tau)$-robust pattern $\alpha$, the number of core patterns is $\Omega(2^d)$.**
- **Colossal patterns tend to have a very large number of core patterns** — this density is the **theoretical foundation of Pattern-Fusion**: *the larger the pattern, the more "clues" (core patterns) it leaves scattered in the search space.*

**Pattern distance** (a Jaccard distance over support sets):

$$\text{Dist}(\alpha, \beta) = 1 - \frac{|D_\alpha \cap D_\beta|}{|D_\alpha \cup D_\beta|}$$

**Bounding ball**: if $\alpha$ and $\beta$ are both core patterns of the same pattern, their distance is bounded by a radius that depends only on the core ratio $\tau$:

$$\text{Dist}(\alpha, \beta) \;\le\; 1 - \frac{1}{\frac{2}{\tau} - 1} \;=\; r(\tau)$$

> **Key insight**: **once we identify ONE core pattern, we can find ALL the other core patterns inside a bounding ball of radius $r(\tau)$** in pattern space. That is what turns a random lucky hit into a full reconstruction.

### 3.6 Example: a colossal pattern has far more core patterns than a small one

With $\tau = 0.5$:

| Transaction (# of Ts) | Core patterns ($\tau = 0.5$) |
|---|---|
| **(abe)** (100) | (abe), (ab), (be), (ae), (e) |
| **(bcf)** (100) | (bcf), (bc), (bf) |
| **(acf)** (100) | (acf), (ac), (af) |
| **(abcef)** (100) | (ab), (ac), (af), (ae), (bc), (bf), (be), (ce), (fe), (e), (abc), (abf), (abe), (ace), (acf), (afe), (bcf), (bce), (bfe), (cfe), (abcf), (abce), (bcfe), (acfe), (abfe), (abcef) |

**Consequences (from the slides):**
- A colossal pattern has **far more core patterns** than a small-sized pattern.
- A colossal pattern has **far more core descendants of a given smaller size $c$**.
- Therefore **a random draw from the complete set of patterns of size $c$ is much more likely to pick a core descendant of a colossal pattern**.
- And **a colossal pattern can be generated by MERGING a set of its core patterns.**

> **The lecture's numeric illustration**: the small pattern $(bcf)$ has **low $(d,\tau)$-robustness** — only a couple of items can be removed before the sub-pattern stops being a core pattern, so $d$ is small and there are very few "entry points" for a search algorithm. The colossal pattern $(abcef)$, being larger, generates an **exponentially larger pool of core patterns**. E.g. $(ab)$ qualifies as a core pattern of $\alpha = (abcef)$: if $\alpha$ appears in 100 transactions and $(ab)$ appears in 200, then
> $$\frac{|D_\alpha|}{|D_{(ab)}|} = \frac{100}{200} = 0.5 \ge \tau \;\;\checkmark$$
> Because $(abcef)$ is $(d,\tau)$-robust with a **larger $d$**, an algorithm can **afford to miss or drop several items and still stay inside the colossal pattern's sphere of influence**.

**Geometric intuition — colossal patterns correspond to DENSE BALLS:**
- Due to their robustness, colossal patterns correspond to **dense balls**, $\Omega(2^d)$ in population.
- **A random draw in pattern space will hit somewhere in the ball with high probability.** In the example above, the probability of drawing a descendant of $abcef$ is **0.9**; for smaller patterns it drops drastically.

### 3.7 The Pattern-Fusion algorithm

#### Methodology (the two ideas)
1. **Bounded-breadth traversal of the pattern tree** — always push down a **frontier of a bounded-size candidate pool**. Only a **fixed number** of patterns in the current pool are used as starting nodes to descend the pattern tree → **avoids the exponential search space**.
2. **"Shortcuts" and leaps** — pattern growth is **NOT performed by single-item addition** but by **agglomeration of multiple patterns from the pool**. These shortcuts direct the search down the tree **much more rapidly** toward the colossal patterns.

#### Idea, step by step
1. **Initial pattern generation**: generate a **complete set of frequent patterns up to a small size** (e.g. size 3) using an existing algorithm (Apriori / FP-Growth).
2. **Seed selection**: **randomly pick a pattern $\beta$** — it has a **high probability of being a core-descendant of some colossal pattern $\alpha$** (this is the statistical property from §3.6).
3. **Core-descendant identification and fusion**: identify **all patterns within the bounding ball centred at $\beta$**, and **merge (fuse) them all** — this generates a **much larger core-descendant of $\alpha$**.
4. **Iterative refinement**: repeat by selecting **$K$ seed patterns**; the resulting set of larger core-descendants becomes the **candidate pool for the next iteration**.

#### The algorithm (phases)

- **Initialization (initial pool)**: use an existing algorithm to mine **all frequent patterns up to a small size**, e.g. 3.
- **Iteration (iterative pattern fusion)**, repeated until termination:
  - **Sampling**: $K$ seed patterns are **randomly picked** from the current pool.
  - **Bounding-ball computation**: for each seed, find **all patterns within a bounding ball centred at the seed** — distance measured by the similarity of their supporting transaction sets (TID sets).
  - **Fusion**: all patterns found are **fused together into super-patterns**; all super-patterns generated form the **new pool** for the next iteration.
- **Termination**: when the current pool contains **no more than $K$ patterns at the beginning of an iteration**.

> **Practical note from the slides**: when the number of such $\beta$ patterns inside a ball **exceeds a system-determined threshold**, a **sampling technique** is used to decide which subset of $\beta$ to retain.

#### Why is Pattern-Fusion efficient?

- **Bounded-breadth pattern-tree traversal** — it **avoids the explosion of mid-sized patterns** entirely (traditional bottom-up mining suffers exactly there).
- **Randomness comes to help to stay on the right path** — random sampling keeps the algorithm on a computationally feasible route while retaining a high probability of hitting core-descendants of colossal patterns.
- **Ability to identify short-cuts and take leaps** — **fuse small patterns together in one step** to generate new patterns of significant size, jumping through the pattern lattice.

#### Approximation quality: why the approximation is *good*

- **Gearing toward colossal patterns**: **the larger the pattern, the greater the chance it will be generated** — a larger pattern has more core-descendants in the initial pool, so a random seed is more likely to belong to its "family". *(Paradoxically, big patterns are EASIER to find than mid-sized ones here.)*
- **Catching outliers**: **the more distinct the pattern, the greater the chance it will be generated** — distinct/rare patterns have unique TID-set signatures.

> **The two things that must be proved** (as the professor framed it): **(1)** that the solutions returned really are **good approximations of the colossal patterns** in the transaction database, and **(2)** that we can **reach that approximation in reasonable time**. The experiments below address both.

### 3.8 Experimental setting and results

**Datasets:**
- **Synthetic — $\text{Diag}_n$**: an $n \times (n-1)$ table where the $i$-th row contains the integers $1..n$ **except $i$**; each row is an itemset; `min_support = n/2`. (This is exactly the "delete the diagonal" construction of §3.2 — a dense environment full of overlapping near-colossal patterns.)
- **Real — REPLACE**: a **program trace dataset** collected from the `replace` program, widely used in software-engineering research.
- **Real — ALL**: a popular **gene expression dataset**, clinical data on **ALL-AML leukemia**. Each **item is a column**, representing the **activity level of a gene/protein in the sample**. Frequent patterns here would reveal **important correlations between gene expression patterns and disease outcomes**.

**Results on $\text{Diag}_n$:**
- **LCM's run time increases exponentially** with the pattern size $n$; **Pattern-Fusion finishes efficiently** (sub-exponential).
- **Approximation error** of Pattern-Fusion (with `min_sup = 20`) compared with the complete set is **rather close to uniform sampling** — i.e. close to the "gold standard" of picking $K$ patterns perfectly at random from the full answer set.
- **The error decreases as the number of mined patterns (pool size) increases** → more seeds/iterations ⇒ convergence toward the true colossal set.

**Results on ALL:**
- 38 transactions, each with **866 columns**, **1736 items** in total; the table reports results at a **high frequency threshold of 30**.
- **Pattern-Fusion stays stable even when `min_sup` is decreased**, whereas the classical approach shows an **explosion of computational time**.

**Results on REPLACE:**
- **4395 calls/transitions → 4395 transactions with 57 items** in total. **Aim**: identify frequent — and therefore **normal** — program-execution structures (so that deviations flag anomalies).
- With a support threshold of **0.03**, the **largest patterns are of size 44**. All of them are **discovered by Pattern-Fusion** for different settings of $K$ and $\tau$, starting from an **initial pool of 20,948 patterns of size $\le 3$**.
- **Recall**: out of the **98 patterns of size $\ge 42$**, with $K = 100$ Pattern-Fusion returns **80** of them.
- **Average distance error**: any pattern in the complete set is on average **at most 0.17 items away** from one of those 80 — i.e. even when the exact pattern is missed, the algorithm supplies a **nearly identical substitute**.

> **Conclusion of the section** (lecture): the whole intuition is that **a colossal pattern is characterised by having a huge number of small sub-patterns, densely concentrated**, so we have a **high probability of picking one of them up**. Unlike exhaustive search, **Pattern-Fusion does not guarantee finding every colossal pattern**, but it gives a **high-quality approximation in a fraction of the time** — the superior choice for extremely large or dense datasets where traditional mining is computationally prohibitive.

---

## 4. Mining Compressed or Approximate Patterns

> **The framing question** (lecture, 12/03): when we find frequent patterns, we need a way to **reduce as much as possible the information transmitted to the user**. Compression is one way; redundancy-aware top-$k$ is another.

### 4.1 Mining compressed patterns: δ-clustering

**Why compressed patterns?** Because the output is **too large, but less meaningful** — quantity is not quality.

**Pattern distance measure**: the same Jaccard-style distance over supporting transaction sets used in §3.5, where $O(P)$ denotes the set of transactions (objects) supporting $P$.

**δ-clustering**: for each pattern $P$, find **all patterns $S$ that can be expressed by $P$** (that is, $O(S) \subset O(P)$) and whose **distance to $P$ is within $\delta$** — the **δ-cover**. **All patterns in the cluster can then be represented by $P$ alone.** *(Xin et al., "Mining Compressed Frequent-Pattern Sets", VLDB'05.)*

**Worked example:**

| ID | Item-set | Support |
|---|---|---|
| $P_1$ | {38, 16, 18, 12} | 205 227 |
| $P_2$ | {38, 16, 18, 12, 17} | 205 211 |
| $P_3$ | {39, 38, 16, 18, 12, 17} | 101 758 |
| $P_4$ | {39, 16, 18, 12, 17} | 161 563 |
| $P_5$ | {39, 16, 18, 12} | 161 576 |

Three possible outputs:

| Strategy | Output | Verdict |
|---|---|---|
| **Closed frequent patterns** | report **all five** $P_1 \dots P_5$ | **emphasises support too much ⇒ NO COMPRESSION** |
| **Max-pattern only** | report **$P_3$** | **INFORMATION LOSS** |
| **δ-clustering** | report **$P_2$, $P_3$, $P_4$** | **the desirable output** |

> **The reasoning behind the desirable output** (lecture): by the Apriori property all sub-patterns are frequent by definition, and lacking other information we would also have to assume they all share the same support — **but if we return only the maximal pattern we lose the support information**. With δ-clustering, **$P_2$ absorbs $P_1$** (its sub-pattern, whose support 205 227 is almost identical to 205 211) and **$P_4$ absorbs $P_5$** (161 563 vs. 161 576). **We lose some information, but the amount is limited and bounded by $\delta$** — that is the whole point: the loss is *controlled*.

### 4.2 Redundancy-aware top-k patterns

Mining the **top-$k$ most frequent patterns** is a natural strategy for reducing the number of returned patterns. **But frequent patterns are not mutually independent — they are often clustered in small regions.**

> **The professor's analogy**: it is like finding the **20 largest population centres in the world** — you end up with cities **clustered in a small number of countries** rather than evenly distributed across the globe. *(In the lecture's version: the $k$ biggest cities would all be in China.)* If we rank on frequency alone, all returned patterns can be **close to each other** and thus fail to express the total information about the problem.

**We must balance HIGH SIGNIFICANCE against LOW REDUNDANCY.** The proposed instrument is **MMS — Maximal Marginal Significance** — for measuring the **combined significance of a pattern set** *(Xin et al., "Extracting Redundancy-Aware Top-K Patterns", KDD'06)*.

**Significance measure $S$**: a function mapping a pattern $p \in P$ to a real value, where $S(p)$ is the **degree of interestingness (or usefulness)** of $p$. (In our case, significance can simply be the **frequency**.)

- **Objective measures** depend only on the **structure of the pattern and the underlying data** used in the discovery process.
- **Subjective measures** are based on **user beliefs** about the data — they depend on **who examines the patterns**.

**Redundancy between two patterns:**
$$R(p, q) = S(p) + S(q) - S(p, q)$$

> **How to read it**: if $p$ and $q$ overlap heavily, the significance of the pair $S(p,q)$ is much less than the sum of the individual significances, so $R$ is large.

**The practical route**: the ideal $R(p,q)$ is usually **hard to obtain**, but we can **approximate redundancy using the distance between patterns**. Then finding redundancy-aware top-$k$ patterns becomes **finding a $k$-pattern set that maximises the marginal significance** — a **well-studied problem in information retrieval**:

> *A document has high marginal relevance if it is **both relevant to the query and has minimal marginal similarity to previously selected documents**, where the marginal similarity is computed by choosing the most relevant already-selected document.*

**To sum up: return only the patterns that are significant AND not redundant with respect to each other.**

---

## 5. Summary

- **Roadmap**: many aspects and extensions exist on top of basic pattern mining.
- **Mining patterns in multi-level, multi-dimensional space** — flexible/group-based supports, redundancy filtering, inter- vs. hybrid-dimensional rules, discretization of quantitative attributes.
- **Mining rare and negative patterns** — group-based thresholds; **null-invariant** measures (Kulczynski) instead of raw support.
- **Constraint-based pattern mining** — anti-monotone, monotone, succinct, convertible; pattern-space **and** data-space pruning.
- **Specialized methods for mining high-dimensional data and colossal patterns** — Pattern-Fusion.
- **Mining compressed or approximate patterns** — δ-clustering and redundancy-aware top-$k$.

> **Bridge to the next chapter** (end of the lecture): so far, **frequent patterns ignore the ORDER between items**. Next we analyse **frequent SEQUENCES**, where the approach has to change in order to take ordering into account.

---

## Key points / potential exam pitfalls

### Multi-level and multi-dimensional
- **Lower level ⇒ lower support**, always. That is *why* uniform support over-prunes; **reduced/varying support per level** is the fix.
- **Group-based `min_sup`** exists because **value and frequency are not the same thing** (diamonds vs. bread). Lowering the global threshold instead is the **wrong** answer — it causes combinatorial explosion.
- **⚠️ Redundancy is not "the specific rule has lower support"** — that is *expected*. A rule is redundant when its support/confidence are **close to what the ancestor predicts**. A descendant with *surprising* confidence is **not** redundant.
- **Inter-dimensional = no repeated predicate; hybrid-dimensional = repeated predicate.** Both are handled by **encoding each attribute–value pair as an item** and running the standard algorithms.
- **Static vs. dynamic discretization**: static uses **predefined concept hierarchies** (data cube); dynamic uses **the data distribution** (binning); a third route is **clustering first, association after**; a fourth is **deviation** from the global mean.

### Rare and negative patterns
- **⚠️ The support-based definition of negative correlation is NOT NULL-INVARIANT** — the needle example flips from "negative" to "positive" just by adding transactions containing neither item. Know both numeric cases (200 vs. $10^5$ transactions).
- **Kulczynski $\frac{P(X|Y)+P(Y|X)}{2} < \varepsilon$ is null-invariant** because it only uses **conditional** probabilities.
- **Infrequent negatively-correlated patterns are more interesting than frequent ones** — the opposite of the usual instinct.

### Constraint-based mining
- **Constraint-based mining is still COMPLETE** — it returns *all* answers satisfying the constraints, unlike heuristic search. Do not confuse it with sampling/approximation.
- **Anti-monotone = "if violated, all supersets violate" ⇒ prune the branch.** **Monotone = "if satisfied, all supersets satisfy" ⇒ stop checking.** Monotonicity saves *checks*, anti-monotonicity saves *search space*.
- **Support count is the anti-monotone constraint** — Apriori is literally an instance of constraint pushing.
- **⚠️ Reversing the inequality usually swaps the two properties**: $\text{sum} \le v$ anti-monotone, $\text{sum} \ge v$ monotone (for non-negative values); $\min \ge v$ anti-monotone, $\min \le v$ monotone.
- **Succinct = decidable from the ITEMS ALONE, before any counting** ("pre-counting pushable"). $\min(I.price) \ge v$ is succinct; $\text{sum}(I.price) \ge v$ is **not**, because no per-item test decides it.
- **⚠️ Convertible constraints (avg, median) CANNOT be pushed into Apriori.** Memorise the counter-example: $\{d,f\}$ violates $\text{avg} \ge 25$ but $\{a,d,f\}$ satisfies it, and Apriori **needs** $\{d,f\}$ to assemble $\{a,d,f\}$. They **can** be pushed into **FP-Growth** because it grows by **prefix paths**, and the item order makes the average monotone along a path.
- **Descending value order ⇒ convertible ANTI-monotone; ascending order ⇒ convertible MONOTONE; both ⇒ STRONGLY convertible** (avg and median are strongly convertible).
- **`sum` is only *convertible* when item values may be NEGATIVE** — and the **sign of $v$** decides which direction it converts to.
- **Data anti-monotonicity acts on TRANSACTIONS, not itemsets**, and its power comes from **recursive data reduction** inside projected databases. Be able to redo the $\{b,c\}$ projected-DB example with $\text{range}(I.profit) \ge 25$.
- **Multiple constraints**: no conflict if **one order works for both**; otherwise **satisfy one first, then re-order and mine the projected databases** (or post-filter).

### Colossal patterns
- **⚠️ Downward closure is the enemy here.** A frequent pattern of size 100 implies $2^{100}-1$ frequent sub-patterns; **BFS and DFS suffer equally**.
- **Closed/maximal patterns reduce OUTPUT SIZE, not SEARCH COST** — and in the diagonal example **all $\binom{40}{20} \approx 1.37\times 10^{11}$ size-20 patterns are simultaneously closed and maximal**, so they reduce nothing at all.
- **$\tau$-core pattern**: $\frac{|D_\alpha|}{|D_\beta|} \ge \tau$, with $\beta \subseteq \alpha$ hence $D_\alpha \subseteq D_\beta$. **Careful with which set is on top** — the colossal pattern's support set is the **numerator**.
- **$(d,\tau)$-robust ⇒ $\Omega(2^d)$ core patterns.** **Bigger pattern ⇒ more core patterns ⇒ EASIER to hit by random sampling.** This inversion ("large patterns are easier to find than mid-sized ones") is the counter-intuitive heart of the method.
- **$\text{Dist}(\alpha,\beta) = 1 - \frac{|D_\alpha \cap D_\beta|}{|D_\alpha \cup D_\beta|}$** and the bound $\text{Dist} \le 1 - \frac{1}{2/\tau - 1} = r(\tau)$: **one core pattern found ⇒ all the others are inside the ball**.
- **Pattern-Fusion terminates when the pool has $\le K$ patterns** at the start of an iteration — not when patterns stop growing.
- **⚠️ Pattern-Fusion is an APPROXIMATION with no completeness guarantee.** Its selling point is *quality of approximation per unit of time* (REPLACE: 80 of 98 patterns of size $\ge 42$, average error 0.17 items).
- **Randomness is a feature, not a shortcut**: it is what keeps the traversal bounded-breadth while still landing in dense balls with high probability.

### Compressed / top-k patterns
- **δ-clustering keeps the REPRESENTATIVE $P$ of a cluster**, absorbing sub-patterns whose support-set distance is within $\delta$. **Returning only the max-pattern loses support information; returning all closed patterns compresses nothing.**
- **In the VLDB'05 example the desired output is $P_2, P_3, P_4$** — know *why* $P_1$ and $P_5$ get absorbed.
- **Top-$k$ by frequency alone yields REDUNDANT, spatially clustered patterns** (the "20 largest cities" analogy). **MMS** trades significance against redundancy.
- **$R(p,q) = S(p) + S(q) - S(p,q)$** — and since the true $R$ is hard to compute, it is **approximated by pattern distance**, turning the task into the **maximal marginal relevance** problem of information retrieval.
- **Objective vs. subjective significance measures**: structure/data vs. user beliefs.

### Cross-chapter connections
- **Apriori, FP-Growth, support, confidence, closed/maximal patterns** all come from **Chapter 6 (Frequent Patterns)** — this chapter is built entirely on top of them, and the **downward-closure property** is reinterpreted here as a *liability*.
- **Null-invariance** and the **Kulczynski measure** continue the discussion of **interestingness measures beyond support/confidence** started in Chapter 6 (lift, $\chi^2$, all-confidence, cosine): the same null-transaction problem that plagued lift and $\chi^2$ reappears here for negative correlations.
- **Concept hierarchies** and **discretization (binning, static vs. dynamic)** come straight from **Chapter 3 (Data Preprocessing)**; **clustering-based discretization** links to **Chapter 5**.
- **The curse of dimensionality** returns again — here as the reason why colossal pattern mining is intractable in high-dimensional biological data (micro-arrays, gene expression).
- **The Jaccard-style distance $1 - \frac{|A \cap B|}{|A \cup B|}$** used for pattern distance is the **binary/asymmetric similarity measure of Chapter 2**, applied to support sets instead of attribute vectors.
- **The "normal behaviour" framing of the REPLACE experiment** (frequent = normal execution structures, deviations = anomalies) connects directly to **Chapter 7 (Outlier Analysis)**.
- **Redundancy-aware top-$k$** mirrors the **representative-selection problem** seen in clustering (Chapter 5): coverage of the space beats raw score.

---

*File auto-generated by merging `8-FPAdvanced.pdf` (professor's slides, 32 pages) and `8 - FPAdvanced sbobine.pdf` (lecture notes of 10/03 plus the opening of 12/03, 27 pages). Formulas garbled by PDF text extraction (Kulczynski measure, core-ratio and pattern-distance definitions, the needle-package supports) have been reconstructed to their correct standard form. For questions about this chapter, refer only to this file.*
