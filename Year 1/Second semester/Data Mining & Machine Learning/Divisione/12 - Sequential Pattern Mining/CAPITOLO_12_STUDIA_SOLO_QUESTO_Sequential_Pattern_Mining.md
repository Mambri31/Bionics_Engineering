# Chapter 12 — Sequential Pattern Mining

> **Course:** Data Mining and Machine Learning / Bioinspired Computational Methods — Biological Data Mining
> **Instructor:** Prof. Francesco Marcelloni (University of Pisa)
> **Sources merged into this file:** professor's slides (`12-SequentialPatternMining.pdf`, 36 pages) + lecture notes (`12 - SequentialPatternMining Sbobine.pdf`, 35 pages — lesson **L09, 31/03/2026**, plus the second part given on **14/04**)
>
> **Instructions for Claude:** this is the single reference file for Chapter 12. When the user asks study questions about this chapter, use **only** this file as context (no need to re-read the original PDFs unless a detail is missing). The file merges the official slide content with the lecture's discursive explanations and worked examples, keeping both levels of detail.
>
> **📌 The professor's own exam hint, stated verbatim at the end of the lecture:** *"For the exam: **describe the Apriori algorithm, its advantages, and why we explore some smarter solutions.**"* Structure your revision around that arc.

---

## Chapter outline
1. Introduction: sequences vs. frequent patterns — what changes when order matters
2. Web usage mining as an application
3. Formal definitions: itemsets, sequences, lexicographic order, subsequences, support
4. Closed vs. maximal sequential patterns — and what each one loses
5. The five phases of the Apriori-style approach
6. AprioriAll (count-all)
7. AprioriSome (count-some)
8. AprioriDynamicSome
9. Performance comparison and the bottlenecks of Apriori-like methods
10. FreeSpan: FP-growth for sequential pattern mining
    - Projected sequence databases; parallel vs. partition projection
    - Level-by-level projection
    - Alternative-level projection and the frequent item matrix
11. Performance study and why FreeSpan wins

---

## 1. Introduction: what changes when order matters

> **The framing question of the whole chapter** (lecture): *what is the real difference between **sequences** and the **patterns** we studied in frequent pattern analysis?*
>
> **In frequent pattern analysis the ORDER INSIDE the pattern is NOT relevant.** If we go to the supermarket and buy several products, **the order appearing on the bill just depends on the sequence in which the products are passed over the reader** — it carries no information. **The situation is different with frequent SEQUENCES: here the order IS relevant.**

> **Sequential pattern mining discovers FREQUENT SUBSEQUENCES as patterns in a sequence database.**
> **A sequence database stores a number of records, where all records are SEQUENCES OF ORDERED EVENTS, with or without concrete notions of time.**

- **Records are stored as:** `[Transaction/Customer ID, <Ordered Sequence Events>]`

**The reference example — retail customer transactions**, i.e. purchase sequences in a grocery store showing, **for each customer, the collection of store items purchased every week for one month**:

```
[T1, <(bread, milk), (bread, milk, sugar), (milk), (tea, sugar)>]
[T2, <(bread), (sugar, tea)>]
```

> **How to read it** (lecture): **we must think of a sequence as ONE customer who goes to the supermarket SEVERAL TIMES.** User T1 purchased **bread and milk the first time**, **bread, milk and sugar the second time**, and so on.

> **⚠️ THE structural point of the chapter, stated twice in the lecture:**
> **A sequence is a sequence of ELEMENTS, and each element can be an ITEMSET.** Inside the parentheses we write the itemset.
> **The order INSIDE the parentheses is NOT relevant. The order BETWEEN elements IS relevant.**
> Everything difficult in this chapter follows from that double structure.

> **On time**: in the supermarket example we have a natural notion of time, **but we can also have sequences in which no time is associated with the fact that there is a sequence** — order alone is enough.

## 2. Web usage mining as an application

> **The goal** (lecture): understand the usage of the web pages of a web site — **when a customer starts interacting with our site, we want to know whether there are specific sequences of pages that the customer visits.**

> **⚠️ Why this application is SINGULAR**: **each element of the sequence is a SINGLE ITEM**, because we consider a single web page. **In this specific application an element cannot be an itemset**, under the assumption that **a web user can physically access only one web page at any given point in time.**

Given a set of events $E = \{a, b, c, d, e, f\}$, a web-access sequence database for four users:

```
[T1, <a, b, d, a, c>]
[T2, <e, a, e, b, c, a, c>]
[T3, <b, a, b, f, a, e, c>]
[T4, <a, b, f, a, c>]
```

**The aim is to find a frequent sequence such as $\langle a, b, a, c\rangle$**, indicating that **over 90% of the users who visit product $a$'s web page also immediately visit product $b$'s web page.**

> **Why a site owner cares** (lecture): **they can create links between page $a$ and page $b$** if they are confident that users will visit $b$ after $a$.

### Where to collect the log: server-side, client-side or proxy?

| Location | Pros | Cons |
|---|---|---|
| **Server-side** — put software inside our web server to record, for each user, the collection of visited pages | **reflects the access of a web site by MULTIPLE users** — exactly what we want; **good for mining multiple users' behaviour and for web recommender systems** | **⚠️ server logs may not be entirely reliable due to CACHING — cached page views are NOT recorded** |
| **Client-side** | **eliminates caching and session-identification problems**; useful for **web content personalization** | **can be very complex**: **requires a remote agent to be implemented, or a modified browser**, and **we must install it on many users** to collect data |
| **Proxy server** | **reveals the actual HTTP requests from multiple clients to multiple web servers**, thus **characterizing the browsing behaviour of a group of anonymous users sharing a common proxy**; **still quite reliable** | **a user can interact with different proxies**, so **we can still miss some information** |

> **⚠️ The caching detail the professor spelled out**: when a user interacts with a website there are **TWO LEVELS OF CACHE** — **the first inside the web browser on the client side, the second in the proxy between client and server**. So on the server side **we are simply not able to collect all the possible visits.**

**Data format** (a typical log line):
```
137.207.76.120 - [30/Aug/2009:12:03:24 -0500] "GET /jdk1.3/docs/relnotes/deprecatedlist.html HTTP/1.0" 200 2781
```
**The user here is ANONYMOUS.**

> **Other techniques such as COOKIES and SNIFFERS may be needed — but they are not very reliable**: **we all know that cookies can be rejected**, and the sniffer has a similar problem.
>
> **Conclusion**: **in most cases researchers ASSUME that the user's web-visit information is completely recorded in the web server log**, which is then **preprocessed to obtain the transaction database to be mined for sequences.**

## 3. Formal definitions

### 3.1 Itemsets and the problem statement

**An ITEMSET is a set drawn from the items in $I$**, denoted $(i_1, i_2, \dots, i_k)$, where each $i_j$ is an **item or event**.

> **Problem definition.**
> **GIVEN**:
> - a set of sequential records (**sequences**) representing a **sequential database $D$**;
> - a minimum support threshold **`min_sup`**;
> - a set of $k$ unique items or events $I = \{i_1, i_2, \dots, i_k\}$.
>
> **FIND**: **the set of all FREQUENT SEQUENCES $S$** in $D$ at the given `min_sup`.

> **The lecture's gloss on the two inputs:**
> - **`min_sup`** fixes, for the specific application, when a sequence counts as frequent. In the supermarket case it means **at least `min_sup` customers purchased that sequence of products**.
> - **$I$ is a sort of ALPHABET** for the analysis we want to perform.
>
> **And the warning that motivates §4**: *"we can have the problem of a HIGH NUMBER of sequences, so also in this case we have to reason about **closed** sequences or **maximal** sequences."*

### 3.2 Lexicographic order

**A sequence $S$ is denoted as a sequence of elements $\langle e_1 e_2 e_3 \dots e_q \rangle$**, where **each element $e_j$ is an ITEMSET** — e.g. $(be)$ in $\langle a(be)c(ad)\rangle$ — which **might contain only one item** (a **1-itemset**, e.g. $a$).

**A sequence element is a LEXICOGRAPHICALLY ORDERED LIST OF ITEMS.** Given two itemsets of distinct items
$$t = \{i_1, \dots, i_k\}, \qquad t' = \{j_1, \dots, j_l\}$$
with $i_1 \le \dots \le i_k$ and $j_1 \le \dots \le j_l$ (where $\le$ is the **"occurs before" relationship in lexicographic order**), then **$t < t'$ iff either:**

1. **for some integer $h$, $0 \le h \le \min\{k,l\}$: $i_r = j_r$ for $r < h$, and $i_h < j_h$**; **or**
2. **$k < l$, and $i_1 = j_1,\; i_2 = j_2,\; \dots,\; i_k = j_k$.**

**Examples:** *(rule 1)* $(abc) < (abec)$ and $(af) < (bf)$; *(rule 2)* $(ab) < (abc)$.

> **⚠️ Why we bother with lexicographic order at all** (lecture — the same argument as in Chapter 6): *"of course it doesn't make sense in practice that **milk precedes coffee**, but we fix it because **doing this we can apply the JOIN OPERATOR in an easier way**."* Now that elements are itemsets, **we apply the same philosophy inside the itemset.**
>
> **So: we use lexicographic order FOR COMPUTATION, but inside an itemset the order is not really relevant.** Do not confuse this bookkeeping order with the semantic order *between* elements, which is the real subject of the chapter.

### 3.3 Sequences and subsequences

- **A sequence with $k$ elements is called a $k$-SEQUENCE.**
- **⚠️ An item can occur ONLY ONCE IN AN ITEMSET, but it CAN occur several times in DIFFERENT ITEMSETS of a sequence.**

> **The justification** (lecture): inside a single element **we cannot have a repetition of an item, because we are not interested in whether a user purchased 3, 4 or 5 pieces of the same product — only whether the product is present in the itemset.** But **if the customer goes to the supermarket twice, we ARE interested in knowing whether they bought bread BOTH times.** Hence: **the same item can appear in the sequence, but not twice inside the same itemset.**

> **SUBSEQUENCE.** A sequence $\alpha = \langle e_{i_1} e_{i_2} \dots e_{i_m}\rangle$ is a **subsequence** of $\beta = \langle e_1 e_2 \dots e_n\rangle$, denoted $\alpha \preccurlyeq \beta$, **if there exist integers $i_1 < i_2 < \dots < i_m$** such that
> $$e_{i_j} \subseteq e_j \quad \text{for all } j, \qquad i_1 \ge 1,\; i_m \le n$$

> **⚠️ The definition mixes TWO relations**: elements must appear **IN ORDER** (strictly increasing indices, **not necessarily contiguous**) **AND** each element of $\alpha$ must be a **SUBSET** of the corresponding element of $\beta$.

**A sequential pattern is MAXIMAL if it is not a subsequence of any other sequential pattern** — *i.e. we do not have a supersequence of the sequence we are observing.*

### 3.4 The running sequence database

| Seq. ID | Sequence |
|---|---|
| 10 | $\langle (bd)\,c\,b\,(ac)\rangle$ |
| 20 | $\langle (bf)(ce)\,b\,(fg)\rangle$ |
| 30 | $\langle (ah)(bf)\,a\,b\,f\rangle$ |
| 40 | $\langle (be)(ce)\,d\rangle$ |
| 50 | $\langle a(bd)\,b\,c\,b\,(ade)\rangle$ |

> **Reading the first sequence** (lecture): it has **4 elements** — the first is the **itemset $(bd)$**, the second is just the **item $c$**, the third the **item $b$**, the fourth the **itemset $(ac)$**.

**Two facts stated on the slide:**

**(a) $\langle ad(ae)\rangle$ is a subsequence of $\langle a(bd)bcb(ade)\rangle$.**
> **Why** (lecture): **when we have an itemset such as $(bd)$, we can use the whole itemset, but we can also use a SINGLE ITEM from it.** Here $ad$ is obtained because in the longer sequence we have **$a$ as a single item** and **$d$ as an item inside the itemset $(bd)$**.
>
> **⚠️ And this is exactly where the complexity of the problem comes from**: *"itemsets are very complex to manage because **we have to consider all the possible items inside the itemset**, and this increases the complexity of the problem."*

**(b) Given `min_sup` $= 2$, $\langle (bd)cb\rangle$ is a sequential pattern.**
> **Why** (lecture): in **sequence 10** we have $(bd)$ as an itemset, then $c$, then $b$; in **sequence 50** we again have $(bd)$ followed by $c$ and $b$. Support $= 2$ ⇒ frequent. **And inside this frequent subsequence there are others** — taking $b$ as a single item, $\langle bcb \rangle$ is frequent too.

### 3.5 Support of a sequence

> **The FREQUENCY or SUPPORT of a sequence $S$, denoted $\sigma(S)$, is the total number of sequences of which $S$ is a subsequence DIVIDED BY the total number of sequences in $D$** *(the same definition we had in frequent pattern analysis)*.
> **The ABSOLUTE support (support count) is the total number of sequences in $D$ of which $S$ is a subsequence.**
> **A sequence is FREQUENT if its frequency is not less than the user-specified minimum support, denoted `min_sup` or $\xi$.**

> **⚠️ What is counted: SEQUENCES (customers), not occurrences.** A customer whose sequence contains the pattern three times still contributes **1**.

## 4. Closed vs. maximal sequential patterns — and what each one loses

> **A frequent sequence $S_\alpha$ is a FREQUENT CLOSED SEQUENCE if there exists no proper supersequence of $S_\alpha$ with the same support** — no $S_\beta$ with $S_\alpha \preccurlyeq S_\beta$ and $\sigma(S_\alpha) = \sigma(S_\beta)$. **Otherwise $S_\alpha$ is ABSORBED by $S_\beta$.**

**The slide's worked reasoning**: assume $S_\beta = \langle beadc\rangle$ is the only superset of $S_\alpha = \langle bea\rangle$. Then:
- **if $\sigma(S_\alpha) = \sigma(S_\beta)$ ⇒ $S_\alpha$ is NOT closed;**
- **if $\sigma(S_\alpha) > \sigma(S_\beta)$ ⇒ $S_\alpha$ IS closed.**

**And: $\sigma(S_\beta)$ CANNOT be greater than $\sigma(S_\alpha)$, because $S_\alpha \preccurlyeq S_\beta$** — *the support cannot be higher for the supersequence.*

> **⚠️ This is the DOWNWARD-CLOSURE (Apriori) property transplanted to sequences**, and it is what makes the entire chapter's machinery work.

### The information-theoretic comparison (the most valuable part of the lecture here)

> **CLOSED sequences lose NOTHING.** *"If we compute all the frequent closed sequences, we've achieved our goal — the closed sequences represent our result because **we don't have any missing information**."*
>
> **Why**: we obtain **all the frequent sequences AND all the information about their support**. If $\sigma(S_\alpha) = \sigma(S_\beta)$ we return only the supersequence $S_\beta$; **from $S_\beta$ we can directly DERIVE the support of $S_\alpha$**, straight from the definition of closed sequence.
>
> **⚠️ MAXIMAL sequences DO lose information.** *"Be careful: in the case of maximal sequences **we don't talk about support at all** — we only ask whether a supersequence exists. So if we return only the maximal sequences, **we do NOT have information about the support of the subsequences**."*
>
> **The practical tension**: our target *should* be to reduce the total pool to the **closed and frequent** sequences. **But most of the time the closed sequences are still a great many** — so in practice **we return the MAXIMAL sequences instead**, accepting the loss.

### The introductory example (maximality, and what it costs)

**The database** (`min_sup = 25%`, i.e. **at least 2 of the 5 customers**):

| Customer ID | Customer sequence |
|---|---|
| 1 | $\langle (30)(90)\rangle$ |
| 2 | $\langle (10\;20)(30)(40\;60\;70)\rangle$ |
| 3 | $\langle (30\;50\;70)\rangle$ |
| 4 | $\langle (30)(40\;70)(90)\rangle$ |
| 5 | $\langle (90)\rangle$ |

> **How to read it** (lecture): **customer 1** purchased item **30 the first time** and item **90 the second time**, so their sequence has **two elements, both single items**. **Customer 2** is more interesting: **the first time they purchased 2 products**, so the first element is an **itemset**.

**The two observations printed on the slide:**
- **$\{(10\;20)(30)\}$ does NOT have `min_sup`** — **only supported by customer 2**.
- **$\{(30)\}$, $\{(70)\}$, $\{(30)(40)\}$ … are NOT MAXIMAL.**

> **The lecture's point about the second observation**: *"if we consider 30 separately it is frequent, but it is not maximal because we have the sequence $\{(30)(40)\}$. This means that **if we return $\{(30)(40)\}$ as a frequent sequence, we do not return the support for 30 and for 40**. This is an example showing that **if we return only the maximal sequences, we LOSE the information about the frequency of the single elements of the sequence.**"*

### ⚠️ Where does a sequence database actually come from?

> **The practical question raised in class**: *in reality, can a supermarket obtain a sequence database from its transaction database?* **The problem is identifying the single user.** The answer: **the FIDELITY CARD**, or identification **by credit card**. *"This means that this sequence database can really be obtained in a supermarket — most of the time using the fidelity card."*
>
> **So the pipeline is: transaction database → (customer identification) → sequence database.** This is exactly what the *Sort phase* below does.

## 5. The five phases of the Apriori-style approach

> **We will analyse THREE versions of a sort-of-Apriori algorithm**: first **the natural version**, then **two modifications aimed at speeding up the process.**

The classical (Agrawal & Srikant) approach runs in **five phases**:

1. **Sort phase** 2. **Litemset phase** 3. **Transformation phase** 4. **Sequence phase** 5. **Maximal phase**

### 5.1 Sort phase

> **Sort the database with the CUSTOMER ID as the major key and the TRANSACTION-TIME as the minor key.**
> **This converts the original transaction database into a database of CUSTOMER SEQUENCES.**

*(As noted above, this is possible because we can identify customers by fidelity/credit card. The sort phase exists precisely because **we normally start from a transaction database and want to create a sequence database**.)*

### 5.2 Litemset phase

> **A LITEMSET (large itemset) is an itemset supported by a fraction of CUSTOMERS larger than `min_sup`** — i.e. **litemsets are the frequent itemsets.**

> **⚠️ Recall: EACH ITEMSET IN A LARGE SEQUENCE HAS TO BE A LARGE ITEMSET.**
> **Why** (lecture): *"it is not possible that a single element of a frequent sequence is not frequent. **This derives from the Apriori property**: each subsequence of a frequent sequence has to be frequent. So in any case, **to build frequent sequences we have to start from frequent elements**."*

**Support counting is measured by the FRACTION OF CUSTOMERS** who purchased that itemset.

**The worked litemset analysis** on the database of §4, `min_sup` = 25% ⇒ **at least 2 customers**:

| Candidate | Supported by | Frequent? |
|---|---|---|
| $(30)$ | customers 1, 2, 3, 4 | **✓ frequent** |
| $(90)$ | customers 1, 4, 5 | **✓ frequent** |
| $(10\;20)$, $(10)$, $(20)$ | only customer 2 | ✗ |
| $(40\;60\;70)$, $(40\;60)$, $(60\;70)$ | — | ✗ |
| $(40\;70)$ | customers 2, 4 | **✓ frequent** |
| $(40)$ | customers 2, 4 | **✓ frequent** |
| $(70)$ | customers 2, 3, 4 | **✓ frequent** |
| $(30\;50)$, $(30\;70)$, $(50\;70)$ | — | ✗ |

> **⚠️ Note carefully** (lecture): **we have $(40)$ separately, $(70)$ separately AND $(40\;70)$ as an itemset.** This is because **in our sequences we can have $(40\,70)$ as an itemset and also $40$ or $70$ separately.** Missing this is the classic mistake in the transformation phase.

**Then: each large itemset is mapped to a set of CONTIGUOUS INTEGERS** — **used to compare large itemsets in CONSTANT TIME and to reduce the time required to check whether a sequence is contained in a customer sequence.**

| Litemset | Mapped to |
|---|---|
| $(30)$ | **1** |
| $(40)$ | **2** |
| $(70)$ | **3** |
| $(40\;70)$ | **4** |
| $(90)$ | **5** |

### 5.3 Transformation phase

> **We need to repeatedly determine which of a given set of large sequences are contained in a customer sequence. To make this fast:**
> - **replace each transaction with ALL THE LITEMSETS CONTAINED IN IT**;
> - **transactions with no litemsets are DROPPED** — **but still considered for support counts** (the customer stays in the denominator).

| CID | Original sequence | After transformation | After mapping |
|---|---|---|---|
| 1 | $\langle (30)(90)\rangle$ | $\langle \{(30)\}\{(90)\}\rangle$ | $\langle \{1\}\{5\}\rangle$ |
| 2 | $\langle (10\,20)(30)(40\,60\,70)\rangle$ | $\langle \{(30)\}\{(40),(70),(40\,70)\}\rangle$ | $\langle \{1\}\{2,3,4\}\rangle$ |
| 3 | $\langle (30\,50\,70)\rangle$ | $\langle \{(30)\},\{(70)\}\rangle$ | $\langle \{1,3\}\rangle$ |
| 4 | $\langle (30)(40\,70)(90)\rangle$ | $\langle \{(30)\}\{(40),(70),(40\,70)\}\{(90)\}\rangle$ | $\langle \{1\}\{2,3,4\}\{5\}\rangle$ |
| 5 | $\langle (90)\rangle$ | $\langle \{(90)\}\rangle$ | $\langle \{5\}\rangle$ |

**The two notes on the slide**: **$(10\;20)$ dropped for lack of support**; **$(40\;60\;70)$ replaced by $\{(40),(70),(40\;70)\}$ because $(60)$ does not have `min_sup`.**

> **The lecture's walk-through of customer 2**: *the original element $(10,20)$ is not a large itemset, so **we remove it**. Then $(30)$ is maintained. Then $(40,60,70)$: only $(40,70)$ is large, **so we remove 60** — but since we can also have 40 and 70 separately, **in the transformed sequence we use all THREE litemsets: $(40)$, $(70)$ and $(40\,70)$**.*

> **⚠️ Understand what a transformed transaction IS**: no longer an itemset, but **the SET OF ALL LITEMSETS contained in it**. Checking "does this customer support sequence $X$?" then becomes **membership tests on integers**.

### 5.4 Sequence phase

> **Use the set of large itemsets to find the desired sequences.** **Similar structure to the Apriori algorithm:** **use a seed set to generate candidate sequences → count support for each candidate → eliminate the candidates that are not large.**

> **The philosophy** (lecture): *"we start from one sequence, we generate subsequences and so on; **we increase the number of elements in the sequences to check whether they are frequent**. The strategy is the same as the Apriori property: **if we want a 2-element sequence to be frequent, the single elements must be frequent**. To create sequences with more elements, we must start from subsequences that are frequent."*

**Two families of algorithms:**

| Family | Idea | Algorithms |
|---|---|---|
| **Count-all** — *the application of the Apriori algorithm* | **count ALL large sequences, including NON-MAXIMAL ones** — *careful with respect to the minimum support* | **AprioriAll** |
| **Count-some** | **avoid counting non-maximal sequences by counting LONGER sequences FIRST** — *careful with respect to maximality* | **AprioriSome**, **DynamicSome** |

### 5.5 Maximal phase

**Find the maximal sequences among the large sequences.** With $S$ the set of all large sequences and $n$ the maximum length:

```
for (k = n; k > 1; k--) do
    foreach k-sequence s_k do
        delete from S all subsequences of s_k
```

**This leaves in $S$ only the maximal sequences.** **Data structures and an algorithm exist to do this efficiently (HASH TREES).**

## 6. AprioriAll (count-all)

> *"This is the easiest version to mine the frequent patterns."*

```
L_1 = {large 1-sequences}
for (k = 2; L_{k-1} != {}; k++) do
begin
    C_k = New candidates generated from L_{k-1}
    foreach customer-sequence c in the database do
        Increment the count of all candidates in C_k that are contained in c
    L_k = Candidates in C_k with minimum support
end
Answer = Maximal Sequences in  U_k L_k
```
*Notation: $L_k$ = large $k$-sequences; $C_k$ = candidate $k$-sequences.*

> **The loop in words** (lecture): starting from the **$(k-1)$-frequent sequences we generate the candidate $k$-sequences**. To decide whether they are frequent **we must SCAN the sequence database and count the number of customers in which the sequence appears**. At the end, **$L_k$ holds the candidates of $C_k$ with minimum support**. **Finally the maximal phase reduces $\bigcup_k L_k$ to the maximal sequences.**

### 6.1 Candidate generation

**Step 1 — JOIN two sequences in $L_{k-1}$ to generate $C_k$:**
> **For each two sequences in $L_{k-1}$ that have the SAME 1st to $(k-2)$-th itemsets, select the 1st to $(k-1)$-th itemsets from the first sequence and join with the LAST itemset from the other.**

**Example:** $L_3 = \{123\},\{234\},\{124\},\{134\},\{135\}$

> **The lecture's walk-through**: we consider sequences **sharing the same first two itemsets**. Here **the 1st and the 3rd ($\{123\}$ and $\{124\}$) share the first two itemsets $1$ and $2$**, so joining them gives **$\{1234\}$ but ALSO $\{1243\}$**.
> **⚠️ We must consider BOTH cases, because we are talking about SEQUENCES and the order between elements matters** — and since we don't know in advance which occurs, **we must test both candidates**. When we scan the sequence database we will find out whether both are possible or only one.

$$C_4 = \{1\,2\,3\,4\},\;\{1\,2\,4\,3\},\;\{1\,3\,4\,5\},\;\{1\,3\,5\,4\}$$

**Step 2 — PRUNE:** **delete all sequences in $C_k$ if some of their subsequences are not in $L_{k-1}$.**

> **Applied**: $\{1\,3\,4\,5\}$ is deleted because **$\{3\,4\,5\}$ is not in $L_3$** — **by the Apriori property, if a subsequence of the sequence we are analysing is not frequent, the sequence cannot be frequent.** Likewise for the others.

$$\Rightarrow\; C_4 = \{1\,2\,3\,4\} \quad \text{— the only 4-candidate derivable from } L_3$$

### 6.2 The full worked example

**Transformed customer sequences** (`min_sup` = 25%, i.e. 2 customers):

| CID | Customer sequence |
|---|---|
| 1 | $\langle (1\,5)(2)(3)(4)\rangle$ |
| 2 | $\langle (1)(3)(4)(3\,5)\rangle$ |
| 3 | $\langle (1)(2)(3)(4)\rangle$ |
| 4 | $\langle (1)(3)(5)\rangle$ |
| 5 | $\langle (4)(5)\rangle$ |

| Level | Large sequences | Note from the lecture |
|---|---|---|
| **$L_1$** | $\langle 1\rangle{:}4$, $\langle 2\rangle{:}2$, $\langle 3\rangle$, $\langle 4\rangle$, $\langle 5\rangle$ | all are large itemsets |
| **$L_2$** | $\langle 12\rangle, \langle 13\rangle, \langle 14\rangle, \langle 15\rangle, \langle 23\rangle, \langle 24\rangle, \langle 34\rangle, \langle 35\rangle, \langle 45\rangle$ | **all frequent EXCEPT $\langle 2\,5\rangle$, which has support 0** |
| **$L_3$** | $\langle 123\rangle, \langle 124\rangle, \langle 134\rangle, \langle 135\rangle, \langle 234\rangle$ | **5 frequent sequences** |
| **$L_4$** | $\langle 1234\rangle{:}2$ | **we stop here — the join step cannot be applied again** |

> **Two details from the lecture:**
> - At level 2, *"theoretically we should generate $\langle 2\,1\rangle$, $\langle 3\,1\rangle$ and so on, but **we can see they are not present in the customer sequences**"* — the deck **skips drawing $C_2$** and shows $L_2$ directly.
> - At level 3, joining $\langle 1\,2\rangle$ with $\langle 1\,3\rangle$ gives $\langle 1\,2\,3\rangle$ **and** $\langle 1\,3\,2\rangle$; **$\langle 1\,3\,2\rangle$ is a candidate but cannot be frequent, because the subsequence $\langle 3\,2\rangle$ is not among the frequent sequences.**

**ANSWER (maximal phase): $\langle 1\,2\,3\,4\rangle$, $\langle 1\,3\,5\rangle$, $\langle 4\,5\rangle$.**

### 6.3 ⚠️ Closed vs. maximal, worked on this very example

This is the passage that makes §4 concrete — **know it**.

| Sequence | Closed? | Maximal? | Reason |
|---|---|---|---|
| $\langle 1234\rangle$ | **✓** | **✓** | the longest found; no supersequence at all |
| $\langle 123\rangle$ | ✗ | ✗ | **has the supersequence $\langle 1234\rangle$ WITH THE SAME SUPPORT** |
| $\langle 124\rangle$ | ✗ | ✗ | same reason |
| $\langle 134\rangle$ | **✓** | ✗ | **closed because its support is HIGHER than that of its supersequence**, but it is contained in $\langle 1234\rangle$ |
| $\langle 135\rangle$ | **✓** | **✓** | **no supersequence exists** — the only maximal one in $L_3$ |
| $\langle 234\rangle$ | ✗ | ✗ | absorbed |
| $\langle 45\rangle$ (in $L_2$) | — | **✓** | maximal |
| everything in $L_1$ | — | ✗ | **all 5 items are included in some supersequence ⇒ no maximal sequence at level 1** |

> **The lecture also notes**: performing the same analysis on $L_2$ yields **a good number of closed sequences** — which is precisely why, in practice, one falls back on maximal sequences.

### 6.4 ⚠️ The cost, and the role of `min_sup`

> **The computational time is not negligible: FOR EACH $k$ WE MUST SCAN THE SEQUENCE DATABASE.** With many sequences this is expensive.
>
> **The one parameter that governs everything is `min_sup`**, and it controls the complexity through the length of the sequences:
> - **HIGH `min_sup`** (say 80% of customers) ⇒ **as the number of elements grows, the probability of remaining frequent DROPS VERY FAST** ⇒ **we stop after 4–5 elements** ⇒ **few scans, light computation.**
> - **LOW `min_sup`** ⇒ **the opposite**: **sequences will have many elements and we will scan the database several times.**
>
> **But `min_sup` cannot simply be set high: its value depends on the application domain.**
>
> *(The underlying principle, restated: **if a sequence is not frequent, its supersequences cannot be frequent either** — if nobody buys bread, the sequence "bread then salami" cannot be frequent. This is how the combination tree gets pruned. It is also intuitively clear that a 2-product sequence is far more likely than a 7-product one to appear in 80% of customers.)*

## 7. AprioriSome (count-some)

> **The motivation** (lecture): *"if we think about what our final aim is — **to return the MAXIMAL sequences** — we can think about how to reduce the computational cost by arriving at the maximal sequences FASTER."*

> **The idea: avoid counting non-maximal sequences by COUNTING LONGER SEQUENCES FIRST.**

**Two additional phases:**
- **Forward phase** — **find all large sequences of certain lengths**;
- **Backward phase** — **find all the remaining large sequences.**

> *For example, we might count sequences of length **1, 2, 4 and 6** in the forward phase and lengths **3 and 5** in the backward phase — **but we can exploit the maximal sequences with 6 elements to remove directly the sequences with 5 elements** when generating the backward phase.*
>
> **In other words**: **jump from a length to length $+3$, $+4$; generate and verify the longer candidate sequences; then use them to remove all the candidates we are sure are included in a maximal sequence.**

### 7.1 The `next()` function and the hit ratio

**`next()` takes as parameter the length of the sequence counted in the last pass** and computes the next length to count.
- **`next(k) = k + 1` ⇒ exactly AprioriAll.**
- **`next(k) = k + constant` ⇒ we differ from AprioriAll and expect to reduce computation**, because we can **generate maximal sequences and use them to remove sequences with fewer elements**.

**The tuning quantity:**
$$\text{hit}_k = \frac{|L_k|}{|C_k|}$$

> **The rule as given in the lecture:**
> - **$\text{hit}_k < 0.666$ ⇒ use the classical AprioriAll** (jump of 1);
> - **$0.666 \le \text{hit}_k < 0.75$ ⇒ jump of 2**; *and so on with increasing jumps;*
> - **$\text{hit}_k > 0.85$ ⇒ the largest jump** — here **$|L_k| \approx |C_k|$**, i.e. **practically all the candidates we generate are also frequent.**
>
> **Intuition: as $\text{hit}_k$ INCREASES, the time wasted by counting extensions of small candidates DECREASES.**
>
> *(The deck shows the full step table as an image; the standard Agrawal–Srikant version is $k{+}1$ / $k{+}2$ / $k{+}3$ / $k{+}4$ / $k{+}5$ for $\text{hit}_k$ below 0.666, then 0.666–0.75, 0.75–0.80, 0.80–0.85, and $\ge 0.85$ — consistent with the three anchors quoted in class.)*

**`next()` balances a TRADE-OFF between: counting non-maximal sequences** (AprioriAll's waste) **and counting extensions of small candidate sequences** (the waste incurred by skipping).

### 7.2 Forward phase

```
L_1 = {large 1-sequences}
C_1 = L_1
last = 1
for (k = 2; C_{k-1} != {} and L_last != {}; k++) do
begin
    if (L_{k-1} known) then
        C_k = New candidates generated from L_{k-1}
    else
        C_k = New candidates generated from C_{k-1}

    if (k == next(last)) then begin        // is this the next k to count?
        foreach customer-sequence c in the database do
            Increment the count of all candidates in C_k that are contained in c
        L_k = Candidates in C_k with minimum support
        last = k
    end
end
```

> **⚠️ The crucial subtlety, on its own slide**: **if $L_{k-1}$ is NOT available, we use the CANDIDATE set $C_{k-1}$ to generate $C_k$. Correctness is maintained because $L_{k-1} \subseteq C_{k-1}$.**
>
> **So: we always generate all the candidate sequences — from $L_{k-1}$ when we have it, from $C_{k-1}$ otherwise — but we only build $L_k$ when $k = \text{next}(\text{last})$.** The price is that **$C_k$ is generated from a superset, hence larger.**

### 7.3 Backward phase

> **For all the lengths we skipped:**
> - **delete the sequences in the candidate set that are CONTAINED IN SOME LARGE SEQUENCE** (they cannot be maximal);
> - **count the remaining candidates and find those with minimum support**;
> - **also delete the large sequences found in the forward phase that are NON-MAXIMAL.**

```
for (k--; k >= 1; k--) do
    if (L_k not found in forward phase) then begin
        Delete all sequences in C_k contained in some L_i, i > k
        foreach customer-sequence c in D_T do
            Increment the count of all candidates in C_k that are contained in c
        L_k = Candidates in C_k with minimum support
    end
    else            // L_k already known
        Delete all sequences in C_k contained in some L_i, i > k

Answer = U_k L_k        // (Maximal Phase NOT needed)
```
*Notation: $D_T$ = transformed database.*

> **⚠️ Note the pay-off on the last line: the MAXIMAL PHASE IS NOT NEEDED** — the backward phase has already removed every non-maximal sequence.

### 7.4 The worked example, with `next(k) = 2k`

**Forward phase:**
1. Start with **$L_1$**, generate and count **$L_2$**.
2. Apply `next`: since **$\text{next}(2) = 4$**, the next length to COUNT is 4.
3. To reach $C_4$ we must first **generate $C_3$ from $L_2$ — but we do NOT check whether the $C_3$ sequences are frequent**, and **we do not build $L_3$**.
4. **Generate $C_4$ directly from the candidate set $C_3$**, and **count it**: the only frequent 4-sequence is **$\langle 1\,2\,3\,4\rangle$**.
5. We stop generating: **$L_4$ contains a single sequence**, so the backward phase begins.

**Backward phase:**
1. Start from the last level generated and **delete from $C_3$ all the sequences contained in $L_4 = \langle 1234\rangle$** — so $\langle 123\rangle$, $\langle 124\rangle$, $\langle 134\rangle$, $\langle 234\rangle$ all go, **because we are only interested in maximal sequences.**
2. **The survivors are $\langle 1\,2\,5\rangle$, $\langle 1\,3\,5\rangle$, $\langle 3\,4\,5\rangle$.** For these three we **scan the sequence database** and discover that **only $\langle 1\,3\,5\rangle$ is frequent.**

> **The advantage, in the lecture's words**: **for at least part of the generation path, candidate sequences are generated from candidate sequences with $k-1$ elements** — so **we do NOT verify all the sequences against the database; we verify only a SUBSET, because we exploit the backward mechanism.**

## 8. AprioriDynamicSome

> **Like AprioriSome, it skips counting candidate sequences of certain lengths in the forward phase — but the lengths counted are determined by the variable `step`.**

**The phases:**

**(1) Initialization phase** — **all the candidate sequences of length up to and including `step` are counted**, using the AprioriAll philosophy.

**(2) Forward phase** — **all sequences whose lengths are MULTIPLES of `step` are counted.**

> **With `step = 3`**: count lengths **1, 2, 3 in the initialization phase**, and **6, 9, 12, … in the forward phase.**
> **We can generate sequences of length 6 by joining sequences of length 3; length 9 by joining length 6 with length 3; etc. But to generate the sequences of length 3 we need lengths 1 and 2 — hence the initialization phase.**
>
> **Operationally** (lecture): start with $k = \text{step}$ and proceed **until $L_k$ is empty**, **generating $L_{k+\text{step}}$ from $L_k$ and $L_{\text{step}}$** via a specific generation function.

**(3) Intermediate phase and (4) Backward phase** — **count the sequences for the lengths skipped in the forward phase.**

> **⚠️ The difference from AprioriSome**: **these candidate sequences were NOT GENERATED in the forward phase. The INTERMEDIATE phase generates them.**
>
> **The slide's trace**: assume we count $L_3$ and $L_6$, and **$L_9$ turns out empty**. We **generate $C_7$ and $C_8$ (intermediate phase)**, then **count $C_8$ followed by $C_7$** after deleting non-maximal sequences **(backward phase)**. **The process is then repeated for $C_4$ and $C_5$.** *(Longest first, so shorter candidates can be discarded as non-maximal.)*

### 8.1 The `otf-generate` procedure

**The join condition is**
$$X_2.\text{end} < X_2.\text{start}$$
*(where $X_2$ denotes the set of sequences of length 2)* — **we consider the START and the END element of the sequences, and combine only when the END of one is lower than the START of the other.** In the deck's example this lets us **join $\langle 1\,2\rangle$ with $\langle 3\,4\rangle$**, and the result of the join is **the single sequence $1\,2\,3\,4$.**

**Why do we need `otf-generate`?**
1. **The `apriori-generate` procedure used for AprioriSome could generate MORE candidates.** *(It would also need generalizing to produce $C_{k+j}$ from $L_k$: the join condition must require equality of the first $k-j$ terms and concatenation of the remaining ones.)*
2. **If $|L_k| + |L_{\text{step}}|$ is smaller than $|C_{k+\text{step}}|$ as generated by AprioriSome, it may be FASTER to find all members of $L_k$ and $L_{\text{step}}$ contained in $c$ than all members of $C_{k+\text{step}}$.**
3. **The intuition**: **if $s_k \in L_k$ and $s_j \in L_j$ are both contained in $c$ and they DO NOT OVERLAP in $c$, then $\langle s_k . s_j\rangle$ is a candidate $(k+j)$-sequence.**

> **⚠️ THE PROBLEM WITH DYNAMICSOME, stated bluntly in the lecture**: *"the big difference with respect to AprioriSome is that **when we generate, say, $C_6$ starting from $C_3$, we just generate ALL the possible combinations** starting from $C_3$ — **so we generate a LOT of candidate sequences. This is the big problem we have.**"*

## 9. Performance, and the bottleneck

**Setting**: generated (synthetic) datasets; we are not interested in the datasets themselves but in the relative performance.

**The findings:**
- **AprioriSome does a little better than AprioriAll** — **the backward phase does help reduce computational time**, **but the advantages are not fantastic.** It **avoids counting many non-maximal sequences.**
- **⚠️ The advantage of AprioriSome is REDUCED for two reasons: (1) AprioriSome GENERATES MORE CANDIDATES; (2) candidates remain MEMORY RESIDENT even if skipped over.**
- **⚠️ DynamicSome DOES NOT WORK: it generates TOO MANY candidates.** *"If we eliminate the intermediate steps we don't have any decrease in the possible generation of candidate sequences."* In the plots **its line simply BREAKS OFF as the minimum support decreases — the computational cost becomes too high to finish.**
- Across different datasets and parameter combinations **the behaviour is very similar.**

> **The lecture's conclusion**: **AprioriAll does generate the frequent sequences, but it is not good computationally — when the minimum support decreases, the time increases a lot. Of the two speed-ups proposed, AprioriSome works a little better than AprioriAll; DynamicSome does not work.**

### Bottlenecks of Apriori-like methods

1. **A HUGE set of candidates could be generated.**
2. **MANY SCANS of the database in mining.**
3. **Difficulty when mining LONG sequential patterns** — an **exponential number of short candidates**:
   $$\sum_{i=1}^{100}\binom{100}{i} = 2^{100} - 1 \approx 10^{30} \quad\text{candidates for a length-100 pattern}$$

> **⚠️ The key sentence that motivates all of §10**: *"**MOST of these candidate sequences are NOT frequent**, so we have to find a way to **directly generate candidate sequences that are also frequent**. We did exactly this in frequent pattern analysis by introducing **FP-growth** — and we will see that we can do something similar for sequences, with an approach known in the literature as **FreeSpan**."*

## 10. FreeSpan: FP-growth for sequential pattern mining

> *J. Han, J. Pei, B. Mortazavi-Asl, Q. Chen, U. Dayal, M.-C. Hsu, "**FreeSpan: Frequent Pattern-Projected Sequential Pattern Mining**", KDD'00.*

### 10.1 The recap that opened the 14/04 lecture

> *Our goal is to explore sequence patterns and find frequent sequences. We began by transforming transactions into sequences. We saw three Apriori-based approaches, but **they are computationally expensive, especially when `min_sup` is LOW**.*
>
> *In standard frequent pattern analysis **we hit the same bottleneck: CANDIDATE GENERATION** — which is why Apriori struggles at low thresholds. **FP-Growth** was introduced to **avoid generating every possible candidate itemset**, instead **guiding the generation so that the candidates produced are already frequent.***
>
> ***With sequences we want the same philosophy — but there are key differences: it is quite difficult to produce an FP-tree for sequential data, so the standard tree construction does not work as effectively here.***

**Hence the three findings on the slide:**
- **a straightforward construction of a sequential-pattern tree does NOT work well;**
- **a level-by-level projection does NOT achieve high performance either;**
- **an interesting method is to explore ALTERNATIVE-LEVEL PROJECTION.**

### 10.2 The `f_list` and the partition of the pattern space

**Find the frequent items; the list of frequent items in SUPPORT-DESCENDING order is the `f_list`.**

**The running example (SDB):**
```
< (bd) c b (ac) >
< (bf) (ce) b (fg) >
< (ah) (bf) a b f >
< (be) (ce) d >
< a (bd) b c b (ade) >
```
$$\texttt{f\_list}:\quad b{:}5,\; c{:}4,\; a{:}3,\; d{:}3,\; e{:}3,\; f{:}2 \qquad (\texttt{min\_sup} = 2;\; g,h \text{ are infrequent})$$

> **All sequential patterns can be divided into several subsets WITHOUT OVERLAP.** **To ensure the subsets do not overlap, we define them by the LEAST FREQUENT item they contain**, and **we start the projection from the LEAST frequent item** (here $f$):
> - patterns containing **$f$**;
> - those containing **$e$ but no $f$**;
> - those containing **$d$ but no $e$ nor $f$**;
> - those containing **$a$ but no $d$, $e$ or $f$**;
> - those containing **$c$ but no $a$, $d$, $e$ or $f$**;
> - those containing **only $b$**.

### 10.3 Projected databases

> **The complete set of sequential patterns containing item $i$ but NO items FOLLOWING $i$ in the `f_list` can be found in the $i$-PROJECTED DATABASE.**
> **A sequence $s$ is projected as $s_i$ into the $i$-projected database if there is at least one item $i$ in $s$**, where **$s_i$ is a copy of $s$ with (a) all the INFREQUENT items removed and (b) every frequent item $j$ FOLLOWING $i$ in the `f_list` removed.**

**Worked example — projecting $\langle (ah)(bf)abf\rangle$:**

| Projection | Action | Result |
|---|---|---|
| **$f$-projected** | **keep $f$ and all items MORE frequent than it** ($b,c,a,d,e$); remove infrequent items (like $h$) | $\langle a(bf)abf\rangle$ |
| **$a$-projected** | **keep $a$ and items more frequent than it** ($b,c$); remove $d,e,f$ and the infrequent $h$ | $\langle a(b)ab\rangle$, simplified as $\langle abab\rangle$ |
| **$b$-projected** | **nothing is more frequent than $b$** | $\langle bb\rangle$ |

> **Why it works** (lecture): **by projecting this way, FreeSpan focuses only on the relevant parts of the data for a specific subset of patterns. This significantly reduces the size of the database searched at each step and avoids the "COMBINATORIAL EXPLOSION" of candidates that makes standard Apriori so expensive.**

### 10.4 Parallel vs. partition projection

> **The shift from parallel to partition projection is a classic trade-off between SPEED and SPACE efficiency.**

**PARALLEL PROJECTION** — **scan the database once and immediately create ALL the $i$-projected databases for every frequent item found in a sequence.**

> **⚠️ The cost, computed on the slide**: **let each transaction contain on average $l$ frequent items. It is then projected into $l-1$ projected databases, and the total size of the projected data from that transaction is**
> $$1 + 2 + \dots + (l-1) = \frac{l(l-1)}{2}$$
> **so the total size of the single-item-projected databases is about $\dfrac{l-1}{2}$ times that of the original database.** **We greatly increase memory occupation, because we REPLICATE subsequences.**
>
> *(The lecture's analogy: it is like **photocopying a chapter of a book for every single student in a class all at once** — fast, because you open the book only once, but you end up with a massive pile of paper.)*

**PARTITION PROJECTION** — proposed to avoid that overhead; **it ensures each sequence is stored in only ONE projected database at any given time.**

> - **Project a sequence into the projected database of the LAST frequent item in it.**
> - **When scanning, a transaction $T$ is projected into the $a_i$-projected database ONLY IF $a_i$ is a frequent item in $T$ and there is no other item after $a_i$ in the list of frequent items appearing in the transaction.**
> - **Since a transaction is projected into only one projected database per scan, after the scan the database is PARTITIONED by projection** — hence the name.
> - **To ensure the remaining projected databases obtain the complete information**, each time a projected database is processed, **each transaction in it is projected into the $a_j$-projected database**, where $a_j$ is the item such that no other item follows it in the list of frequent items of that transaction. **Sequences are "PROPAGATED" ON-THE-FLY.**

> **In other words** (lecture): **when you finish processing the $i$-projected database (e.g. for item $f$), you don't just delete the data — you PASS it on to the next relevant item in the `f_list` (e.g. $e$).** This creates a **"sliding" effect**: **instead of having $N$ databases open at once, you have a set of partitions that are processed and then re-allocated.**

**The comparison table from the lecture:**

| Feature | **Parallel projection** | **Partition projection** |
|---|---|---|
| **Database scans** | **Fewer** (efficient scanning) | **More frequent** (on the fly) |
| **Memory usage** | **Very high** (exponential growth) | **Low** (sequential / dynamic) |
| **Data handling** | **Replicates** sequences | **Partitions / propagates** sequences |
| **Best for…** | **small datasets with few items** | **large-scale sequence mining** |

### 10.5 Mining by level-by-level projected databases

**Algorithm:**
1. **Scan the database once, find the frequent items, get the `f_list`.**
2. **Recursively do database projection LEVEL BY LEVEL.**

**Worked trace on the $f$-projected DB** (with `min_sup` = 2, we find the two frequent items $b$ and $f$, then generate and test the possible 2-element sequences):
```
f-projected database:
    < (bf) (ce) b f >
    < a (bf) a b f >
    Frequent items: b, f
    Seq. patterns: <bf>, <fb>, <(bf)>, <ff>, <(bf)f>, <fbf>
    One more scan for {b, b, f}:  <(bf)b>, <bbf>, <(bf)bf>
```

**Pros and cons:**

| | |
|---|---|
| **Benefits** | **only need to find frequent ITEMS in each projected database, instead of exploring candidate SEQUENCE generation**; **the number of combinations is much smaller than all the possible combinations** |
| **Cost** | **partition and projection of the databases** |
| **Regime** | **works well in SPARSE databases** |

### 10.6 Mining by alternative-level projected databases

> **Postpone the generation of projected databases: take each database as a LEVEL-SHARED, COMBINED projected database.**

**Algorithm:**
1. **Scan the database to find the frequent items and get the `f_list`.**
2. **Perform alternative-level projection mining:**
   - **construct the FREQUENT ITEM MATRIX** — *this matrix gives us the possibility to generate the length-2 sequential patterns and the annotations on item-repeating patterns and projected databases*;
   - **scan the database to generate the item-repeating patterns and the projected databases**;
   - **do matrix projection mining on the projected databases RECURSIVELY**, if there are still longer candidate patterns to mine.

#### The frequent item matrix

> **A TRIANGULAR matrix $F[j,k]$, $1 \le j \le m$, $1 \le k \le j$, with $m$ the number of frequent items.**
> - **$F[j,j]$ has ONE counter, recording the appearance of the sequence $\langle jj\rangle$.**
> - **$F[j,k]$ has THREE counters $(A,B,C)$:**
>   - **$A$: number of occurrences in which $k$ occurs AFTER $j$ — $\langle jk\rangle$;**
>   - **$B$: number of occurrences in which $k$ occurs BEFORE $j$ — $\langle kj\rangle$;**
>   - **$C$: number of occurrences in which $j$ occurs CONCURRENTLY with $k$ — $\langle (jk)\rangle$.**

> **⚠️ HOW TO ORIENT THE MATRIX — the lecture makes this explicit:** **$j$ is the $x$-axis (the COLUMN) and $k$ is the $y$-axis (the ROW).**
> So for a cell at **row $k$, column $j$** the triple reads
> $$(A,B,C) = \big(\sigma\langle j\,k\rangle,\;\; \sigma\langle k\,j\rangle,\;\; \sigma\langle (jk)\rangle\big) \quad\text{— the COLUMN item comes first in } A.$$
> **The lecture's own check**: taking **$j = b$ (column) and $k = e$ (row)**, the cell $(3,1,1)$ says **$\langle be\rangle$ occurs 3 times in the SDB, $\langle eb\rangle$ once, $\langle (be)\rangle$ once.**
> **The diagonal represents $\langle jj\rangle$** — e.g. **$\langle bb\rangle$ occurs in the first, second, third and last sequences ⇒ 4.**

**The slide's illustration**: **the first sequence $\langle (bd)\,c\,b\,(ac)\rangle$ increases the FIRST TWO counters of $F[b,c]$ by 1**, since **$\langle b\,c\rangle$ and $\langle c\,b\rangle$ — but NOT $\langle (bc)\rangle$ — occur there.**

**The complete matrix for the running example:**

```
b   4
c   (4,3,0)   1
a   (3,2,0)   (2,1,1)   2
d   (2,2,2)   (2,2,0)   (1,2,1)   1
e   (3,1,1)   (1,1,2)   (1,0,1)   (1,1,1)   1
f   (2,2,2)   (1,1,0)   (1,1,0)   (0,0,0)   (1,1,0)   2
     b         c         a         d         e        f
```

*(Cross-check on row $a$, column $b$: $(3,2,0)$ ⇒ $\langle ba\rangle{:}3$, $\langle ab\rangle{:}2$, $\langle (ab)\rangle{:}0$ — matching the length-2 table below.)*

#### Generating the length-2 sequential patterns

> **It is quite easy, since we only have to analyse the single entries of the matrix: for each counter, if the value is no less than `min_sup`, output the corresponding sequential pattern.** With `min_sup` $=2$ we look at every value $\ge 2$; **when the counter is 1 the corresponding 2-sequence is not frequent and it is useless to generate it.**

#### Annotations on item-repeating patterns

**For row $j$:**
- **if $F[j,j] \ge$ `min_sup`, generate $\langle jj{+}\rangle$** — **the counts of $\langle jjj\rangle$, $\langle jjjj\rangle$, … are registered in the next round;**
- **for a column $i \ne j$: if $F[i,i] \ge$ `min_sup`, generate $i{+}$** (**potentially more than one $i$ appears in the pattern**); **if $F[j,j] \ge$ `min_sup`, generate $j{+}$;**
- **⚠️ if ONLY ONE of the three counters of $F[i,j]$ is frequent, a SEQUENCE is used as the annotation; OTHERWISE a SET is used.**

> **Why the distinction matters**: it is **used to enhance STRING FILTERING**. **The annotation $\langle b\,f\rangle$ indicates there is no chance for the subsequence $\langle f\,b\rangle$ to survive — but this is not so for $\{b\,f\}$.**

> **The two examples, explained in the lecture:**
> - **$\langle b{+}e\rangle$**: **$b$ CAN be replicated in the same sequence** (the diagonal gives $F[b,b]=4$), **but it is not possible to have sequences like $\langle bee\rangle$ — which is why there is NO plus after $e$** (indeed $F[e,e]=1 <$ `min_sup`).
> - **$\{b{+}f{+}\}$**: **on the diagonal $b$ has 4 and $f$ has 2**, so **we can have $bbf$, $bff$ or even $bbff$.** *(And since all three counters of $F[f,b]=(2,2,2)$ are frequent, a SET is used.)*
>
> **"In this way we are addressing our search in a very specific way."**

#### Annotations on projected databases

**For row $j$:**
- **for each $i < j$: if $F[i,j]$, $F[k,j]$ and $F[i,k]$ (with $k<i$) may form a PATTERN-GENERATING TRIPLE** (**all the corresponding pairs are frequent**), **then $k$ is added to $i$'s projected column set;**
- **if there is a choice between sequence or set, SEQUENCE IS PREFERRED.**

**Example**: **$\langle (ce)\rangle{:}\{b\}$** — *in the matrix $\langle (ce)\rangle$ appears as an itemset twice, so it can really be projected only along $b$*, i.e. **generate the $\langle (ce)\rangle$-projected database with $\{b\}$ as the only item included.**

#### The complete length-2 result table

| Item | Length-2 seq. patterns | Ann. on repeating items | Ann. on projected DBs |
|---|---|---|---|
| **f** | $\langle bf\rangle{:}2$, $\langle fb\rangle{:}2$, $\langle (bf)\rangle{:}2$ | $\{b{+}f{+}\}$ | None |
| **e** | $\langle be\rangle{:}3$, $\langle (ce)\rangle{:}2$ | $\langle b{+}e\rangle$ | $\langle (ce)\rangle{:}\{b\}$ |
| **d** | $\langle bd\rangle{:}2$, $\langle db\rangle{:}2$, $\langle (bd)\rangle{:}2$, $\langle cd\rangle{:}2$, $\langle dc\rangle{:}2$, $\langle da\rangle{:}2$ | $\{b{+}d\}$, $\langle da{+}\rangle$ | $\langle da\rangle{:}\{b,c\}$, $\{cd\}{:}\{b\}$ |
| **a** | $\langle ba\rangle{:}3$, $\langle ab\rangle{:}2$, $\langle ca\rangle{:}2$, $\langle aa\rangle{:}2$ | $\langle aa{+}\rangle$, $\{a{+}b{+}\}$, $\langle ca{+}\rangle$ | $\langle ca\rangle{:}\{b\}$ |
| **c** | $\langle bc\rangle{:}4$, $\langle cb\rangle{:}3$ | $\{b{+}c\}$ | None |
| **b** | $\langle bb\rangle{:}4$ | $\langle bb{+}\rangle$ | None |

#### One more scan: item-repeating patterns and projected databases

**Based on the annotations, the SDB is scanned ONE MORE TIME.**

**The item-repeating patterns generated:**
$$\{\langle bbf\rangle{:}2,\; \langle fbf\rangle{:}2,\; \langle (bf)b\rangle{:}2,\; \langle (bf)f\rangle{:}2,\; \langle (bf)bf\rangle{:}2,\; \langle (bd)b\rangle{:}2,$$
$$\langle bba\rangle{:}2,\; \langle aba\rangle{:}2,\; \langle abb\rangle{:}2,\; \langle bcb\rangle{:}3,\; \langle bbc\rangle{:}2\}$$

**There are FOUR projected databases**: $\langle (ce)\rangle{:}\{b\}$, $\langle da\rangle{:}\{b,c\}$, $\{cd\}{:}\{b\}$, $\langle ca\rangle{:}\{b\}$.

> **The recursion rule:**
> - **for a projected database whose annotation contains exactly three items, its sequential patterns are obtained by a SIMPLE SCAN of the projected database;**
> - **for a projected database whose annotation contains MORE than three items, construct a frequent item matrix FOR THAT projected database and RECURSIVELY mine it by the alternative-level projection technique.**

**The four projected databases and their patterns:**

| Annotation | $\langle (ce)\rangle{:}\{b\}$ | $\langle da\rangle{:}\{b,c\}$ | $\{cd\}{:}\{b\}$ | $\langle ca\rangle{:}\{b\}$ |
|---|---|---|---|---|
| **Projected DB** | $\langle b(ce)b\rangle$, $\langle b(ce)\rangle$ | $\langle (bd)cbc\rangle$, $\langle (bd)cb(ac)\rangle$, $\langle (bd)bcbd\rangle$ | $\langle bcd\rangle$, $\langle (bd)bcba\rangle$ | $\langle bcba\rangle$, $\langle bbcba\rangle$ |
| **Seq. patterns** | $\langle bce\rangle{:}2$ | $\langle (bd)a\rangle{:}2$, $\langle dba\rangle{:}2$, $\langle (bd)ba\rangle{:}2$, $\langle (bd)cba\rangle{:}2$, $\langle dca\rangle{:}2$, $\langle (bd)ca\rangle{:}2$, $\langle dcba\rangle{:}2$ | $\langle bcd\rangle{:}2$, $\langle dcb\rangle{:}2$, $\langle (bd)cb\rangle{:}2$, $\langle (bd)bc\rangle{:}2$ | $\langle bca\rangle{:}2$, $\langle cba\rangle{:}2$, $\langle bcba\rangle{:}2$ |

## 11. Performance study and why FreeSpan wins

**Data sets**: 10 000 items. **Comparison algorithms:**
- **GSP**; **Improved GSP** (*using pattern growth to find the length-2 sequential patterns*); **FreeSpan-1** (*level-by-level projection*); **FreeSpan** (*alternative-level projection*).

**Results:**
- **Dataset 1, average 2.5 items per transaction**: *"it is very interesting that **when we decrease the support, FreeSpan can still achieve a result in reasonable time**"* — where the Apriori-family curves blow up.
- **Dataset 1, average 5 items per transaction**: **FreeSpan still works very well** even as the number of sequences grows (note the runtime scale rises to ~100 s instead of ~10 s, but the ranking is unchanged).
- **Scalability with the number of sequences**: **both FreeSpan and improved GSP are LINEARLY SCALABLE, but FreeSpan is much better.**
- **Scalability in large databases** (up to $10^6$ sequences): **FreeSpan remains linear** at all thresholds.

### Why does FreeSpan outperform Apriori-like methods?

> 1. **Efficient database projection**: **it recursively projects a large sequence database into a set of SMALLER, manageable projected databases**, based on the frequent items currently being mined.
> 2. **Reduced scanning costs**: **alternative-level projection minimizes the overhead of scanning multiple databases**, while **retaining the advantages of the Apriori-like 3-WAY CANDIDATE FILTERING** — finding a **"sweet spot"** between the two methodologies.
> 3. **Computational efficiency**: *"in many real-world scenarios **we are practically OBLIGED to use FreeSpan** (or similar projection-based methods), because **Apriori fails to converge or produce results in a reasonable timeframe** when dealing with low support thresholds or very long sequences."*
>
> **In short: while Apriori struggles with the "combinatorial explosion" of candidates, FreeSpan's ability to PARTITION THE SEARCH SPACE makes it the superior choice for sequence pattern exploration.**

---

## Key points / potential exam pitfalls

> **📌 The professor's stated exam question**: *"describe the Apriori algorithm, its advantages, and **why we explore some smarter solutions**."* — i.e. be able to tell the whole story: **AprioriAll → its cost → AprioriSome/DynamicSome as attempted fixes → why they barely help → candidate generation as THE bottleneck → FreeSpan's projection answer.**

### Definitions
- **⚠️ A sequence is an ORDERED LIST OF ITEMSETS. Order holds BETWEEN elements, NOT inside them.** The supermarket-bill argument is the professor's own justification.
- **An item appears at most once inside an itemset, but MAY recur in different elements** — we don't care how many pieces of a product were bought, but we do care whether bread was bought on **both** visits.
- **The subsequence relation requires BOTH increasing (non-contiguous allowed) indices AND element-wise SUBSET containment.** Be able to verify $\langle ad(ae)\rangle \preccurlyeq \langle a(bd)bcb(ade)\rangle$, remembering that **a single item can be taken out of a larger itemset**.
- **Support counts SEQUENCES (customers), not occurrences.** **Frequency $\sigma(S)$ is a RATIO; absolute support is a COUNT.**
- **Lexicographic order is a COMPUTATIONAL convention** (it makes the join easy), **not a semantic claim** — "milk before coffee" means nothing in practice.
- **⚠️ CLOSED vs. MAXIMAL — the highest-value distinction in this chapter:**
  - **closed** = no proper supersequence **with the SAME support** ⇒ **NO information is lost**: from a closed supersequence you can *derive* the absorbed subsequence's support;
  - **maximal** = not a subsequence of any other pattern; **support plays NO role in the definition** ⇒ **the support of the subsequences IS LOST**;
  - **we return maximal sequences anyway, because even the closed ones are usually too many.**
  - **$\sigma(S_\beta) \le \sigma(S_\alpha)$ whenever $S_\alpha \preccurlyeq S_\beta$** — downward closure in sequence form.
- **Be able to redo the closed/maximal table of §6.3**: $\langle 123\rangle$ not closed (same support as $\langle 1234\rangle$), $\langle 134\rangle$ closed but not maximal, $\langle 135\rangle$ **closed AND maximal**, $\langle 45\rangle$ maximal, nothing in $L_1$ maximal.

### The five phases
- **Sort (customer ID major, transaction-time minor) → Litemset → Transformation → Sequence → Maximal.**
- **⚠️ "Each itemset in a large sequence has to be a large itemset" — and the reason is the APRIORI PROPERTY**, not a convention.
- **⚠️ In the transformation phase a transaction becomes the SET OF ALL LITEMSETS it contains**: $(40\;60\;70) \to \{(40),(70),(40\,70)\}$ — **all three**, because in a sequence we may need $40$, $70$ or the pair. **$(60)$ is dropped for lack of support.** **Transactions with no litemsets are dropped BUT the customer still counts for support.**
- **Litemsets → contiguous INTEGERS so comparisons are $O(1)$.**
- **Count-all is careful about MINIMUM SUPPORT; count-some is careful about MAXIMALITY.**
- **Sequence databases are obtainable in practice via FIDELITY CARDS or credit-card identification** — a likely oral-exam aside.

### AprioriAll / AprioriSome / DynamicSome
- **Candidate generation joins sequences agreeing on the first $k-2$ itemsets, and because order matters ONE pair yields TWO candidates** ($\{123\}+\{124\} \Rightarrow \{1234\}$ **and** $\{1243\}$). **Then prune any candidate with a subsequence not in $L_{k-1}$** — e.g. $\{1345\}$ dies because $\{345\} \notin L_3$; $\langle 132\rangle$ dies because $\langle 32\rangle \notin L_2$.
- **Memorise the worked answer: $\langle 1234\rangle$, $\langle 135\rangle$, $\langle 45\rangle$**, and that **$\langle 2\,5\rangle$ is the one 2-candidate with support 0**.
- **⚠️ Cost: EVERY level requires a FULL SCAN of the sequence database.** And **the only parameter is `min_sup`**: **high ⇒ sequences die out after 4–5 elements (fast); low ⇒ long sequences and many scans (slow)** — but its value is dictated by the application domain.
- **`next(k)=k+1` reduces AprioriSome to AprioriAll.** **$\text{hit}_k = |L_k|/|C_k|$**, with the thresholds **0.666 / 0.75 / … / 0.85**: **higher hit ⇒ bigger jump**, because nearly all candidates are frequent anyway.
- **⚠️ In AprioriSome's forward phase, when $L_{k-1}$ is unknown, $C_k$ is generated from $C_{k-1}$; correctness holds because $L_{k-1} \subseteq C_{k-1}$** — the price is a **larger candidate set**.
- **⚠️ AprioriSome needs NO maximal phase** — the backward phase already deleted the non-maximal sequences. Its answer is $\bigcup_k L_k$.
- **Be able to retrace the `next(k)=2k` example**: count $L_1$, $L_2$; generate $C_3$ **without counting it**; generate $C_4$ **from $C_3$**; count ⇒ $L_4 = \langle 1234\rangle$; backward: delete from $C_3$ everything inside $\langle 1234\rangle$, leaving **$\langle 125\rangle, \langle 135\rangle, \langle 345\rangle$**, of which **only $\langle 135\rangle$ is frequent**.
- **DynamicSome uses `step`** (initialization counts lengths $\le$ `step`; forward counts multiples of `step`, generating $L_{k+\text{step}}$ from $L_k$ and $L_{\text{step}}$), and **its skipped candidates were NEVER generated — the INTERMEDIATE phase makes them.** Backward proceeds **longest-first** ($C_8$ before $C_7$).
- **`otf-generate`: join condition $X_2.\text{end} < X_2.\text{start}$; intuition — if $s_k$ and $s_j$ both occur in $c$ and DO NOT OVERLAP, their concatenation is a candidate $(k+j)$-sequence.**
- **⚠️ The performance verdict**: **AprioriSome beats AprioriAll only "a little"** (it generates more candidates, and **candidates stay memory-resident even when skipped**); **DynamicSome FAILS — it generates far too many candidates, and its curve breaks off at low support.**
- **The bottleneck: $2^{100}-1 \approx 10^{30}$ candidates for a length-100 pattern — and MOST candidates are NOT frequent**, which is the whole argument for projection-based mining.

### FreeSpan
- **A straightforward sequential-pattern TREE does not work** — this is *why* FreeSpan is not simply "FP-growth with order".
- **The `f_list` is SUPPORT-DESCENDING**, and the pattern space is split into **disjoint, exhaustive subsets defined by the LEAST FREQUENT item each pattern contains**, processed **from the least frequent item backwards**.
- **⚠️ Projection rule**: keep item $i$ **and everything MORE frequent than it**; drop infrequent items and every frequent item **following** $i$ in the `f_list`. Redo $\langle (ah)(bf)abf\rangle \to \langle a(bf)abf\rangle / \langle abab\rangle / \langle bb\rangle$.
- **Parallel projection: one scan, all projected DBs, total size $\approx \frac{l-1}{2}\times$ the original** (from $1+2+\dots+(l-1)$). **Partition projection: each sequence lives in exactly ONE projected DB (that of its LAST frequent item), propagated on the fly.** Know the **scans / memory / handling / best-for** comparison table.
- **⚠️ Frequent item matrix: the DIAGONAL has ONE counter ($\langle jj\rangle$); off-diagonal cells have THREE.** **$j$ = column ($x$-axis), $k$ = row ($y$-axis)**, so **$A = \sigma\langle j\,k\rangle$ counts the COLUMN item first.** Verify on $j{=}b$, $k{=}e$: $(3,1,1) \Rightarrow \langle be\rangle{:}3, \langle eb\rangle{:}1, \langle (be)\rangle{:}1$.
- **The sequence-vs-set annotation rule: exactly ONE frequent counter ⇒ SEQUENCE; more than one ⇒ SET.** Purpose: **string filtering** — $\langle bf\rangle$ rules out $\langle fb\rangle$, $\{bf\}$ does not. **For projected-DB annotations, SEQUENCE is preferred when there is a choice.**
- **The `+` marks a repeatable item and comes from the DIAGONAL**: $\langle b{+}e\rangle$ means $b$ may repeat but $e$ may not (so no $\langle bee\rangle$); $\{b{+}f{+}\}$ allows $bbf$, $bff$, $bbff$.
- **Level-by-level projection works well in SPARSE databases**; its cost is partitioning/projecting. **Alternative-level projection = FreeSpan proper.**
- **Why FreeSpan wins: recursive projection into small databases + alternative-level projection reducing scans + Apriori-like 3-way candidate filtering.** At low `min_sup` **Apriori-family methods simply do not finish.**

### Web usage mining
- **In web-log sequences the elements are SINGLE ITEMS** — a user can access only one page at a time.
- **⚠️ Server-side logs miss CACHED views, and there are TWO cache levels: the browser and the proxy.** Client-side logging fixes caching/session identification but needs an agent or modified browser installed on many users; proxy logs cover anonymous users sharing that proxy but **a user may go through different proxies**.
- **Cookies and sniffers are unreliable** (cookies can be rejected), so **researchers assume the server log is complete**.

### Cross-chapter connections
- **This chapter is Chapter 6 (Frequent Patterns) plus ORDER.** Apriori → AprioriAll; downward closure becomes $\sigma(S_\beta) \le \sigma(S_\alpha)$ for $S_\alpha \preccurlyeq S_\beta$; closed/maximal patterns are redefined for sequences; **FP-growth → FreeSpan**, with **projected databases** as the analogue of **conditional pattern bases**.
- **The candidate-explosion argument is the same one that opened the colossal-pattern discussion in Chapter 8 §3** — the property enabling level-wise mining is the property forbidding long patterns.
- **The `f_list` in support-descending order** is the **FP-tree header-table ordering** of Chapter 6, and **projection under an imposed item order** is what made **convertible constraints** pushable in **Chapter 8 §2.9**.
- **The lecture closing Chapter 8 announced this chapter verbatim**: *"when we talk about frequent patterns we don't take the order between items into consideration; for this reason we'll also analyse frequent SEQUENCES."*
- **Web-log and click-through data** appeared as bipartite/graph data in **Chapter 10 §3**; here the same logs are mined as **sequences**.

---

*File auto-generated by merging `12-SequentialPatternMining.pdf` (professor's slides, 36 pages) and `12 - SequentialPatternMining Sbobine.pdf` (lecture notes, 35 pages — lesson L09 of 31/03/2026 plus the second part of 14/04). Tables and formulas that the deck stored as images (the customer-sequence and transformation tables, the AprioriAll/AprioriSome runs, the `next()` step table, the frequent item matrix and the annotation tables) have been reconstructed in standard form and are consistent with both the answers printed on the slides and the lecture's step-by-step commentary. Informal asides in the notes have been omitted; their technical content is retained. For questions about this chapter, refer only to this file.*
