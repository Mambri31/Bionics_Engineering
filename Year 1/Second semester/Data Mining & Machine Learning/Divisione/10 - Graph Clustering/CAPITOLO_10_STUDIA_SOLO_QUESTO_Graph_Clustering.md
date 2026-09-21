# Chapter 10 — Clustering Graphs and Network Data

> **Course:** Data Mining and Machine Learning / Bioinspired Computational Methods — Biological Data Mining
> **Instructor:** Prof. Francesco Marcelloni (University of Pisa)
> **Sources merged into this file:** professor's slides (`10-GraphClustering.pdf`, 33 pages — Chapter 11 of Han–Kamber–Pei) + lecture notes (`10 - Graph Clustering sbobine.pdf`, 22 pages)
>
> **Instructions for Claude:** this is the single reference file for Chapter 10. When the user asks study questions about this chapter, use **only** this file as context (no need to re-read the original PDFs unless a detail is missing). The file merges the official slide content with the lecture's discursive explanations and worked examples, keeping both levels of detail.

---

## Chapter outline
1. What is a network? Why networks matter
2. Current problems in network analysis (ranking, cascades, link prediction, content, communities)
3. Applications of graph/network clustering
4. Basics on networks and graphs
5. Similarity measure (I): geodesic distance
6. Similarity measure (II): similarity in a social network
7. SimRank: similarity based on structural context and random walk
8. Graph clustering: cuts, sparsest cut, modularity
9. Challenges of finding good cuts; the two families of methods
10. SCAN: density-based clustering of networks

---

## 1. What is a network?

> **Definition**: a **network is a collection of entities that are interconnected with links.**

| Network type | Entities | Links |
|---|---|---|
| **Social networks** | people | friendships |
| **Communication networks** | people | e-mail exchange |
| **Communication networks** | Internet nodes | communication between nodes |
| **Biological networks** | **proteins** | **interactions** |
| **Biological networks** | **metabolites, enzymes** | **chemical reactions** |
| **Information/media networks** | web pages | hyperlinks |
| **Information/media networks** | Twitter users | follows / conversations |

> **Why this matters for this course** (lecture): a lot of networks shape our society, especially social networks, where **links represent relationships, not merely connections** — so it is important to study them. In the biological setting the same formalism describes **protein-interaction networks** and **metabolic networks**. *(The slides cite a concrete biological study: phenotypic subgrouping and multi-omics analyses revealing reduced DBI protein levels in autism spectrum disorder with severe language impairment, PLoS ONE 14(3):e0214198, 2019.)*

### Why are networks important?

> **We cannot truly understand a complex system unless we understand the underlying network. Everything is connected; studying individual entities gives only a PARTIAL view of a system.**

**Two main themes:**
1. **What are the structural properties of the network?**
2. **How do processes happen in the network?**

> **A concrete motivation from the lecture**: if there are **small clusters of people who strongly interact with each other**, this can be a **security threat** — they could be attackers. Finding such cohesive subgroups is exactly the clustering problem of this chapter.

### Graphs and networks

- **In mathematics, networks are called GRAPHS, entities are NODES, links are EDGES.**
- **Graph theory starts in the 18th century, with Leonhard Euler.**
- Graphs have long been used to model existing networks (highway networks, social networks) — but **those networks were usually small**, so **visual inspection could reveal a lot of information**.

**Networks now:**
- **More and larger networks appear**, as products of technology (Internet, Web, Facebook, Twitter), as a result of our ability to **collect more, better and more complex data** (e.g. **gene regulatory networks**), and because **users are willing to contribute data** (making their relationships public online).
- **Networks of thousands, millions or billions of nodes** → **impossible to process visually**, **problems become harder**, **processes are more complex**.

## 2. Current problems in network analysis

**(a) Ranking of nodes on the web.** *Is my home page as important as the Google page?* **We need algorithms to compute the importance of nodes in a graph** — for instance the **PageRank** algorithm in Google. **Theoretically, it is impossible to develop a web search engine without understanding the web graph.**

> **The lecture's gloss**: PageRank ranks pages by exploiting **the links between pages** — if there are **many links coming to a page, it must be important**. Two side remarks made in class: today there is **AI support in search, so the energy consumption of a single query has gone up a lot**, and the first results returned are usually **sponsored pages** (commercial interest coming into play).

**(b) Information / virus cascades.** **How do viruses spread between individuals, and how can we stop them? How does information propagate? What items become viral? Who are the influencers and trend-setters?** We need **models and algorithms** to answer these.

**(c) Link prediction.** *Given a snapshot of a social network at time $t$, accurately predict the edges that will be added between $t$ and a future time $t'$.* **Applications**: **accelerate the growth of a social network** (Facebook, LinkedIn, Twitter) that would otherwise take longer to form; **identify suspect relationships**.

**(d) Network content.** Users generate content, and **mining the content together with the network is useful**: *Do friends post similar content? Can we understand a user's interests by looking at those of their friends? **Social recommendations**: can we predict a movie rating using the social network?*

**(e) Social media.** Social media have **supplanted traditional media sources**; information is generated and disseminated **mostly online by users** (e.g. **the assassination of Bin Laden appeared first on Twitter**). **Twitter has become a global "sensor"**. Interesting problems: **automatic event detection**, **earthquake news propagation**, **crisis detection and management**, **sentiment mining**, **tracking the evolution of events socially, geographically and over time**.

> **The critical remarks made in class** — worth keeping, they are the "why this is hard" part:
> - News propagated through social media carries a **bias**: **people from the same context polarize information in a specific direction**, and **people in a small community circulate news according to their own preferences**. To get an unbiased view, **information should be acquired from different perspectives**.
> - **Fast propagation makes it hard to check validity → fake news spreads very fast.** **AI is used both to CREATE and to DETECT fake news.**
> - **Sentiment mining** used to require interviews and manual analysis; today it can be done directly on social networks. **But there is a bias: some people express their opinion more loudly than others — so the analysis should be done on the USERS, not on the tweets per se.**

**(f) Clustering and finding communities.**

> **Definition of community** [Wasserman & Faust '97]: *"**Cohesive subgroups are subsets of actors among whom there are relatively strong, direct, intense, frequent, or positive ties.**"*

The canonical illustration is the **Karate club example** [W. Zachary, 1970].

> **The lecture's operational reading**: **if by reaching one user it is very likely that we reach all the others, we are in the presence of a community.**

**(g) Community evolution — HOMOPHILY.** *"Birds of a feather flock together."* Caused by **two related social forces** [Friedkin '98, Lazarsfeld '54]:

- **Social influence**: **people become similar to those they interact with**;
- **Selection**: **people seek out similar people to interact with**.

> **⚠️ Both processes contribute to homophily, but they pull in OPPOSITE directions**:
> - **Social influence leads to community-wide HOMOGENEITY**;
> - **Selection leads to FRAGMENTATION of the community.**

**Applications in online marketing**: **viral marketing relies on social influence affecting behaviour**; **recommender systems predict behaviour based on similarity**.

## 3. Applications of clustering graphs and network data

**Bi-partite graphs** — e.g. **customers and products**, **authors and conferences**:
- **cluster customers buying similar products** (build profiles of customers with similar purchasing habits);
- **identify customers OUT of the clusters** (the outliers).

**Web search engines — click-through graphs and web graphs:**
- **Click-through information**: **an edge links a query to a web page if a user clicks that page when asking the query.** Valuable information is obtained by **cluster analysis on the query–web page bipartite graph**.
- **Web graph**: **each web page is a vertex, each hyperlink is an edge** pointing from a source page to a destination page.

**Social networks, friendship/co-author graphs:**
- vertices are **individuals or organizations**, links are **interdependencies** (friendship, common interests, collaborative activities).
- *Ex.*: the **customers of a company** form a social network — each customer a vertex, an edge if they know each other. **Customers within a cluster may influence one another regarding purchase decisions** (so the company can propagate marketing by reaching just one customer).
- *Ex.*: the **authors of scientific publications** form a social network. **In general this is a WEIGHTED graph**, because an edge between two authors can carry a **weight representing the strength of the collaboration** — e.g. **how many publications they co-authored**.

## 4. Basics on a network

| Concept | Network language | Graph language |
|---|---|---|
| Objects | **nodes**, **vertices** $N$ | vertex |
| Interactions | **links**, **edges** $E$ | edge |
| System | **network** $G(N,E)$ | **graph** |
| Typical use | **real systems**: Web, social network, metabolic network | **mathematical representation**: web graph, social graph, knowledge graph |

> **The concept of network and graph is the same — only the context changes.**

**How to build a graph: what are the nodes? What are the edges?**

> **⚠️ The choice of the proper network representation of a given domain/problem DETERMINES OUR ABILITY to use networks successfully:**
> - in some cases there is a **unique, unambiguous representation**;
> - in other cases **the representation is by no means unique**.
>
> **The way you assign the links will determine the nature of the question you can study.** *(In reality we only have a model of our problem, so we must decide which are the nodes and which are the links before anything else.)*

### The two strategies for clustering graphs

| Strategy | How it works | Instances |
|---|---|---|
| **(1) Standard clustering algorithms + a graph-specific similarity measure** | Objects (vertices) are described by their links; **compute similarity between every pair of vertices → build a (dis)similarity matrix → run an existing clustering algorithm on it** | **geodesic distance**, **distance based on random walk (SimRank)** |
| **(2) Graph clustering methods** | **New algorithms applied directly to the graph**, exploiting its peculiarities | **minimum cuts: FastModularity** (Clauset, Newman & Moore 2004); **density-based: SCAN** (Xu et al., KDD'2007) |

> **Note on strategy (1)**: since most clustering algorithms work on **dissimilarities**, and these measures produce **similarities in $[0,1]$**, we convert with **dissimilarity $= 1 - $ similarity**.

## 5. Similarity measure (I): geodesic distance

**Distance between two vertices in a graph = the SHORTEST PATH between them.**

- **Geodesic distance $(A,B)$**: the **length (number of edges) of the shortest path** between $A$ and $B$. **If they are not connected, it is defined as infinite.**
- **Eccentricity of $v$, $\text{eccen}(v)$**: the **largest geodesic distance between $v$ and any other vertex** $u \in V - \{v\}$.
  *It measures how much the vertex is at the centre of the graph.*
- **Radius of $G$**: the **MINIMUM eccentricity of all vertices** — the distance between the **"most central point"** and the **"farthest border"**:
  $$r = \min_{v \in V} \text{eccen}(v)$$
- **Diameter of $G$**: the **MAXIMUM eccentricity of all vertices** — the largest distance between any pair of vertices:
  $$d = \max_{v \in V} \text{eccen}(v)$$
- **A PERIPHERAL VERTEX is a vertex that achieves the diameter.**

**The worked example from the slides/lecture:**
$$\text{eccen}(a) = \text{eccen}(b) = 2, \qquad \text{eccen}(c) = \text{eccen}(d) = \text{eccen}(e) = 3$$
$$\Rightarrow \text{radius}(g) = 2, \qquad \text{diameter}(g) = 3, \qquad \text{peripheral vertices} = c, d, e$$

### ⚠️ How well does geodesic distance measure similarity in a network?

Consider two customers, **Ada** and **Bob**, in a customer social network. **The geodesic distance is the shortest path along which a message can be passed from Ada to Bob and vice versa.** We can certainly build a full dissimilarity matrix this way and cluster it.

> **But is this information useful?** **Typically the company is NOT interested in how a message is passed from Ada to Bob.** What we normally care about is **how many people can reach Ada and Bob** — i.e. **how the customers are connected to each other**, not the length of a routing path. **We need to define what similarity MEANS in a social network.**

## 6. Similarity measure (II): similarity in a social network

**Two different meanings — both based on the NEIGHBOURS of the vertices:**

**(a) Structural-context-based similarity**
> **Two customers are similar to one another if they have SIMILAR NEIGHBOURS in the social network.**
> **Intuitive justification**: **two people receiving recommendations from a good number of common friends often make similar decisions.**

**(b) Similarity based on random walk**
> The company sends promotional information to both Ada and Bob. **They may randomly forward it to their friends (neighbours).** The **closeness between Ada and Bob is then measured by the LIKELIHOOD that other customers simultaneously receive the promotional information originally sent to Ada and Bob.**
> *If Ada and Bob are able to reach the same people, they have the same neighbourhood.*

> **⚠️ Key point stressed twice in the lecture: these two views are ESSENTIALLY THE SAME MEASURE, seen from different perspectives.** This is not a hand-wave — §7 ends with the formal theorem proving the two definitions coincide.

## 7. SimRank: similarity based on random walk and structural context

### 7.1 The structural-context definition

In a **directed graph** $G = (V, E)$:
- **individual IN-neighbourhood** of $v$: $\;I(v) = \{u \mid (u,v) \in E\}$ — the vertices whose edges **enter** $v$ *(this is the more relevant one)*;
- **individual OUT-neighbourhood** of $v$: $\;O(v) = \{w \mid (v,w) \in E\}$ — the vertices reached by edges **leaving** $v$.

**SimRank similarity:**
$$s(u,v) \;=\; \frac{C}{|I(u)|\,|I(v)|} \sum_{x \in I(u)} \sum_{y \in I(v)} s(x,y)$$

where **$C$ is a constant between 0 and 1**, and:
- **$s(u,v) = 1$ if $u = v$**;
- **if a vertex has no neighbour at all, we define $s(u,v) = 0$.**

> **Read the formula in words** (lecture): the denominator is the **product of the cardinalities of the two in-neighbourhoods**; the numerator sums the similarity **over every pair $(x,y)$ with $x$ an in-neighbour of $u$ and $y$ an in-neighbour of $v$**.
>
> **⚠️ NB: two vertices are similar if they RECEIVE THE INPUT FROM THE SAME VERTICES.**
>
> **The definition is RECURSIVE** — similarity is defined in terms of similarity itself — so **it must be computed ITERATIVELY, and it needs an initial condition.**

### 7.2 Computing SimRank iteratively

- Let $n$ be the number of nodes. **For each iteration $i$ we keep $n^2$ entries $s_i(*,*)$**, where $s_i(u,v)$ is the score between $u$ and $v$ at iteration $i$.
- **Start from $s_0(*,*)$**, which is a **lower bound** on the actual SimRank score:
  $$s_0(u,v) = \begin{cases} 1 & \text{if } u = v \\ 0 & \text{otherwise}\end{cases}$$
- **Iterate**:
  $$s_{i+1}(u,v) \;=\; \frac{C}{|I(u)|\,|I(v)|} \sum_{x \in I(u)} \sum_{y \in I(v)} s_i(x,y) \qquad (\text{and } s_{i+1}(u,v)=1 \text{ if } u=v)$$
  until a **fixed point** is reached.

> **The values $s_i(*,*)$ are NON-DECREASING as $i$ increases** — because we started from a **lower bound**, so the iteration can only push the scores up towards the true value. **Normally 4–5 iterations suffice to reach a stable value.**

**Complexity:**
$$O(K\,n^2\,d^2)$$
where **$d^2$ is the average of $|I(u)|\,|I(v)|$**, **$n$ is the number of vertices** (note the **quadratic** dependence — this is the real bottleneck) and **$K$ is the number of iterations, typically 5**.

**What we do with the result**: after the iterations we have a **similarity matrix**, one entry per pair of vertices. **Convert it into a dissimilarity matrix ($1 - s$) and run any clustering algorithm on it.**

### 7.3 The random-walk formulation

Assume a **strongly connected graph** (a path exists between every two nodes).

**Expected distance from $u$ to $v$:**
$$d(u,v) \;=\; \sum_{t:\, u \rightsquigarrow v} P[t]\;\ell(t)$$

where **the sum is over all tours $t$ that start at $u$, end at $v$, and DO NOT TOUCH $v$ except at the end.**

- For a tour $t = \langle w_1, \dots, w_k\rangle$, its **length is $\ell(t) = k-1$** ($k$ = number of vertices it touches).
- The **probability of travelling $t$** is
  $$P[t] \;=\; \prod_{i=1}^{k-1} \frac{1}{|O(w_i)|}$$
  i.e. the product, over the vertices of the tour, of the probability of picking **one specific out-edge** among the $|O(w_i)|$ available.

> **The case $u = v$, for which $d(u,v)=0$, is a SPECIAL CASE of the formula**: only one tour is in the summation and it has length 0.
>
> **Interpretation**: **the expected distance from $u$ to $v$ is exactly the expected number of steps a random surfer — who at each step follows a random out-edge — would take before FIRST reaching $v$, starting from $u$.**

### 7.4 Expected Meeting Distance (EMD)

> **Definition**: the **expected meeting distance $m(u,v)$** between $u$ and $v$ is the **expected number of steps required before two surfers — one starting at $u$, the other at $v$ — would MEET, if they walked randomly IN LOCK-STEP.**
>
> **The EMD is SYMMETRIC by definition.**

> **What it means for the application** (lecture): we want to know whether two users of a social network **have similar friends**. **If the meeting distance is low, the two surfers reach the same friend in very few steps** — so the users are similar.

**The three examples from the slides:**
1. **A directed cycle**: **EMD $= \infty$** — *the two surfers always travel the same circle one behind the other, so they never meet.*
2. **A graph where** $m(u,v) = m(u,w) = \infty$ **but** $m(v,w) = 1$ — *$v$ and $w$ need to traverse only one edge before meeting.*
3. **A configuration with EMD $= 3$** — *whatever the position of $u$ and $v$, there are always 3 edges before they meet.*

### 7.5 Formalising EMD: the derived graph $G^2$

To define EMD formally we use the **derived graph $G^2$ of node-pairs**:

- **Each node $(u,v)$ of $V^2$ represents the PRESENT STATE of a pair of surfers** in $V$ — one at $u$, one at $v$.
- **An edge from $(u,v)$ to $(c,d)$ in $G^2$ says that in the original graph $G$, one surfer can move from $u$ to $c$ while the other moves from $v$ to $d$** — i.e. **$(a,b)$ points to $(c,d)$ iff in $G$, $a$ points to $c$ and $b$ points to $d$**.
- **A tour in $G^2$ of length $n$ represents a PAIR of tours in $G$ also of length $n$.**

**The worked example** (from the slides, expanded in the lecture): the web pages of **two professors ProfA and ProfB**, their **students StudentA and StudentB**, and the **home page of their university Univ** (the main page links to the professors, who link to their students).

> **How to trace it** (lecture): start with **two surfers at Univ**. At each step both surfers randomly choose one outgoing edge. After the first move, one surfer may reach **ProfA** while the other reaches **ProfB** — that pair *is* a node of $G^2$. From there the first may go to **StudentA** and the second to **StudentB**; next, the first could return to **Univ** while the second goes back to **ProfB**; and so on. **This generates two parallel random walks**, and $G^2$ **represents all the possible movements of this pair of surfers.**
>
> **Why the construction is useful**: we want to compare nodes **based on their neighbourhoods and structure**, so **we analyse PAIRS of paths, not individual nodes**.

**Formal EMD:**
$$m(u,v) \;=\; \sum_{t:\,(u,v) \rightsquigarrow (x,x)} P[t]\;\ell(t)$$

where the sum is over **all tours $t$ starting at $(u,v)$ that touch a SINGLETON node $(x,x) \in V^2$ at the end and only at the end** — singleton nodes represent the states where **both surfers are at the same node** (before the last step the surfers are always at different nodes; at the last step they meet).

**⚠️ Two problems with this definition:**
1. **$G^2$ may not be strongly connected even if $G$ is.** In that case **there is no tour $t$ in the summation and $m(u,v) = \infty$** — no meeting point can be found.
2. **The definition misbehaves when, from a given $(u,v)$, some tours lead to singleton nodes while others lead back to $(u,v)$** without meeting. *Are these two nodes similar? It would depend on the path taken* — an inconsistency that prevents a clean, well-behaved distance.

### 7.6 The fix: expected-$f$ meeting distance

> **Solution: map all distances to a FINITE interval.** Instead of computing the expected length $\ell(t)$ of a tour, compute the **expected $f(\ell(t))$**, for a **non-negative, monotonic function $f$ that is BOUNDED on the domain $[0, \infty)$**.

$$s'(u,v) \;=\; \sum_{t:\,(u,v) \rightsquigarrow (x,x)} P[t]\; f(\ell(t))$$

**Properties obtained:**
- **$s'(a,b) = 0$** ⇒ **no tour from $(a,b)$ to any singleton node**;
- **$s'(a,b) = 1$** ⇔ **$a = b$**;
- **$s'(a,b) \in [0,1]$ for all $a,b$.**

> **⚠️ Note the INVERSION of the reading**: distances of **0 map to 1** and distances of **$\infty$ map to 0**, so **CLOSE nodes get a HIGH score** — the quantity now behaves as a **similarity**, matching our intuition. *(To sum up: close nodes → surfers meet quickly → $\ell(t)$ small → high similarity. Far nodes → they meet late or never → low or zero similarity.)* **The trick exists precisely to avoid infinite distances.**

**Examples with $C = 0.8$** (same three graphs as before): $s'(a,b) = 0$; $s'(u,v) = s'(u,w) = 0$ while $s'(v,w) = 0.8$; and $s'(a,b) = 0.47$.

### 7.7 The equivalence theorem

> **It has been proved that the SimRank score with parameter $C$ between two nodes is exactly their expected-$f$ meeting distance travelling BACK-EDGES, for**
> $$f(z) = C^{\,z}$$
> **In other words, $s(u,v) = s'(u,v)$ for any two vertices $u$ and $v$.**
>
> **That is: SimRank is based on BOTH structural context AND random walk.** We start from two different intuitions and obtain **the same values**. Either way, we end up with a **dissimilarity matrix between nodes** that implements our intuition about social networks.

## 8. Graph clustering: cuts, sparsest cut, modularity

> **How should we conduct clustering in a graph?** **Intuitively, we should cut the graph into pieces — each piece a cluster — such that the vertices WITHIN a cluster are well connected and the vertices in DIFFERENT clusters are connected in a much weaker way.**

**Definitions.** Let $G = (V,E)$ be a directed graph.
- **A cut $C(S,T)$ is a partitioning of $V$**: $V = S \cup T$ and $S \cap T = \emptyset$ (**two clusters, no overlap**).
- **The CUT SET of a cut** is the set of edges $\{(u,v) \in E \mid u \in S,\; v \in T\}$ — the edges connecting the two sides.
- **Size of the cut**: the **number of edges in the cut set**. **If the edges are weighted, the VALUE of the cut is the sum of the weights.**

### 8.1 Why the minimum cut is NOT good enough

**Minimum cut**: a cut whose **size is not greater than that of any other cut**. **Polynomial-time algorithms exist** to compute minimum cuts (**Edmonds–Karp algorithm**).

**The worked example**: a graph with two densely interconnected subgraphs joined by only two edges, plus a vertex $l$ hanging off one side by a single edge.

- **The minimum cut simply separates the vertex $l$ from everything else** — only **one** edge to cut, versus two for the "natural" split. **We are only isolating an OUTLIER, not finding clusters.**
- The cut $C_2 = (\{a,b,c,d,e,f,l\},\;\{g,h,i,j,k\})$ **leads to a much better clustering** than $C_1$. The edges in the cut set of $C_2$ — namely $(d,h)$ and $(e,k)$ — are those connecting the two **"natural clusters"**: **for those edges, MOST of the edges connecting $d$, $h$, $e$ and $k$ belong to ONE cluster.**

> **⚠️ This is the key observation that motivates sparsity**: an absolute count of cut edges is the wrong criterion, because **it is systematically minimised by shaving off tiny pieces (outliers)**. We need the **relation between what we are cutting and the connections inside the clusters**.

### 8.2 Sparsity and the sparsest cut

> **Intuition**: choose a cut where, **for each vertex $u$ involved in an edge of the cut set, MOST of the edges connecting to $u$ belong to one cluster.**

**The sparsity of a cut $C = (S,T)$:**
$$\Phi \;=\; \frac{\big|\{(u,v) \in E \mid u \in S,\; v \in T\}\big|}{\min\{|S|,\;|T|\}}$$

(the denominator counts **vertices**, not edges).

- **A cut is SPARSEST if its sparsity is not greater than that of any other cut.**
- **This favours solutions that are both SPARSE (few edges crossing the cut) and BALANCED (close to a bisection)** — the $\min\{|S|,|T|\}$ denominator is exactly what penalises shaving off a single vertex.
- **⚠️ The problem is NP-HARD**; the best known algorithm is an **$O(\sqrt{\log n})$ approximation** due to **Arora, Rao & Vazirani (2009)**.
- In the example, **$C_2 = (\{a,b,c,d,e,f,l\}, \{g,h,i,j,k\})$ is the sparsest cut.**

### 8.3 Modularity — assessing the quality of a $k$-clustering

$$Q \;=\; \sum_{i=1}^{k} \left[ \frac{l_i}{|E|} \;-\; \left(\frac{d_i}{2|E|}\right)^{\!2} \right]$$

where:
- **$l_i$ = number of edges between vertices in the $i$-th cluster**;
- **$d_i$ = the sum of the DEGREES of the vertices in the $i$-th cluster** (degree of $u$ = number of edges connecting to $u$);
- **$|E|$ = total number of edges.**

The two terms are, respectively, **the probability that an edge is in cluster $i$** and **the probability that a RANDOM edge would fall into cluster $i$**.

> **In words**: **the modularity of a clustering is the DIFFERENCE between the fraction of all edges that fall into individual clusters and the fraction that would do so if the graph vertices were RANDOMLY connected.** **The optimal clustering maximises the modularity.**
>
> **High modularity means the connection we have inside the cluster is HIGHER than what we should expect under the random model.** *(This is the criterion optimised by **FastModularity**, Clauset, Newman & Moore 2004.)* **But this approach is quite expensive.**

## 9. Challenges of finding good cuts, and the two families of methods

**Four challenges:**
1. **High computational cost** — many graph-cut problems are computationally expensive; **the sparsest cut problem is NP-hard**. **We need a trade-off between efficiency/scalability and quality.**
2. **Sophisticated graphs** — they may involve **weights and/or cycles**.
3. **High dimensionality** — a graph can have very many vertices, and **in a similarity matrix a vertex is represented as a vector (a row) whose dimensionality is the NUMBER OF VERTICES in the graph**.
4. **Sparsity** — **a large graph is often sparse** (each vertex connects on average to only a few others), so **the similarity matrix derived from it is also sparse**, which **creates problems for the clustering algorithms**.

**The two families (restated):**
- **Clustering methods for high-dimensional data**: **extract a similarity matrix** from the graph using a similarity measure, then **apply a clustering algorithm for high-dimensional data** *(this is where Chapter 9 plugs in)*.
- **Clustering methods designed specifically for graphs**: **exploit the peculiarities of the graph** to perform the clustering directly.

## 10. SCAN: density-based clustering of networks

*(Xu et al., KDD'2007.)*

**The questions SCAN answers**: *How many clusters? What size should they be? What is the best partitioning? **Should some points be segregated?***

**Application**: *given simply the information of who associates with whom, can one identify **clusters of individuals with common interests or special relationships** (families, cliques, terrorist cells)?*

### 10.1 Cliques, hubs and outliers

> **⚠️ SCAN finds THREE kinds of vertices — this is its distinguishing feature:**
> - **CLIQUE members**: individuals in a tight social group **know many of the same people, regardless of the size of the group** — strong interconnection with each other.
> - **HUBS**: individuals who **know many people in different groups but belong to NO single group**. **Politicians**, for example, **bridge multiple groups**.
> - **OUTLIERS**: individuals who **reside at the margins of society**. **Hermits**, for example, **know few people and belong to no group** — few connections.

**The neighbourhood of a vertex**: $\Gamma(v)$ is the **immediate neighbourhood** of $v$ — *the set of people that an individual knows*:
$$\Gamma(v) = \{w \in V \mid (v,w) \in E\} \cup \{v\}$$

> **⚠️ Note that $\Gamma(v)$ INCLUDES $v$ ITSELF** — when we consider a vertex, $\Gamma$ is obtained by taking all the vertices connected to it **plus the vertex itself**. This matters in every numeric computation below.

### 10.2 Structural similarity

$$\sigma(v,w) \;=\; \frac{|\Gamma(v) \cap \Gamma(w)|}{\sqrt{|\Gamma(v)|\;|\Gamma(w)|}}$$

**At the numerator: the intersection of the two immediate neighbourhoods. At the denominator: the square root of the product of their cardinalities.**

> **Why it captures exactly what we want**: **$\sigma$ is LARGE for members of a clique and SMALL for hubs and outliers.**
> - If two vertices are **highly connected**, the **numerator is high** and comparable with $|\Gamma(v)|$ and $|\Gamma(w)|$ → **$\sigma$ close to 1**.
> - In the case of a **hub**, the **numerator is not so high**, but **the cardinalities of both neighbourhoods are high**, so **the denominator exceeds the numerator** → **$\sigma$ small**.
>
> **The example from the slides**: **vertex 6 is a hub** — it is **not strongly connected to the vertices of a single cluster** but **connected to vertices of different clusters**, so its **intersection with each neighbour is relatively low while its own neighbourhood is large**.

### 10.3 Structural connectivity — the DBSCAN-style machinery

SCAN uses a **similarity threshold $\varepsilon$** to define cluster membership.

- **$\varepsilon$-neighbourhood of $v$:**
  $$N_\varepsilon(v) = \{w \in \Gamma(v) \mid \sigma(v,w) \ge \varepsilon\}$$
- **CORE vertex** — a vertex **inside** a cluster. $v$ is a core vertex **iff**
  $$\text{CORE}_{\varepsilon,\mu}(v) \iff |N_\varepsilon(v)| \ge \mu$$
  where **$\mu$ is a POPULARITY threshold**.

> **The mapping onto DBSCAN** (made explicit in the lecture): **SCAN grows clusters from core vertices exactly like DBSCAN.** *The role played by the radius $\varepsilon$ in DBSCAN is played here by the structural-similarity threshold $\varepsilon$; the role of MinPts is played by $\mu$.* **If a vertex $v$ is in the $\varepsilon$-neighbourhood of a core $u$, then $v$ is assigned to the same cluster as $u$; the growing process continues until no cluster can be further grown.**

- **Directly structure-reachable**: $w$ can be **directly reached** from a core $v$ if
  $$\text{DirREACH}_{\varepsilon,\mu}(v,w) \iff \text{CORE}_{\varepsilon,\mu}(v) \;\wedge\; w \in N_\varepsilon(v)$$
- **Structure-reachable** — the **transitive closure** of direct structure reachability: $v$ can be reached from a core $u$ if there exist vertices $w_1, \dots, w_n$ such that $w_1$ can be reached from $u$, $w_i$ from $w_{i-1}$ ($1 < i \le n$), and $v$ from $w_n$.
- **Structure-connected**: two vertices $v$ and $w$ — **which may or may not be cores** — are **connected** if **there exists a core $u$ such that both $v$ and $w$ can be reached from $u$**:
  $$\text{CONNECT}_{\varepsilon,\mu}(v,w) \iff \exists u \in V:\; \text{REACH}_{\varepsilon,\mu}(u,v) \;\wedge\; \text{REACH}_{\varepsilon,\mu}(u,w)$$

**Structure-connected cluster $C$** — defined by two properties:
- **Connectivity**: $\forall v, w \in C: \text{CONNECT}_{\varepsilon,\mu}(v,w)$;
- **Maximality**: $\forall v, w \in V:\; v \in C \wedge \text{REACH}_{\varepsilon,\mu}(v,w) \Rightarrow w \in C$.

**And the two special roles:**
- **HUBS**: **do not belong to any cluster**, but **bridge to MANY clusters**;
- **OUTLIERS**: **do not belong to any cluster** and **connect to FEWER clusters**.

> **⚠️ The output of SCAN is not just clusters — it is clusters PLUS hubs PLUS outliers.** When a vertex is classified as a **non-member**, we **apply the structural-similarity definition to its neighbours to decide whether it is a hub or an outlier**.

### 10.4 The worked example ($\varepsilon = 0.7$, $\mu = 2$)

The reference graph has vertices $0..13$; **vertex 6 turns out to be a HUB and vertex 13 an OUTLIER**.

**Step 1 — try vertex 13.**
- $\Gamma(13) = \{13, 9\} \Rightarrow |\Gamma(13)| = 2$ (the vertex itself plus its only neighbour).
- $\Gamma(9) = \{8, 12, 10, 13, 9\} \Rightarrow |\Gamma(9)| = 5$.
- $\Gamma(13) \cap \Gamma(9) = \{9, 13\} \Rightarrow$ cardinality $2$.
$$\sigma(13,9) = \frac{2}{\sqrt{2 \times 5}} = \frac{2}{\sqrt{10}} = 0.63$$
**$0.63 < \varepsilon = 0.7$**, and 9 is 13's only neighbour ⇒ **no neighbour clears the threshold** ⇒ **$|N_\varepsilon(13)| < \mu$** ⇒ **13 is NOT a core vertex, so no cluster can start from it.**

**Step 2 — try vertex 8.**
- $\Gamma(8) = \{9, 12, 7, 8\} \Rightarrow |\Gamma(8)| = 4$.
- For each neighbour: $|\Gamma(9)| = 5$, $|\Gamma(12)| = 7$, $|\Gamma(7)| = 5$.
- The three structural similarities come out as **$\sigma(8,9) = 0.67$, $\sigma(8,12) = 0.82$, $\sigma(8,7) = 0.67$**.
- **Only vertex 12 exceeds $\varepsilon = 0.7$** ⇒ **$|N_\varepsilon(8)| = 1 < \mu = 2$** ⇒ **8 is NOT a core vertex either.**

> *(Counting convention: the lecture tallies against $\mu$ only the **neighbours** whose similarity clears $\varepsilon$. Since $\sigma(v,v)=1$ always holds and $v \in \Gamma(v)$, a formulation that also counts $v$ itself would give one more — follow the professor's counting in the exam.)*

**Step 3 — try vertex 12.** **It IS a core vertex**, so **a cluster starts from it** and is grown by structure-reachability, adding points **until we arrive at vertex 6**.

**Step 4 — classify vertex 6.** It is **not a core object and not directly reachable from 12**, so it is a **non-member**. Evaluating its structural similarity with all adjacent vertices shows **comparable structural connectivity towards BOTH clusters** ⇒ **vertex 6 is a HUB.** *(Its computed similarities in the slide animation are around 0.51 and 0.68 — moderate values towards two different groups, the numerical signature of a hub.)*

**Vertex 13, by contrast, remains attached to nothing** ⇒ **OUTLIER.**

### 10.5 Running time

$$\textbf{Running time} = O(|E|) \qquad \Rightarrow \qquad \textbf{for sparse networks } O(|V|)$$

> **The verdict from the lecture**: a **big advantage is that the computational time is quite stable as the number of vertices increases**. **But — exactly as in DBSCAN — the problem is the definition of the TWO PARAMETERS ($\varepsilon$ and $\mu$): we must find the correct combination for the specific application domain.**

---

## Key points / potential exam pitfalls

### Networks, graphs, representation
- **Network = graph; node = vertex; link = edge.** Only the **context** differs (real system vs. mathematical representation).
- **⚠️ The choice of representation determines which questions you can study.** "What are the nodes / what are the edges" is a modelling decision, not a technicality, and it is **not always unique**.
- **Homophily has TWO causes with OPPOSITE effects**: **social influence ⇒ community-wide homogeneity**; **selection ⇒ fragmentation of the community**. Getting these two swapped is the classic slip.
- **Co-author and collaboration networks are WEIGHTED** (e.g. number of joint publications).

### Geodesic distance
- **eccentricity = the LARGEST geodesic distance from a vertex**; **radius = MIN eccentricity**; **diameter = MAX eccentricity**; **peripheral vertex = one that achieves the diameter**. Memorise the worked numbers: eccen $= 2,2,3,3,3$ ⇒ radius 2, diameter 3, peripheral $c,d,e$.
- **Disconnected vertices ⇒ geodesic distance $= \infty$.**
- **⚠️ The point of the Ada/Bob discussion**: geodesic distance is **perfectly computable but answers the wrong question** — "how a message travels" rather than "how similar the two customers' social positions are". Be ready to state *why* a shortest path is a poor similarity in a social network.

### SimRank
- **SimRank is recursive**: $s(u,v) = \frac{C}{|I(u)||I(v)|}\sum_{x \in I(u)}\sum_{y\in I(v)} s(x,y)$, with **$s(u,v)=1$ if $u=v$** and **$s(u,v)=0$ if a vertex has no neighbours**.
- **⚠️ It uses the IN-neighbourhood $I(\cdot)$**: *two vertices are similar if they RECEIVE input from the same vertices.* Do not write it with $O(\cdot)$.
- **The iteration starts from a LOWER bound ($s_0 = $ identity), hence the scores are NON-DECREASING in $i$.** Typically **$K \approx 5$ iterations**.
- **Complexity $O(K n^2 d^2)$** — **quadratic in the number of vertices**, which is the practical limitation.
- **Similarity → dissimilarity via $1 - s$** before feeding a standard clustering algorithm.
- **Tour length is $\ell(t) = k-1$** for $k$ touched vertices, and **$P[t] = \prod 1/|O(w_i)|$** uses the **OUT-degree** (a random surfer picks an out-edge). **Note the asymmetry with the structural definition, which uses in-neighbours — the equivalence theorem reconciles them by travelling BACK-edges.**
- **EMD is symmetric by definition**; the **expected distance $d(u,v)$ is NOT**.
- **⚠️ $G^2$: a node is a PAIR of positions, an edge is a SIMULTANEOUS move of both surfers, and singleton nodes $(x,x)$ are the meeting states.** $G^2$ **may fail to be strongly connected even when $G$ is** ⇒ $m(u,v)=\infty$ — this is precisely why the plain EMD is unusable.
- **The expected-$f$ fix INVERTS the scale**: $f$ non-negative, monotonic, **bounded on $[0,\infty)$**; distance $0 \mapsto 1$, distance $\infty \mapsto 0$, so **the result is a SIMILARITY in $[0,1]$**.
- **The equivalence theorem: SimRank with parameter $C$ = expected-$f$ meeting distance over back-edges with $f(z) = C^z$.** Hence **structural context and random walk give the same numbers** — a favourite exam question.

### Cuts, sparsity, modularity
- **⚠️ The MINIMUM cut is a trap**: it tends to **isolate outliers** (one edge is cheaper to cut than two), which is why the example's $C_1$ merely detaches vertex $l$. **Minimum cut is polynomial (Edmonds–Karp); sparsest cut is NP-HARD** ($O(\sqrt{\log n})$ approximation, Arora–Rao–Vazirani 2009).
- **Sparsity $= \dfrac{\text{cut size}}{\min\{|S|,|T|\}}$** — the denominator counts **VERTICES** and is what enforces **balance**; sparsest cuts are simultaneously **sparse and balanced**.
- **Modularity $Q = \sum_i \left[\frac{l_i}{|E|} - \left(\frac{d_i}{2|E|}\right)^2\right]$** — **observed intra-cluster edge fraction MINUS the fraction expected under random connection**. **Optimal clustering MAXIMISES modularity.** Know what $l_i$ and $d_i$ are ($d_i$ is a sum of **degrees**, not of edges).
- **The four challenges**: NP-hardness/cost, weighted-and-cyclic graphs, **high dimensionality (a row of the similarity matrix has dimension $|V|$)**, and **sparsity of both the graph and the derived similarity matrix**.

### SCAN
- **⚠️ $\Gamma(v)$ INCLUDES $v$ itself** — every worked computation depends on it ($|\Gamma(13)| = 2$, not 1).
- **Structural similarity $\sigma(v,w) = \dfrac{|\Gamma(v)\cap\Gamma(w)|}{\sqrt{|\Gamma(v)||\Gamma(w)|}}$** — a **cosine-style normalisation**. **Large for clique members, small for hubs and outliers**, and be able to explain *why* for a hub (**big denominator, modest intersection**).
- **SCAN = DBSCAN transplanted onto a graph**: **$\varepsilon$ replaces the radius, $\mu$ replaces MinPts**, and **core / directly reachable / reachable (transitive closure) / connected** are the same four notions.
- **⚠️ SCAN outputs clusters AND hubs AND outliers.** **Hubs bridge MANY clusters; outliers connect to FEW.** Both are **non-members** — the distinction is made *afterwards*, by looking at the structural similarities of the non-member.
- **Be able to redo the numeric check**: $\sigma(13,9) = 2/\sqrt{10} = 0.63 < 0.7$ ⇒ 13 not core; for vertex 8 the three values $0.67, 0.82, 0.67$ leave only one neighbour above threshold ⇒ $|N_\varepsilon(8)| = 1 < \mu = 2$ ⇒ not core.
- **Running time $O(|E|)$, i.e. $O(|V|)$ for sparse networks** — **stable as the graph grows**; **the real difficulty is TUNING $\varepsilon$ and $\mu$**, exactly as in DBSCAN.

### Cross-chapter connections
- **SCAN is DBSCAN** (Chapter 5 §5.2) with structural similarity in place of Euclidean density — core objects, direct reachability, transitive closure, connectivity, and the same **two-parameter tuning problem**.
- **Strategy (1) — build a similarity matrix, then cluster — feeds directly into Chapter 9**: the resulting matrix is **high-dimensional (dimension $|V|$) and sparse**, precisely the setting of high-dimensional clustering. **Spectral clustering (Chapter 9 §7) is literally a graph method**: it builds an **affinity matrix** and cuts the graph via eigenvectors — **normalized cuts** are the same family as the cuts of §8 here.
- **Bi-partite graphs (customers × products, authors × conferences) are the same objects as the BI-CLUSTERING matrices of Chapter 9 §5** — gene × condition matrices are bipartite graphs in disguise.
- **Identifying "customers out of the clusters"** and **SCAN's outliers** connect to **Chapter 7 (Outlier Analysis)**, especially the **clustering-based** detection methods.
- **The similarity-to-dissimilarity conversion ($1-s$) and cosine-style normalisation** come from **Chapter 2's** proximity measures.
- **Biological networks** (protein interactions, metabolic reactions, gene regulatory networks) are the course's recurring application domain — the same one that motivated **micro-array bi-clustering** (Chapter 9) and **colossal patterns** (Chapter 8).
- **Constraints on the clustering of spatial/graph data** are the subject of **Chapter 11**, which also uses a **graph construction (the visibility graph)** to redefine distances.

---

*File auto-generated by merging `10-GraphClustering.pdf` (professor's slides, 33 pages) and `10 - Graph Clustering sbobine.pdf` (lecture notes, 22 pages). Formulas that the slide deck stored as images or that the PDF text extraction garbled (the SimRank recursion and its iterative form, the tour probability, the expected and expected-$f$ meeting distances, the sparsity of a cut, modularity, the SCAN structural similarity and the CORE/DirREACH/CONNECT predicates) have been reconstructed in their standard form. For questions about this chapter, refer only to this file.*
