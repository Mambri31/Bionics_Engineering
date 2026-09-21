# Chapter 14 — Federated Learning and Explainable AI: a Favorable Synergy

> **Course:** Data Mining and Machine Learning / Bioinspired Computational Methods — Biological Data Mining
> **Instructor:** Prof. Francesco Marcelloni (University of Pisa) — Coordinator, Artificial Intelligence Group; IT2PAO lab (joint laboratory with LogObject AG, Switzerland)
> **Sources merged into this file:** professor's slides (`14-FederatedLearning.pdf`, 17 pages) + lecture notes (`14 - FederatedLearning sbobine.pdf`, 25 pages — lessons **L14, 23/04/2026** and **L15, 28/04/2026**)
>
> **⚠️ Note**: this chapter is **not based on Han–Kamber–Pei**. It is an original deck built on the professor's own research group's publications (Corcuera Bárcena, Ducange, Marcelloni, Renda, Bechini) plus FL survey literature (Li et al. 2021, Zhu et al. 2021, Kairouz et al.).
>
> **⚠️ Note on the sources**: the **second half of the slide deck (federated decision trees and federated c-means) is almost entirely FIGURES** — the algorithmic detail there comes from the lecture notes, which are far more complete. Conversely the **taxonomy tables** come from the slides. The two sources agree throughout.
>
> **Instructions for Claude:** this is the single reference file for Chapter 14. When the user asks study questions about this chapter, use **only** this file as context (no need to re-read the original PDFs unless a detail is missing). The file merges the official slide content with the lecture's discursive explanations and worked examples, keeping both levels of detail.

---

## Chapter outline (the official syllabus slide)
1. **Federated Learning**
2. **Explainable Artificial Intelligence**
3. **Federated Learning of XAI models for Regression**
4. **Federated Clustering**
5. **Conclusions**

---

## 1. The motivation: two clashing requirements

### 1.1 The EU view of AI

The chapter opens from the **seven requirements toward Trustworthy AI** (the European Commission High-Level Expert Group guidelines — the same framework as Chapter 13):

**Human agency and oversight · Accountability · Technical robustness and safety · Societal and environmental wellbeing · Diversity, non-discrimination and fairness · Privacy and data governance · Transparency**

> **Two of the seven are highlighted on the slide, because they are the two this chapter is about:**
> - **PRIVACY AND DATA GOVERNANCE** — *"the need to collect data to train accurate AI models **CLASHES** with the need to preserve the privacy of data owners."*
> - **TRANSPARENCY** — *"AI systems and their decisions should be explained in a manner adapted to the stakeholder concerned."*
>
> **⇒ The answer proposed by the group: Fed-XAI — FEDERATED LEARNING OF eXPLAINABLE AI MODELS.**

*(Context from the lecture: these guidelines are not the only regulatory framework — there is also the whole data-protection regulation. And this research sits inside **HEXA-X**, the European 6G flagship project, Horizon 2020 Grant Agreement ID 101015956, where **"the use of AI will be fundamental"**.)*

### 1.2 ⚠️ The privacy problem, and MACHINE UNLEARNING

> **From a legal perspective, individuals have the right to request that companies REMOVE their personal data from trained models**, especially when the information is sensitive or private. *For instance, if a user interacts with a system like ChatGPT and later discovers that private information has been incorporated into the model, they should in principle have the right to have it removed.*
>
> **⚠️ But from a practical standpoint this is EXTREMELY CHALLENGING.** Modern large-scale models have **a vast number of parameters**, which makes it **difficult to identify and isolate the contribution of a single data point**. **Models do not store data explicitly — they encode it IMPLICITLY within their parameters.** As a result, **removing specific information is not straightforward.**
>
> **This has led to the emergence of MACHINE UNLEARNING**, a research area focused on **enabling models to "FORGET" specific data used during training.** Despite recent progress, **effective unlearning for large models remains a complex and OPEN problem.**

**So the problem splits into two aspects** (lecture):
1. **Detection and removal of private data** — *is there private information in the model, how did it get in, and how do we take it out?* ⇒ **machine unlearning**.
2. **Data availability and distributed learning** — *the data needed to train a model are typically NOT in one place.*

### 1.3 Why do we need Federated Learning?

> - **Machine learning algorithms, especially DEEP LEARNING algorithms, are DATA HUNGRY.**
> - **Data are generally spread over different devices with different owners and under the protection of privacy restrictions.**
> - **In practice, we cope with ISOLATED DATA ISLANDS and WE CANNOT TRANSFER DATA.**

> **The fundamental tension** (lecture): **larger and more diverse datasets generally lead to more accurate and reliable models**, so **we would like to leverage ALL available data — but we are NOT ALLOWED to move or centralize raw data** from different sources, because of **privacy, legal and ownership constraints.**

## 2. What Federated Learning is

> **DEFINITION** *(Kairouz et al., "Advances and Open Problems in Federated Learning", Foundations and Trends in Machine Learning, Vol. 14)*:
> **"Federated learning is a machine learning setting where multiple entities (CLIENTS) collaborate in solving a machine learning problem… Each client's RAW DATA IS STORED LOCALLY AND NOT EXCHANGED OR TRANSFERRED; instead, FOCUSED UPDATES intended for IMMEDIATE AGGREGATION are used to achieve the learning objective."**

**The two things FL does** (slide):
> - **to AGGREGATE the updates, learned at the edge devices, of the distributed individual versions of the global ML model, and to exploit the aggregation for UPDATING the model;**
> - **to BROADCAST the model to allow edge devices to continuously REFINE their individual versions.**

> **The intuition** (lecture): **instead of transferring raw data, FL relies on the exchange of AGGREGATED INFORMATION or MODEL PARAMETERS.** Each client **trains a local model on its own data** and shares **only the resulting updates (gradients or weights)**; these are **aggregated into a global model** capturing the knowledge of all participants.
>
> **⚠️ The crucial privacy property: THE SERVER DOES NOT HAVE DIRECT ACCESS TO THE RAW DATA, and therefore CANNOT RECONSTRUCT IT in practice.**

### 2.1 The worked example: medical imaging

> **Problem**: build a model for the **detection of lung nodules from CT scans**. Each hospital (e.g. **Pisa** or **Florence**) independently manages and stores the CT scans of its own patients. **Due to privacy regulations it is generally NOT possible to transfer raw medical data between hospitals or to create a centralized repository.**
>
> **At the same time, combining data from multiple hospitals would significantly improve the quality and reliability of the model.** *How do we exploit all the information without physically sharing the data?*
>
> **The federated protocol:**
> 1. **A central server defines the model architecture** (for example a neural network) and **shares it with all participating clients** (the hospitals).
> 2. **Each hospital trains a LOCAL INSTANCE of the model using its own private data.**
> 3. **Each client sends only the MODEL UPDATES (weights or gradients) to the server — not the raw data.**
> 4. **The server AGGREGATES these updates (typically by averaging) to obtain a GLOBAL MODEL.**
> 5. **The global model is sent BACK to the clients**, who use it to further refine their local models.
> 6. **The process is repeated ITERATIVELY**, progressively improving performance.
>
> **The two key benefits:**
> - **PRIVACY PRESERVATION**: sensitive medical data never leave the local repositories, ensuring compliance with data protection regulations;
> - **KNOWLEDGE SHARING**: the global model incorporates information from **all** distributed datasets.
>
> **At the end of training, EVERY client has access to the final global model**, collaboratively learned from all participants.

### 2.2 An aside on reading the literature

The "Federated learning is a hot topic" slide shows the **Scopus publication count rising rapidly**. The lecture attached a methodological warning worth keeping:

> **Not all published papers are equally reliable.** Prefer works from **well-established publishers (IEEE, ACM, Elsevier, Wiley, Springer…)**. **The key aspect is whether the paper underwent PEER REVIEW.** Authors often share **preliminary versions (preprints)** — **platforms such as arXiv host NON-peer-reviewed manuscripts.** **If no clear information about the review process or the venue is available, treat the paper with caution.**

## 3. A taxonomy of Federated Learning Systems

*(Li Q. et al., "A Survey on Federated Learning Systems: Vision, Hype and Reality for Data Privacy and Protection", IEEE TKDE, 2021.)*

**Six main characterizing aspects:**

| Aspect | Categories |
|---|---|
| **Data partitioning** | **Horizontal · Vertical · Hybrid** |
| **ML model** | **Linear models · Decision trees · Neural networks · …** |
| **Privacy mechanism** | **Differential privacy · Cryptographic methods · …** |
| **Communication architecture** | **Centralized · Non-centralized** |
| **Scale of federation** | **Cross-silo · Cross-device** |
| **Motivation of federation** | **Incentive · Regulation** |

> **⚠️ The course's focus, stated on the slide: "We are mainly interested in CENTRALIZED, CROSS-DEVICE FLS."**

### 3.1 Data partitioning

> **Based on how data are distributed among the parties over the SAMPLE space and the FEATURE space.**

**HORIZONTAL FL (HFL)** — **the datasets of different parties SHARE THE SAME FEATURE SPACE but have little or no intersection on the SAMPLE SPACE.**

```
Device #1 | Feature 1 … Feature N |  Sample 1, Sample 2, Sample 3
Device #2 | Feature 1 … Feature N |  Sample 4, Sample 5
Device #3 | Feature 1 … Feature N |  Sample 5, Sample 6, Sample 7
```

> *Each client contains data described by the SAME set of features, but referring to DIFFERENT instances.*
> **Example: CT scans collected in different hospitals** — each hospital stores data with the same structure (same features describing the scans) but referring to **different patients**.
> **This is a NATURAL data partitioning especially for the CROSS-DEVICE setting**, where different users try to improve model performance on the same task — e.g. **smartphones**, each holding local data (user interactions) described by the same features.

**VERTICAL FL (VFL)** — **the datasets of different parties have the SAME or SIMILAR SAMPLE SPACE but DIFFER IN THE FEATURE SPACE.**

```
Device #1 | Feature 1, Feature 2          |  Samples 1..6
Device #2 | Feature 3, Feature 4, Feature 5 |  Samples 1..6
Device #3 | Feature 6, Feature 7          |  Samples 1..6
```

> **Example: a MUNICIPALITY REGISTRY and HOSPITAL DATA** referring to the same individuals — the hospital dataset holds **medical** information, the municipal dataset **demographic or socio-economic** attributes. **The same individuals are described by different features across different repositories.**
> **Due to privacy and ownership constraints these datasets cannot be directly merged**, yet **combining such complementary information could be highly beneficial** for tasks such as clustering or predictive modelling.

**HYBRID FL** — **the partition may be a hybrid of horizontal and vertical partition.**

### 3.2 ML models

> **There have been many efforts in developing new models, or reinventing current models, for the federated setting:**
> - **Neural networks** — many studies on **federated stochastic gradient descent**, which can be used to train NNs;
> - **Decision trees** — widely used, as they are **highly efficient to train compared with NNs** (FLS studies for **Gradient Boosting Decision Trees, GBDTs**, have been proposed recently);
> - **SVM** — successfully trained by exploiting a **federated stochastic gradient descent** algorithm *(a well-known approach especially with text documents)*.

### 3.3 Privacy mechanisms

> **⚠️ Model parameters exchanged during FL rounds MAY LEAK SENSITIVE INFORMATION about the data.**
> **Beyond attacks targeting user privacy, there are also other classes of attacks** — e.g. **an adversary might attempt to BIAS the model to produce inferences preferable to the adversary.**

| Technology | Main characteristics |
|---|---|
| **Differential Privacy** | **Add properly tuned RANDOM NOISE to mask the influence of an individual instance on the output.** |
| **Secure Multi-Party Computation** | **Enables two or more parties to compute an agreed-upon function of their private inputs in a way that only reveals the INTENDED OUTPUT to each party, keeping the inputs private.** |
| **Homomorphic Encryption** | **Enables parties to perform mathematical operations DIRECTLY ON ENCRYPTED DATA without decrypting them.** |
| **Trusted Execution Environments (TEE)** | **Provide the ability to trustably run code on a remote machine, EVEN IF YOU DO NOT TRUST THE MACHINE'S OWNER.** TEEs may provide **confidentiality, integrity and remote attestation.** |

> **The lecture's framing of the threat**: although FL avoids transferring raw data, **it still requires communication of aggregated information over the network** — and **any data transmitted across a network can be INTERCEPTED, ALTERED or CORRUPTED.**
> **⚠️ If an attacker can access and MODIFY the updates sent by clients, the integrity of the global model can be severely compromised. EVEN A SMALL NUMBER of malicious or tampered updates can significantly degrade performance or steer the model toward biased behaviour.**
> **So the core challenge is not only how to exchange information EFFICIENTLY, but how to ensure its SECURITY and INTEGRITY.**
>
> **NB on differential privacy: if someone is able to capture the data, they are NOT able to retrieve the raw data.**

### 3.4 Communication architecture

| | **Centralized** | **Non-centralized** |
|---|---|---|
| **Data flow** | **ASYMMETRIC**: the **server aggregates** the information (gradients or model parameters) from the clients and **sends back the updated global model**; **iterated until a convergence criterion is met** | **communications are performed AMONG THE PARTIES**; **every party can update the global parameters directly** — **no need for a trusted central aggregating server** (every client could act as a server) |
| **Status** | **the typical one** | — |
| **Major challenge** | — | **it is HARD TO DESIGN A PROTOCOL that treats every member almost fairly with reasonable communication overhead** |

> **A practical note from the lecture**: **clients may participate in a DYNAMIC and INTERMITTENT manner.** With a federation of smartphones, **a device may not always be available** because of **connectivity issues, power constraints or user activity** — **if a client is temporarily unreachable it is effectively EXCLUDED from the federation at that time.**

### 3.5 Scale of federation: cross-silo vs cross-device

> **The main differences lie in the NUMBER OF PARTIES and the AMOUNT OF DATA stored in each party.**
> **Cross-silo: the clients are really SERVERS in some organization** — computers with **high storage and computation capability**.
> **Cross-device: the clients are a very large number of MOBILE or IoT devices** with **limited storage and computation capability.**

| | **Cross-silo** | **Cross-device** |
|---|---|---|
| **Setting** | training a model on **siloed data**; clients are **different organizations or geo-distributed datacenters** | the clients are **a very large number of mobile or IoT devices** |
| **Data distribution** | data generated locally and remaining decentralized | each client stores its own data and **cannot read the data of other devices** |
| **Data availability** | **almost always available** | **only a FRACTION of clients are available at any one time**, often with **diurnal or other variations** |
| **Scale** | **typically 2 – 100 clients** | **massively parallel, up to $10^{10}$ clients** |
| **Primary bottleneck** | **computation OR communication** | **COMMUNICATION is often the primary bottleneck** — clients generally use **wi-fi or slower connections** |
| **Client reliability** | **relatively few failures** | **⚠️ HIGHLY UNRELIABLE — 5% or more of the clients participating in a round are expected to FAIL OR DROP OUT** |
| **Data partition axis** | **partition is FIXED**; can be **example-partitioned (horizontal) OR feature-partitioned (vertical)** | **FIXED partitioning BY EXAMPLE (horizontal)** |

### 3.6 Motivation of federation

> **In real-world applications, individual parties need the MOTIVATION to get involved. The motivation can be REGULATIONS or INCENTIVES. The parties inside the system can be COLLABORATORS AS WELL AS COMPETITORS.**

**Example 1 — Google Keyboard (Gboard)**: *"Google cannot prevent users who do not provide data from using Gboard. **But those who agree to upload input data may enjoy a HIGHER ACCURACY of word prediction.**"*

**Example 2 — fault prediction for industrial equipment** (from the lecture):
> A manufacturer produces and distributes **the same equipment to multiple customers** operating in different environments. It would be highly beneficial to **collect data from all customers** to observe different failure patterns and build a robust predictive model.
> **⚠️ But customers are often UNWILLING to share their data** — for **privacy, confidentiality, or COMPETITIVE reasons**, especially when the data could reveal sensitive operational information.
> **FL offers a solution**: train collaboratively while keeping data local, and **incentivize participation by distributing the resulting predictive model back to the clients.** **Each participant gains access to an improved model incorporating knowledge from all clients, without exposing its own data** — a **mutually beneficial framework**.

## 4. Popular approaches: FedAvg and FedSGD

**The notation** (used by both algorithms):

| Symbol | Meaning |
|---|---|
| $C$ | **fraction of clients that participates in each federated round** |
| $K$ | **total number of clients** (indexed by $k$) |
| $E$ | **number of training passes each client makes over its local dataset on each round** |
| $B$ | **local minibatch size used for the client updates** — **$B = \infty$ indicates that the full local dataset is treated as a SINGLE minibatch** |
| $P_k$ | **set of indexes of data points on client $k$**, with $n_k = |P_k|$ |

*(Zhu H., Xu J., Liu S., Jin Y., "Federated learning on non-IID data: A survey", Neurocomputing 465, 2021.)*

### 4.1 The two algorithms

> **FedSGD** — **each client $k$ computes the GRADIENT on its local data at the current model $w_t$, and the central server AGGREGATES these gradients and updates the global model.** *For the current global model $w_t$, the average gradient is calculated for each client $k$; the server then aggregates these gradients and applies the update.*
>
> **FedAvg** — **each client locally takes ONE OR MULTIPLE STEPS of gradient descent on the current model $w_t$ using its local data, and the server then takes a WEIGHTED AVERAGE of the resulting MODELS.**

> **⚠️ THE RELATIONSHIP TO MEMORISE: FedSGD COINCIDES WITH FedAvg for $C = 1$, $B = \infty$, $E = 1$.**
> *(All clients participate, the whole local dataset is one batch, and exactly one pass is made — so "multiple local steps" degenerates to a single gradient step.)*

### 4.2 How FedAvg runs, and why it is preferred

> **The mechanism** (lecture): **a central server coordinates the training across multiple clients. At each round only a SUBSET of clients participates.** Each selected client **trains a local copy of the model on its own data for a certain number of epochs**, then **sends its model updates (not the raw data) back to the server** — so at the end **there are different weights for different clients**. **The server computes a WEIGHTED AVERAGE of these updates, where the weights depend on the SIZE OF EACH CLIENT'S DATASET.** This produces a new global model, redistributed for the next round.

**The trade-off between the two:**

| | **FedSGD** | **FedAvg** |
|---|---|---|
| **Communication** | **more frequent** | **reduced overhead** |
| **Local computation** | **less** | **more** (computation is shifted to the clients) |
| **Verdict** | — | **more EFFICIENT and SCALABLE in real-world scenarios** |

> **Why neural networks in the first place** (lecture): the **first approaches to FL were mainly applied to neural networks, in particular MULTILAYER PERCEPTRONS (MLPs)**. The **architecture (number of layers and neurons per layer) is FIXED IN ADVANCE**; **training consists in estimating the WEIGHTS of the connections** so as to minimize a loss function. NNs are chosen because they **handle heterogeneous and non-linear data** and give **high accuracy**…
>
> **⚠️ …but at the cost of REDUCED INTERPRETABILITY — they are BLACK-BOX models.** In domains like **healthcare or finance** this is a real limitation, because **it is important not only to make accurate predictions, but also to UNDERSTAND AND JUSTIFY the model's decisions.** **This is exactly why the chapter now turns to XAI.**

## 5. XAI in the federated setting

### 5.1 Recap of the XAI framework

*(This slide compresses Chapter 13 — see that file for the full treatment.)*

**INTERPRETABLE MODELS — transparent models, interpretable by design.** Within transparency, **three levels are contemplated**:
> - **SIMULATABILITY** — **the ability of a model to be SIMULATED by a human;**
> - **DECOMPOSABILITY** — **the ability to explain EACH OF THE PARTS of a model;**
> - **ALGORITHMIC TRANSPARENCY** — **the ability of the user to understand the PROCESS followed by the model to produce output from input.**

**MODEL INTERPRETABILITY TECHNIQUES — models explained by external XAI techniques: POST-HOC explainability**, comprising the techniques corresponding to the most common ways humans explain things: **text explanations · visual explanations · local explanations · by example · by simplification · feature relevance.**

**And the trade-off**, restated here:
> **Generally, models that are MORE COMPLEX are inherently MORE ACCURATE. The two extremes are represented on the ACCURACY side by NEURAL NETWORKS — often called black-box models — and on the EXPLAINABILITY side by all RULE-BASED SYSTEMS such as decision trees.**
> *(Fernandez, Herrera, Cordon, Del Jesus, Marcelloni, "Evolutionary fuzzy systems for explainable artificial intelligence", IEEE Computational Intelligence Magazine, 2019.)*

### 5.2 ⚠️ The SHAP problem in a federated setting — and its elegant solution

This is one of the most examinable ideas in the chapter, and it comes from the lecture notes.

> **A key requirement of SHAP is the availability of a REFERENCE (background) DATASET, used to estimate feature contributions** *(see Chapter 13 §8.2)*.
>
> **⚠️ In a federated setting this creates a limitation: EACH CLIENT HAS ACCESS ONLY TO ITS LOCAL DATASET**, while data from other clients remain inaccessible. **As a result, explanations are computed using DIFFERENT REFERENCE DATASETS across clients.**
>
> **The consequence: THE SAME INPUT INSTANCE MAY RECEIVE DIFFERENT EXPLANATIONS depending on the client where SHAP is applied**, since each client uses a different local reference distribution. **While each explanation is valid LOCALLY, this lack of consistency is undesirable at the SYSTEM level, where a unique and coherent explanation would be preferable.**

**The solution — FEDERATED CLUSTERING as a provider of shared prototypes:**

> 1. **Compute clusters in a DISTRIBUTED manner**, leveraging data from all clients **without sharing raw data** — each client computes information about its local data (centroids, statistics) and sends it to the server, which **creates global clusters**.
> 2. **From these clusters, extract a set of PROTOTYPES that summarize the GLOBAL data distribution.**
> 3. **Share the prototypes with all clients and use them as a COMMON REFERENCE DATASET for SHAP.**
> 4. **Since the prototypes are SYNTHETIC and do not contain raw data, they PRESERVE PRIVACY while providing a unified reference.**
>
> **⇒ All clients can now generate CONSISTENT explanations for the same instance, because they rely on the same set of prototypes.**

> **⚠️ Note the architecture of the argument: this is WHY the chapter needs federated clustering (§7) — it is not a separate topic, it is the enabler of consistent federated explanations.**

## 6. Federated Decision Trees

### 6.1 Approach A — train locally, then AGGREGATE the trees

*(A. Argente-Garrido, C. Zuheros, M. V. Luzón, F. Herrera, "An interpretable client decision tree aggregation process for federated learning", Information Sciences, Vol. 694, 2025.)*

> **Each client trains a LOCAL decision tree using its own data.** With $N$ clients this produces **$N$ different decision trees**, which are **sent to a central server**. **The server's objective is to AGGREGATE these local models into a SINGLE GLOBAL decision tree.**

**⚠️ But the aggregation is not straightforward.** Decision trees can be transformed into **sets of if–then rules**, and **when combining rules from different clients, CONFLICTS may arise** — in particular **rules with the SAME ANTECEDENT but DIFFERENT PREDICTED CLASSES.**

**The two conflict cases shown on the slide:**

| Case | Example | Resolution |
|---|---|---|
| **Rules share a condition** (same class, different threshold values) — e.g. **rule 2A and rule 4B** | — | **the LESS RESTRICTIVE CONDITION is selected** |
| **Rules are INCOMPATIBLE** | **rule 1A has $x_0 \le 32.5$ while rule 3B has $x_0 > 39$** | **they cannot be merged** |

**Advantages and disadvantages** (lecture):

| Advantages | Disadvantages |
|---|---|
| **LIMITED COMMUNICATION**: each client trains its local model independently and **shares it only ONCE** with the server; the aggregated global model is then distributed back | **⚠️ the global tree contains a VERY HIGH NUMBER OF RULES** — it becomes **very complex and difficult to manage** |
| | **LOWER PREDICTIVE ACCURACY** than more flexible models (e.g. NNs), **especially when data are highly HETEROGENEOUS** |

### 6.2 Approach B — build ONE tree collaboratively (the better solution)

*(José Luis Corcuera Bárcena, Pietro Ducange, Francesco Marcelloni, Alessandro Renda, "Increasing trust in AI through privacy preservation and model explainability: Federated Learning of Fuzzy Regression Trees", Information Fusion, Vol. 113, 2025.)*

> **The idea: instead of training a complete model locally at each client and then merging trees at the server, CONSTRUCT THE DECISION TREE COLLABORATIVELY AND ITERATIVELY in a federated manner.**

#### Regression trees, and why variance is the criterion

> **Here we focus on REGRESSION TREES, where the leaves do not represent classes but contain NUMERICAL VALUES or LINEAR FUNCTIONS** (e.g. **hyperplanes** approximating the output distribution in a region of the input space).
>
> **The structure is still that of a decision tree** — internal nodes test conditions on input variables, leaves give the prediction — **but the MEANING OF THE LEAVES is different.**
>
> **The purpose**: **gradually PARTITION the input space into smaller, more specific regions.** The **path from root to leaf defines these subspaces** by applying a sequence of conditions; the leaf then **approximates the output values within that region.**
>
> **⚠️ The objective at each node is to obtain regions in which the OUTPUT VALUES ARE AS HOMOGENEOUS AS POSSIBLE** — *if the output values in a subspace do not vary much, their variation can be approximated by a line and, ideally, by a single number.*
>
> **⇒ Therefore, unlike CLASSIFICATION trees which rely on INFORMATION GAIN / ENTROPY, REGRESSION trees use VARIANCE REDUCTION as the splitting criterion.**

**The formulas:**

$$\text{Var}(y) \;=\; \frac{1}{n}\sum_{i=1}^{n}(y_i - \bar{y})^2$$

$$\text{VarReduction} \;=\; \text{Var}(S) \;-\; \left(\frac{n_L}{n}\text{Var}(S_L) + \frac{n_R}{n}\text{Var}(S_R)\right)$$

where $S$ is the original set and $S_L$, $S_R$ the two subsets obtained after the split. **The best split is the one that MAXIMIZES this variance reduction.**

#### ⚠️ The key trick: rewriting the variance as aggregated sums

> **The problem**: in a **centralized** setting computing the variance is straightforward — all data are in one place. **In a FEDERATED setting the server CANNOT collect all output values.** But it must evaluate the variance reduction of every candidate split.
>
> **The key observation: the variance can be rewritten in a form that depends ONLY ON AGGREGATED SUMS:**

$$\boxed{\;\text{Var}(y) \;=\; \frac{1}{n}\sum_{i=1}^{n} y_i^2 \;-\; \left(\frac{1}{n}\sum_{i=1}^{n} y_i\right)^{\!2}\;}$$

> *"The variance is just the difference between the mean of the squares of the output and the square of the mean of the output."*
>
> **⇒ To compute the variance, the server does NOT need to know each individual value $y_i$. It only needs THREE quantities:**
> 1. **the NUMBER OF INSTANCES;**
> 2. **the SUM OF THE OUTPUT VALUES;**
> 3. **the SUM OF THE SQUARED OUTPUT VALUES.**
>
> **Each client computes these locally and sends ONLY these aggregated values to the server.** The server **combines the partial sums from all clients and computes the global variance** — **obtaining the SAME RESULT as in a centralized setting, without accessing private data.**

#### The construction procedure

> 1. **Start from the ROOT NODE.** Each client **evaluates the possible splits locally** and computes the aggregated quantities needed to estimate the variance.
> 2. **These aggregated values are sent to the server.**
> 3. **The server combines the information from all clients, computes the variance reduction associated with each possible split, and SELECTS the attribute and condition giving the LARGEST reduction.**
> 4. **The same procedure is repeated at the following levels**: for each node, the clients compute local aggregated statistics **for the instances that fall into that node**; the server aggregates and chooses the best split.

> **⚠️ THE HEADLINE RESULT: the tree is built COLLABORATIVELY AND CONCURRENTLY by clients and server; the clients NEVER send raw data, only aggregated values — and the output is EXACTLY THE SAME as the regression tree that would be computed with open access to every individual value $y_i$.**
>
> **⚠️ And note the CONTRAST WITH FEDERATED NEURAL NETWORKS**: in federated NNs **each client trains a complete local model and the server averages the parameters**. In federated decision trees **the clients do NOT train complete local trees — the global tree is built STEP BY STEP from aggregated information.** *The aim is not to build a separate tree per client and combine them; it is to build ONE GLOBAL TREE through distributed, privacy-preserving computations.*

> **Why this matters for the chapter's thesis**: **an important advantage of regression trees is their INTERPRETABILITY** — each internal node is a condition on an input variable, so **we can understand how the model reaches a prediction.** **This is particularly relevant in FL, because the objective is not only to preserve privacy but also to obtain models that can be EXPLAINED.** ⇒ **Federated decision trees are a solution when privacy preservation AND model interpretability are BOTH required — the "favorable synergy" of the title.**

## 7. Federated Clustering: federated c-means (LLF-CM)

*(J. L. C. Bárcena, F. Marcelloni, A. Renda, A. Bechini, P. Ducange, "Federated c-Means and Fuzzy c-Means Clustering Algorithms for Horizontally and Vertically Partitioned Data", IEEE Transactions on Artificial Intelligence, vol. 5, no. 12, pp. 6426–6441, Dec. 2024.)*

### 7.0 The starting point: classical c-means

> **The aim is to find an optimal partition of the dataset into $C$ clusters, whose number must be defined at the beginning.** The optimization is **ITERATIVE**: starting from an initial configuration, the algorithm **repeatedly updates the assignment of the data points and the position of the cluster centers.** **Each cluster is represented by a PROTOTYPE (CENTROID), the mean value of the points assigned to it**, and **each point is assigned to the cluster with the NEAREST centroid.** Repeat until the partition is stable.
>
> **⚠️ In a federated environment this cannot be applied centrally, because the server cannot access all the data points — but the SAME LOGIC can be followed by exchanging only the information needed.**

### 7.1 Horizontal federated c-means

**Setting**: each client owns **different instances**, all described by the **same features**.

**The local assignment vector.** For each data owner $P_m$:
$$q^{(t),m} = \{q_1^{(t),m},\, q_2^{(t),m},\, \dots,\, q_{N_m}^{(t),m}\}, \qquad q_j^{(t),m} \in \{1, \dots, C\}$$
**specifying, at iteration $t$, the cluster to which each object of the local dataset $P_m$ belongs.** **Each data owner assigns every object to the cluster with the nearest center** — exactly the standard k-means behaviour.

#### The algorithm, step by step

**1. INITIALIZATION**
> The number of clusters **$C$** is fixed by the server and transmitted to all clients, along with two stopping parameters: **$\varepsilon$ (tolerance for convergence)** and **$T$ (maximum number of communication rounds)**.
> The server **generates $C$ initial centroids** $V^{(0)} = \{v_1^{(0)}, \dots, v_C^{(0)}\}$ — **as in classical k-means, these can be generated RANDOMLY** — and **sends them to all clients.**

**2. LOCAL CLUSTER ASSIGNMENT**
> At each round $t$ the server transmits the current centroid vector $V^{(t)}$ to each data owner. **Each client assigns its local objects to the nearest centroid**, computing $q_j^{(t),m} \in \{1,\dots,C\}$ for every object $j$.
> **⚠️ These assignments are computed LOCALLY — the individual objects are NOT transmitted to the server.**

**3. LOCAL AGGREGATED STATISTICS**
> For every cluster $\Gamma_c$, client $P_m$ computes:
> - **$n_c^{(t),m}$** — **the NUMBER of local objects assigned to cluster $c$**;
> - **$LS_c^{(t),m}$** — **the LINEAR SUM of the local objects assigned to cluster $c$**:
> $$LS_c^{(t),m} \;=\; \sum_{x_j^m \in \Gamma_c} x_j^m$$
>
> *Instead of sending the individual objects, the client sends only the SUM of the objects in each cluster, together with the COUNT.*

> **⚠️⚠️ THE PRIVACY CONDITION — a favourite exam detail.** **If a cluster contains only ONE local object, sending its linear sum would be EQUIVALENT TO SENDING THE OBJECT ITSELF**, and the server could directly recover the individual data point. **Therefore, if**
> $$n_c^{(t),m} \le 1 \quad\Longrightarrow\quad LS_c^{(t),m} = 0 \;\text{ and }\; n_c^{(t),m} = 0$$
> **Only when the number of local objects in a cluster is GREATER THAN ONE can the aggregated information be safely transmitted.**

**4. UPDATING THE CLUSTER CENTERS**
> Each client sends the pairs $\big(LS_c^{(t),m},\, n_c^{(t),m}\big)$ for each cluster. The server updates the global centroids:
> $$v_c^{(t+1)} \;=\; \frac{\sum_{m=1}^{M} LS_c^{(t),m}}{\sum_{m=1}^{M} n_c^{(t),m}}, \qquad c \in \{1,\dots,C\}$$
>
> **The intuition**: **in classical k-means a centroid is the AVERAGE of all objects assigned to the cluster. In the federated version the server cannot access the objects — but it can compute the SAME AVERAGE by aggregating the local sums and local counts.**
> The new centroids are then **transmitted again to the clients** and the procedure repeats.

**5. TERMINATION**
> The server compares the new centroids with the previous ones and stops if
> $$\|V^{(t+1)} - V^{(t)}\|_F < \varepsilon$$
> **or if the maximum number of rounds $T$ is reached.** The server then **sets a STOP FLAG and communicates it to all clients**: **if false, the clients proceed with another round; if true, the algorithm terminates.**

> **⚠️ THE EQUIVALENCE RESULT — and its ONE EXCEPTION.** *"The obtained clustering model corresponds to the result that would be achieved by applying c-means directly to the complete dataset obtained by merging all local datasets."*
> **BUT: this equivalence holds ONLY WHEN THE PRIVACY CONDITION DOES NOT REMOVE RELEVANT INFORMATION.** **If, for a given client and cluster, only ONE object belongs to that cluster, the corresponding sum and count are NOT sent — and in this case the final result MAY SLIGHTLY DIFFER from a fully centralized execution.**

### 7.2 Vertical federated c-means

**Setting**: **the same instances are shared across clients, but each client owns only a SUBSET OF THE FEATURES.** **No single client has a complete representation of the objects.**

> **⚠️ Why this is a fundamentally different problem**: **the assignment of an object to a cluster depends on the DISTANCE between that object and the centroids, and computing that distance requires ALL the features.** **In the vertical setting each client can only compute the contribution related to its OWN local features.**

**The key idea — exploit the ADDITIVITY of the Euclidean distance:**
> **The Euclidean distance is computed by SUMMING THE SQUARED DIFFERENCES OVER ALL FEATURES.** Since the features are distributed, **each client can compute only the part of the squared distance corresponding to its local features.** **These PARTIAL SQUARED DISTANCES are transmitted to the server, which AGGREGATES them to obtain the complete distance.**

#### The algorithm

**1. INITIALIZATION — projection of cluster centers**
> The server defines **$C$** and sets the stop flag to false. **Each client works with a PROJECTION of the cluster centers over its LOCAL FEATURE SPACE** — the global centroids exist in all dimensions at server level, but **each client only holds the components corresponding to the features it stores.** Each data owner $P_m$ randomly generates:
> $$V^{(0),m} = \{v_1^{(0),m},\, v_2^{(0),m},\, \dots,\, v_C^{(0),m}\}$$

> **⚠️ A general remark made here about initialization**: **c-means is an OPTIMIZATION-BASED algorithm — the result depends on minimizing a cost function, and the algorithm MAY CONVERGE TO A LOCAL MINIMUM.** **Different random initializations can lead to different final clusterings**, which is why clustering algorithms are **often executed SEVERAL TIMES with different initializations**, or combined with **smarter initialization strategies**.

**2. EXECUTION STAGE**
> First, **each client UPDATES the projection of the cluster centers using the features it owns**, based on the objects assigned to each cluster in the previous iteration — **for each cluster $\Gamma_c$ the projected center is the AVERAGE of the local feature values of the objects assigned to it.**
>
> Then, **each client evaluates the PARTIAL SQUARED DISTANCE between each object and each cluster center projection**:
> $$d_{j,c}^{(t),m} \;=\; \sum_{f}\big(x_{j,f}^{m} - v_{c,f}^{(t),m}\big)^2$$
> **This is NOT the complete distance — it is the LOCAL CONTRIBUTION of that client.** Each client transmits the **matrix of partial squared distances $D^{(t),m}$** to the server.

**3. SERVER-SIDE DISTANCE COMPUTATION AND ASSIGNMENT**
> The server **sums the corresponding entries** from all clients to obtain the complete distance:
> $$d_{j,c}^{(t)} \;=\; \sqrt{\sum_{m=1}^{M} d_{j,c}^{(t),m}}$$
> **This works because the Euclidean distance is ADDITIVE over the feature dimensions when expressed as a sum of squared differences.**
>
> The server then **assigns each object to the CLOSEST cluster** — $q_j^{(t)} \in \{1,\dots,C\}$ is the centroid with the smallest distance — **evaluates the stopping condition**, and, if not converged, **sends the updated cluster assignments and the stop flag to each client**, who use them to update their local centroid projections in the next round.

**4. TERMINATION**
> As in the horizontal version: stop when **the change between two consecutive distance matrices is smaller than $\varepsilon$**, or when **$T$ rounds are reached.**
> **At the end, the resulting clustering corresponds to the clustering that would be obtained by applying c-means to the complete dataset with all features available together — the difference being that THE COMPLETE FEATURE VECTORS ARE NEVER COLLECTED IN A SINGLE PLACE.**

> **Extension**: **the same general reasoning extends to FUZZY c-means.** *In classical c-means each object belongs to only one cluster; in fuzzy c-means a single object can belong to MULTIPLE clusters with different DEGREES OF MEMBERSHIP, allowing a softer form of clustering where the boundary between clusters is not rigid.*

### 7.3 ⚠️ The unifying intuition (horizontal vs. vertical)

This closing passage of the lecture is the single best summary of the chapter — **learn it as a whole.**

> **The difference between the two cases can be understood by asking WHICH QUANTITY IS DIFFICULT TO COMPUTE.**
>
> **HORIZONTAL** — each client owns **different instances** but **complete feature vectors**. ⇒ **The problem is the UPDATE OF THE CENTROIDS**, because **the objects belonging to the same cluster may be distributed across different clients**, so the server cannot directly compute the complete sum. **The solution: decompose the SUM into LOCAL PARTIAL SUMS** — each client sums the objects assigned to each cluster using its local data, and the server aggregates.
>
> **VERTICAL** — the clients refer to **the same instances** but each owns **only a subset of features**. ⇒ **The problem is computing the MEMBERSHIP of each object**, because the distance depends on **all** the features and **no single client can compute the complete distance**. **The solution: decompose the DISTANCE into PARTIAL SQUARED CONTRIBUTIONS** computed locally, which the server sums.
>
> **⇒ IN CONCLUSION, BOTH APPROACHES FOLLOW THE SAME GENERAL PRINCIPLE: A CENTRALIZED COMPUTATION IS REWRITTEN AS A SUM OF PARTIAL CONTRIBUTIONS THAT CAN BE COMPUTED LOCALLY.** In the horizontal case the principle is applied to the **centroid update**; in the vertical case to the **distance computation**. **In both cases the raw data are not moved — the COMPUTATION IS REORGANIZED so that only aggregated or partial information is exchanged.**

> **⚠️ Notice that the SAME principle governs the federated regression tree of §6.2** — there the centralized quantity is the **variance**, rewritten as $\frac{1}{n}\sum y_i^2 - (\frac{1}{n}\sum y_i)^2$ so that it too becomes a sum of locally computable partial terms. **Three algorithms, one idea.**

## 8. Two transversal issues

### 8.1 ⚠️ The non-IID problem

> **When we average the weights learned by different clients in federated neural networks, we are IMPLICITLY ASSUMING THAT THE DATA DISTRIBUTIONS ACROSS THOSE CLIENTS ARE SIMILAR.**
> **If this assumption holds**, the aggregation of local models produces a global model that **performs better than the individual local models**.
> **⚠️ BUT IF THE DATA DISTRIBUTIONS ARE VERY DIFFERENT from one client to another, simply averaging the weights MAY NOT IMPROVE the model — ON THE CONTRARY, IT MAY EVEN REDUCE THE ACCURACY of the global model.**
>
> **This is a well-known problem in federated learning. Ideally the clients should contain data that are IDENTICALLY, or at least SIMILARLY, DISTRIBUTED.** When this is not satisfied **the learning process becomes more complex**, because the algorithm must **take into account the HETEROGENEITY of the local data distributions.** **Several approaches have been proposed in the literature; this remains ONE OF THE MOST CHALLENGING ASPECTS of distributed and federated data analysis.**

### 8.2 Privacy and robustness considerations

> **Even if raw data are not directly transmitted, THE EXCHANGED INFORMATION MAY STILL REVEAL SENSITIVE DETAILS if not properly protected.** **Distances, partial sums, or other intermediate quantities could potentially be exploited by an attacker to INFER information about the original data.** For this reason, in real federated environments, **the exchange of aggregated data should be COMBINED WITH PROPER PRIVACY-PRESERVING PROTOCOLS.**
>
> **⚠️ And there is the MALICIOUS-CLIENT problem: some clients may not behave honestly.** **A client could MODIFY the partial sums, distances, or aggregated values before sending them to the server.** In that case **the server would update the model using corrupted information and the final result could be altered.** **ROBUSTNESS AGAINST ATTACKS is therefore a fundamental design aspect.**
>
> **This is why, in many scientific papers on federated learning, the authors explicitly mention the possibility of attacks and discuss the robustness of the proposed method.** *"It is not sufficient to design an algorithm that works from a computational point of view. It is also necessary to evaluate whether the algorithm remains reliable when some information is intercepted, modified, or generated by malicious participants."*
>
> **The algorithm must simultaneously: PRESERVE DATA PRIVACY, ALLOW CORRECT MODEL TRAINING, and REMAIN ROBUST AGAINST POSSIBLE ATTACKS.**

---

## Key points / potential exam pitfalls

### Motivation
- **⚠️ The chapter exists because TWO of the seven Trustworthy-AI requirements clash and must be satisfied together**: **privacy and data governance** (⇒ federated learning) and **transparency** (⇒ explainable AI). **The combination is called Fed-XAI.**
- **MACHINE UNLEARNING** is the research area on **making a model "forget" specific training data**. It is hard because **models encode data IMPLICITLY in their parameters, not explicitly** — an open problem for large models.
- **"Isolated data islands"** is the phrase for the practical situation: **data are spread across owners under privacy restrictions and CANNOT be transferred**, while **deep learning is DATA HUNGRY.**

### The definition and the protocol
- **Memorise the definition**: multiple clients collaborate; **raw data stays local and is never exchanged**; **focused updates intended for immediate aggregation** achieve the objective.
- **The six-step protocol** (server defines architecture → clients train locally → clients send updates only → server aggregates → global model broadcast back → iterate), and **the server CANNOT reconstruct the raw data.**
- **The two benefits: PRIVACY PRESERVATION and KNOWLEDGE SHARING.** At the end **every client has the final global model.**

### Taxonomy
- **Six aspects: data partitioning · ML model · privacy mechanism · communication architecture · scale of federation · motivation of federation.** The course focuses on **CENTRALIZED, CROSS-DEVICE**.
- **⚠️ HORIZONTAL = same FEATURES, different SAMPLES** (CT scans in different hospitals; smartphones). **VERTICAL = same SAMPLES, different FEATURES** (hospital + municipality registry). Getting these backwards is the single most likely slip in this chapter — **the adjective describes how you CUT the table, not what is shared.**
- **Horizontal is the natural partitioning for cross-device**; cross-silo can be either.
- **Four privacy mechanisms: DIFFERENTIAL PRIVACY (add tuned noise) · SECURE MULTI-PARTY COMPUTATION (reveal only the intended output) · HOMOMORPHIC ENCRYPTION (compute directly on encrypted data) · TEE (trustably run code on an untrusted machine).**
- **Centralized = asymmetric flow through a server; non-centralized = peer-to-peer, no trusted aggregator, and the hard part is DESIGNING A FAIR PROTOCOL with reasonable communication overhead.**
- **⚠️ Cross-silo vs cross-device numbers**: **2–100 clients vs up to $10^{10}$**; **data almost always available vs only a fraction available at any time**; **few failures vs 5%+ expected to drop out**; **computation-or-communication bottleneck vs COMMUNICATION as the primary bottleneck.**
- **Motivation = REGULATION or INCENTIVE**, and **parties can be collaborators AS WELL AS COMPETITORS** (Gboard; the equipment-manufacturer fault-prediction example, where the incentive is getting the improved model back).

### FedAvg / FedSGD
- **Know the five parameters: $C$ (fraction of clients per round), $K$ (total clients), $E$ (local passes per round), $B$ (local minibatch size, $B=\infty$ = whole dataset as one batch), $P_k$ / $n_k$ (client $k$'s data indexes and their number).**
- **⚠️ FedSGD = FedAvg with $C=1$, $B=\infty$, $E=1$.** This identity is the most quotable fact of §4.
- **FedSGD sends GRADIENTS after one step; FedAvg sends MODEL PARAMETERS after one or more local steps, and the server takes a WEIGHTED AVERAGE weighted by LOCAL DATASET SIZE.**
- **FedSGD: more communication, less local computation. FedAvg: less communication, more local computation ⇒ more efficient and scalable.**

### XAI in the federated setting
- **The three transparency levels: SIMULATABILITY (a human can simulate the model) · DECOMPOSABILITY (each part can be explained) · ALGORITHMIC TRANSPARENCY (the user understands the process from input to output).**
- **⚠️ THE SHAP CONSISTENCY PROBLEM**: SHAP needs a **background/reference dataset**; in FL **each client has only its own**, so **the SAME instance gets DIFFERENT explanations at different clients.** **The fix: FEDERATED CLUSTERING → shared synthetic PROTOTYPES → a COMMON reference dataset ⇒ consistent explanations, still privacy-preserving.** Be able to explain why **prototypes are safe** (synthetic, no raw data).

### Federated decision trees
- **Two approaches — know both and why the second wins.**
  - **(A) Aggregate $N$ locally trained trees at the server.** **Pros: minimal communication (one shot).** **Cons: rule CONFLICTS, a global tree with a HUGE number of rules, and lower accuracy under heterogeneity.** **Conflict rules: shared condition ⇒ take the LESS RESTRICTIVE one; $x_0 \le 32.5$ vs $x_0 > 39$ ⇒ INCOMPATIBLE.**
  - **(B) Build ONE tree collaboratively and iteratively** from aggregated statistics.
- **⚠️ REGRESSION trees use VARIANCE REDUCTION, not information gain/entropy** — leaves hold **numerical values or linear functions (hyperplanes)**, and the goal is **homogeneous output within each region.**
- **⚠️ THE CENTRAL TRICK: $\text{Var}(y) = \frac{1}{n}\sum y_i^2 - \left(\frac{1}{n}\sum y_i\right)^2$** ⇒ the server needs only **the COUNT, the SUM, and the SUM OF SQUARES** from each client. Be able to write both the variance and the VarReduction formulas.
- **⚠️ The result is EXACTLY EQUIVALENT to the centralized tree** — no approximation. **And the contrast with federated NNs**: NNs average complete local models; trees build one global model from partial statistics, with **no local complete tree ever trained.**

### Federated c-means
- **HORIZONTAL: the hard part is the CENTROID UPDATE.** Clients send **$\big(LS_c^{(t),m},\, n_c^{(t),m}\big)$** — **linear sum and count per cluster** — and the server computes $v_c^{(t+1)} = \frac{\sum_m LS_c}{\sum_m n_c}$.
- **⚠️⚠️ THE PRIVACY CONDITION: if $n_c^{(t),m} \le 1$ then BOTH $LS_c$ and $n_c$ are set to 0**, because **a linear sum over one object IS that object.** **And this is precisely why the federated result may SLIGHTLY DIFFER from the centralized one** — the only source of discrepancy. This pair of facts (the rule *and* its consequence) is the most examinable detail of §7.
- **VERTICAL: the hard part is the MEMBERSHIP/DISTANCE computation.** Clients send **matrices of PARTIAL SQUARED DISTANCES** $d_{j,c}^{(t),m} = \sum_f (x_{j,f}^m - v_{c,f}^{(t),m})^2$; the server computes $d_{j,c}^{(t)} = \sqrt{\sum_m d_{j,c}^{(t),m}}$ **because the Euclidean distance is ADDITIVE over feature dimensions.**
- **⚠️ Note WHO assigns objects to clusters in each case**: in **horizontal** the **CLIENTS** assign locally; in **vertical** the **SERVER** assigns, because only it can see the full distance. This asymmetry follows directly from what is partitioned.
- **Both stop on $\|V^{(t+1)}-V^{(t)}\|_F < \varepsilon$ or $t = T$**, with the server broadcasting a **STOP FLAG**.
- **Initialization is random and c-means may converge to a LOCAL MINIMUM** ⇒ run it several times or use smarter initialization.
- **The extension to FUZZY c-means: an object belongs to MULTIPLE clusters with DEGREES OF MEMBERSHIP.**
- **⚠️ THE UNIFYING PRINCIPLE — the best one-sentence answer for this chapter: "a centralized computation is REWRITTEN AS A SUM OF PARTIAL CONTRIBUTIONS that can be computed locally"** — applied to the **centroid update** (horizontal), the **distance** (vertical), and the **variance** (regression trees).

### Transversal issues
- **⚠️ NON-IID DATA: averaging weights implicitly assumes SIMILAR data distributions across clients. If distributions differ a lot, averaging may NOT improve — and may DEGRADE — the global model.** One of the most challenging open aspects of FL.
- **Privacy is not automatic**: **partial sums and distances can still leak information**, so aggregation must be **combined with privacy-preserving protocols**.
- **Malicious clients can TAMPER with the values they send**, corrupting the global model ⇒ **ROBUSTNESS against attacks is a design requirement, not an afterthought.**

### Cross-chapter connections
- **This chapter is the direct sequel to Chapter 13 (Explainable AI)**: the transparency/interpretability framework, the **accuracy-vs-interpretability trade-off**, **transparent vs post-hoc models**, and **SHAP with its background dataset** are all reused — here under the extra constraint that **data cannot be centralized**.
- **c-means / k-means, centroids, prototypes and the local-minimum problem** come straight from **Chapter 5 (Clustering)**; **fuzzy c-means** connects to the **fuzzy sets** of Chapter 13 §5.2. The **prototype** idea also links to Chapter 5's cluster representatives.
- **Decision trees, information gain vs. variance reduction, and overfitting** come from **Chapter 4 (Classification)** — here extended to **regression trees**.
- **The "rewrite a global statistic as a sum of local partial sums" trick** is the same algebraic move as **BIRCH's clustering features (CF vectors: $N$, $LS$, $SS$)** in Chapter 5 — note the **identical triple: count, linear sum, sum of squares.**
- **Horizontal vs. vertical partitioning** mirrors the **sample-space / feature-space distinction** that also underlies **bi-clustering** (Chapter 9) and **vertical data formats** in frequent pattern mining (Chapter 6).
- **The distributed/privacy motivation** connects back to the **constraint-based** philosophy of Chapter 11: in both cases, **real-world constraints force a redesign of an algorithm that is otherwise perfectly correct.**

---

*File auto-generated by merging `14-FederatedLearning.pdf` (professor's slides, 17 pages) and `14 - FederatedLearning sbobine.pdf` (lecture notes, 25 pages — lessons L14 of 23/04/2026 and L15 of 28/04/2026). The taxonomy tables come from the slides; the algorithmic detail of federated decision trees and federated c-means comes from the lecture notes, since those slides are almost entirely figures. Formulas stored as images (the variance identity, the linear-sum and centroid-update equations, the partial squared distances, the FedAvg/FedSGD parameters) have been reconstructed in standard form. For questions about this chapter, refer only to this file.*
