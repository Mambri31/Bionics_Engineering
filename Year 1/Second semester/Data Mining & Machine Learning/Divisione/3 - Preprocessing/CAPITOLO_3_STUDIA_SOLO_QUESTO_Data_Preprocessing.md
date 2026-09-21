# Chapter 3 — Data Preprocessing

> **Course:** Data Mining and Machine Learning / Bioinspired Computational Methods — Biological Data Mining
> **Instructor:** Prof. Francesco Marcelloni (University of Pisa)
> **Sources merged into this file:** professor's slides (`3-Preprocessing.pdf`, 65 pages, based on the Han–Kamber–Pei textbook) + lecture notes (`3 - Preprocessing sbobine.pdf`, 31 pages)
>
> **Instructions for Claude:** this is the single reference file for Chapter 3. When the user asks study questions about this chapter, use **only** this file as context (no need to re-read the original PDFs unless a detail is missing). The file merges the official slide content with the lecture's discursive explanations and worked examples, keeping both levels of detail.

---

## Chapter outline (official slide structure)
1. Data Preprocessing: An Overview
2. Data Quality
3. Major Tasks in Data Preprocessing
4. Data Cleaning
5. Data Integration
6. Data Reduction
7. Data Transformation and Data Discretization
8. Summary

---

## 1. Overview: Why Preprocess the Data? (Data Quality)

Data preprocessing groups **all the techniques used to prepare a dataset for ML/data mining applications**, structured in four macro-steps:

1. **Measure data quality** based on accuracy, completeness, consistency, timeliness, believability, interpretability.
2. **Data cleaning and integration** (merging with other databases/files).
3. **Data reduction** — can dimensionality/volume be reduced without losing relevant information? (dimensionality reduction, numerosity reduction).
4. **Data transformation and discretization** — normalization (scale features to a smaller range) and concept hierarchy generation (for categorical variables, used in ML approaches).

### Measures of data quality (multidimensional view)
- **Accuracy**: correct or wrong, accurate or not.
- **Completeness**: not recorded, unavailable, …
- **Consistency**: some values updated but not others, dangling references, …
- **Timeliness**: is the data updated in a timely way?
- **Believability**: how trustworthy/correct is the data considered to be?
- **Interpretability**: how easily can the data be understood?

---

## 2. Data Cleaning

Real-world data is **dirty**: lots of potentially incorrect data due to instrument faults, human/computer error, transmission error.

### Types of "dirtiness"
- **Incomplete**: lacking attribute values, lacking attributes of interest, or containing only aggregate data (e.g., `Occupation = " "`).
- **Noisy**: contains noise, errors, or outliers (e.g., `Salary = "-10"` — an obviously invalid value). We can have noise in databases too, not only in signals; some errors are easy to recognize (e.g., negative salary/age), others are much harder to detect.
- **Inconsistent**: discrepancies in codes or names (e.g., `Age="42"`, `Birthday="03/07/2010"`; rating scale changed from "1,2,3" to "A,B,C" mid-dataset), or discrepancies between duplicate records.
- **Intentional**: e.g., disguised missing data — using default values in place of missing data (e.g., "Jan 1" as everyone's birthday) increases bias and reduces reliability.

### Handling Missing Data
Missing data may be due to: equipment malfunction, inconsistency with other data (hence deleted), data not entered due to misunderstanding, data not considered important at entry time, or history/changes not registered.

**Strategies:**
1. **Ignore the tuple**: only reasonable when you have an enormous amount of data and removing one instance won't affect the output — usually done when the *class label* is missing (in classification). Not effective when the % of missing values per attribute varies a lot.
2. **Fill in manually**: tedious, often infeasible at scale.
3. **Fill in automatically**, options in increasing order of sophistication:
   - a **global constant** (e.g., "unknown") — risky, can create an artificial new class.
   - the **attribute mean**.
   - the **attribute mean restricted to samples of the same class** — much more precise/smarter.
   - the **most probable value**, inferred via Bayesian formula or decision tree.

### Handling Noisy Data — Smoothing

In many signal-based applications, true signal values change smoothly, while noise appears as rapid, random point-to-point changes. **Smoothing** reduces noise: individual points higher than their neighbors are decreased, points lower than their neighbors are increased, producing a smoother signal.

**1) Rectangular / unweighted sliding-average smooth** — the simplest algorithm. Each point is replaced by the mean of $m$ adjacent points ($m$ = smooth width). For a 3-point smooth:

$$S_j = \frac{Y_{j-1} + Y_j + Y_{j+1}}{3}$$

- If the underlying function is constant or changes linearly, **no bias** is introduced.
- A bias **is** introduced if the underlying function has a **nonzero second derivative** — e.g., at a local maximum (a peak), moving-window averaging always **reduces** the peak's value.
- Trade-off on window size: **larger windows reduce more noise**, but risk losing abrupt changes/sharp features (e.g., flattening peaks). The right size is a trade-off between denoising and preserving shape/information.

**2) Triangular smooth**: a *weighted* smoothing function — points closer to the center of the window get a higher weight. For a 5-point smooth, it is equivalent to **two passes of a 3-point rectangular smooth**, and is more effective than the rectangular filter at reducing high-frequency noise. The window width $m$ must be odd, and coefficients are symmetric around the central point — this symmetry is what preserves the x-axis position of peaks and other features.

**3) Savitzky-Golay smoothing filters** — the general form of these (and the previous) filters:

$$S_i = \sum_{n=-n_L}^{n_R} c_n\, Y_{i+n}$$

where $n_L$ = number of points used to the left of point $i$, $n_R$ = number of points to the right.
- Instead of approximating the local function by a **constant** (as the rectangular filter does, via the average), Savitzky-Golay fits a **polynomial of higher order** (typically quadratic or quartic) to the points in the window via least squares, and evaluates that polynomial at position $i$.
- Since least-squares fitting is a linear operation, the coefficients $c_n$ can be **precomputed once** (for a fixed window size $n_L, n_R$ and polynomial degree $m$) and reused as simple linear combinations on real data — this makes the filter very fast to apply.
- Example naming convention: filter labeled like $(n_L, n_R, \text{degree})$ — e.g. $(16,16,0)$ = a rectangular filter of degree 0.
- Comparing a 0-degree filter vs. a higher-degree filter: with a degree-0 (rectangular) filter, higher/sharper peaks with smaller widths get more reduced in amplitude; a higher-degree filter preserves peak heights better.

**Smoothing ratio** = smooth width / (number of points in the half-width of the peak).
- Increasing the smoothing ratio improves signal-to-noise ratio, but reduces peak amplitude and increases peak bandwidth (broadens it).
- If the goal is to preserve **true peak height/width**, use a smoothing ratio **< 0.2**.
- If the goal is only to find the **peak position**, larger smoothing ratios are fine, because smoothing has (almost) no effect on peak position (unless peaks broaden enough to overlap with neighbors).

**When TO smooth:**
1. For cosmetic reasons (nicer-looking graphic for visual inspection/publication).
2. If the signal will be further processed by an algorithm sensitive to high-frequency noise (e.g., detecting maxima/minima/inflection points via zero-crossings of derivatives).

**When NOT to smooth** (important — before statistical procedures like least-squares curve fitting):
- (a) smoothing doesn't meaningfully improve parameter-measurement accuracy from independent samples;
- (b) all smoothing is at least slightly "lossy" — it changes signal shape/amplitude;
- (c) it's harder to evaluate goodness-of-fit from residuals, because smoothed noise can be mistaken for real signal;
- (d) smoothing seriously **underestimates** the parameter errors predicted by propagation-of-error / bootstrap methods.

**Other noise-handling techniques:**
- **Binning**: sort data, partition into bins (typically equal-frequency), then smooth by bin means / bin median / bin boundaries (see §6, discretization, for the detailed mechanics — the same technique is used both for noise smoothing and for discretization).
- **Regression**: smooth by fitting the data to a regression function (linear or multiple).
- **Clustering**: used to detect and remove outliers.
- **Combined computer + human inspection**: detect suspicious values automatically, verify by a human (e.g., to deal with possible outliers).

---

## 3. Data Integration

**Data integration** combines data from multiple sources into a coherent store — useful because using multiple sources together can create incoherences.

- **Schema integration**: e.g., `A.cust-id` ↔ `B.cust-#` — integrating metadata from different sources.
- **Entity identification problem**: identifying that the same real-world entity is represented differently across sources (e.g., `Bill Clinton = William Clinton`; `Bill = William = Will` — same person recorded three different ways).
- **Detecting/resolving data value conflicts**: for the same real-world entity, attribute values differ across sources — due to different representations, different scales/units of measurement (e.g., metric vs. British units).
- After data integration, a **new data-cleaning pass** may be needed to remove redundancies introduced by merging.

### Handling Redundancy in Data Integration
Redundant data is very common when integrating multiple databases.
- **Object/entity identification**: the same attribute/object may have different names in different databases.
- **Derivable data**: one attribute may be a "derived" attribute computable from another table (e.g., annual revenue derivable from monthly figures).
- An attribute is **redundant** if it doesn't provide any different information from another attribute — one such attribute should be deleted to minimize dimensionality.
- Redundant attributes can be detected via **correlation analysis** and **covariance analysis**. Careful integration reduces/avoids redundancy and inconsistency, and improves both mining speed and quality.

### 3.1 Correlation Analysis for Nominal Data — the χ² (Chi-Square) Test

The χ² independence test checks whether two **categorical** (nominal) variables $A = \{a_1,\dots,a_c\}$ and $B=\{b_1,\dots,b_r\}$ are statistically related in a population.

Represent the data as a **contingency table**: rows = values of $A$, columns = values of $B$, each cell $o_{ij}$ = observed frequency (count) of the combination $(A=a_i, B=b_j)$.

$$\chi^2 = \sum \frac{(Observed - Expected)^2}{Expected}$$

The **expected** count under the independence assumption:

$$e_{ij} = \frac{\text{count}(A=a_i)\cdot \text{count}(B=b_j)}{N}$$

where $N$ = total number of instances/observations.

- If observed ≈ expected → the variables are (close to) **independent**, $\chi^2 \approx 0$.
- The **larger** $\chi^2$, the more likely the variables are related/correlated.
- The cells that contribute most to $\chi^2$ are those where the observed count is most different from the expected count.
- **Important**: the χ² test only works on **NOMINAL** data, since it's based on discrete interval/category counts.

**Worked example (like_science_fiction vs. play_chess)**:

| | Play chess | Not play chess | Sum (row) |
|---|---|---|---|
| Like sci-fi | 250 (90) | 200 (360) | 450 |
| Not like sci-fi | 50 (210) | 1000 (840) | 1050 |
| Sum (col.) | 300 | 1200 | 1500 |

(numbers in parentheses = expected counts)

$$\chi^2 = \frac{(250-90)^2}{90} + \frac{(50-210)^2}{210} + \frac{(200-360)^2}{360} + \frac{(1000-840)^2}{840} = 507.93$$

Such a large χ² value shows *like_science_fiction* and *play_chess* are strongly correlated in this group.

#### Hypothesis testing framework
- **Null hypothesis $H_0$**: no association exists between the two variables — they are statistically independent.
- **Alternative hypothesis $H_1$**: the two variables are related in the population.
- **Parametric test**: assumes data follows a distribution of a known form (normal, Bernoulli, …) with an unknown parameter to infer. **Non-parametric test**: doesn't require specifying the distribution's form. The χ² test is **non-parametric**.
- **⚠️ Correlation does not imply causality**: e.g., the number of hospitals and the number of car thefts in a city may be correlated, but both are actually caused by a third variable — city population.

#### Degrees of freedom
$$df = (r-1)(c-1)$$
where $r$ = number of rows, $c$ = number of columns. Intuition: this is the number of cells you'd need to "freely fix" before all other cell values become determined by the row/column marginal totals. (E.g., in a 2×2 table, fixing just one cell — plus the totals — determines all the rest, so $df=1$.)

#### Deciding independence via the χ² distribution
- The shape of the χ² distribution depends on the degrees of freedom, and it is **always right-skewed (asymmetric)**.
- **p-value**: the area under the curve *after* a given value — the probability of observing a value at least that extreme under $H_0$.
- If the p-value is **smaller than the significance level $\alpha$**, we **reject $H_0$** → the variables are proven to be statistically correlated.
- In practice, computing exact areas under the χ² curve is hard, so a **χ² table** is used: rows = degrees of freedom, columns = significance levels $\alpha$. The table cell = the **critical value**, such that $P(\chi^2 \ge \text{critical}) = \alpha$.
- A smaller $\alpha$ → higher confidence against false positives (Type I error: wrongly rejecting $H_0$), but lowering $\alpha$ also increases the false-negative rate (Type II error).
- If $\alpha = 0.05$, the confidence level is $(1-\alpha)\cdot 100\% = 95\%$.

**Procedure:**
1. Compute the χ² statistic.
2. Compare it to the critical value (from the table, given $df$ and $\alpha$).
3. If $\chi^2 \ge$ critical value → **reject $H_0$** → correlation between the variables is proven.

**Worked example (education vs. marital status)**: random sample of $n=300$ people, $df=(5-1)(4-1)=12$. Computed $\chi^2 = 23.57$. Critical value from the table (for $df=12$) $= 21.026$. Since $23.57 \ge 21.026$ → reject $H_0$ → education and marital status **are related** in this population.

### 3.2 Correlation Analysis for Numeric Data — Pearson's Correlation Coefficient

$$r_{A,B} = \frac{\sum_{i}(a_i-\bar{A})(b_i-\bar{B})}{n\,\sigma_A\,\sigma_B}$$

where $n$ = number of tuples, $\bar{A},\bar{B}$ = means of $A,B$, $\sigma_A,\sigma_B$ = standard deviations of $A,B$, and $\sum(a_ib_i)$ is the sum of the AB cross-products.

- $r_{A,B} > 0$: $A$ and $B$ are **positively correlated** (both increase together) — higher $r_{A,B}$ = stronger correlation.
- $r_{A,B} < 0$: $A,B$ are **negatively correlated** (one increases as the other decreases).
- $r_{A,B} = 0$: $A,B$ are **independent** / uncorrelated.
- Generally we look at the **absolute value** of $r_{A,B}$ to gauge correlation strength; the sign tells us the direction.
- **⚠️ Pearson's coefficient only detects LINEAR relationships** — the non-linear analogue is the **Spearman coefficient**.
- Range: $r_{A,B} \in [-1, 1]$.
- Linear correlation can be visualized easily via scatter plots (perfect negative correlation ≈ $-1$, perfect positive ≈ $+1$).

**Equivalent "standardized dot product" view**: standardize $A,B$ then take their dot product:

$$a'_k = \frac{a_k - mean(A)}{std(A)}, \qquad b'_k = \frac{b_k - mean(B)}{std(B)}$$
$$correlation(A,B) = A' \cdot B'$$

### 3.3 Covariance

Similar to correlation, but not normalized by the standard deviations:

$$Cov(A,B) = E\big[(A-\bar{A})(B-\bar{B})\big] = \frac{1}{n}\sum_i a_ib_i - \bar{A}\bar{B}$$

and Pearson's correlation coefficient can be expressed from covariance as:

$$r_{A,B} = \frac{Cov(A,B)}{\sigma_A\sigma_B}$$

- **Positive covariance**: if $Cov_{A,B}>0$, then $A$ and $B$ both tend to be larger than their expected values together.
- **Negative covariance**: if $Cov_{A,B}<0$, then when $A$ is larger than expected, $B$ tends to be smaller than expected.
- **Independence**: $Cov_{A,B}=0$ is a **necessary but NOT sufficient** condition for independence — some variable pairs have zero covariance but are **not** independent. Zero covariance implies independence only under additional assumptions (e.g., data follow multivariate normal distributions).

**Worked example (stocks A, B)**: values over a week $(2,5),(3,8),(5,10),(4,11),(6,14)$.
$$E(A) = \frac{2+3+5+4+6}{5} = 4, \qquad E(B) = \frac{5+8+10+11+14}{5} = 9.6$$
$$Cov(A,B) = \frac{2\cdot5+3\cdot8+5\cdot10+4\cdot11+6\cdot14}{5} - 4\times9.6 = 4$$
Since $Cov(A,B)>0$, stocks $A$ and $B$ tend to rise/fall **together**.

---

## 4. Data Reduction

**Data reduction**: obtain a reduced representation of the dataset that is much smaller in volume but produces the same (or almost the same) analytical results.

**Why?** Databases/data warehouses may store terabytes of data; complex analysis on the full dataset can be extremely slow / computationally costly. Importantly, speeding up computation is often **not the main goal** — data reduction is mostly used to **reduce redundancy** (feature selection) and to increase the **reliability** of the analysis.

**Strategies:**
- **Dimensionality reduction**: remove unimportant attributes — wavelet transforms, PCA, feature subset selection, feature/attribute creation.
- **Numerosity reduction** (sometimes just called "data reduction"): smaller data *representations* via parametric models (e.g., regression, log-linear) or non-parametric models (histograms, clustering, sampling, data cube aggregation).
- **Data compression**.

### 4.1 Dimensionality Reduction

Motivated by the **curse of dimensionality**: as dimensionality increases, data becomes increasingly **sparse**, making density/distance-based methods (crucial for clustering, outlier analysis) less meaningful, while the number of possible subspace combinations grows **exponentially**.

Dimensionality reduction helps:
- avoid the curse of dimensionality;
- eliminate irrelevant features and reduce noise;
- reduce time/space required for data mining;
- allow easier visualization.

Two general approaches: **transform** the problem into a new space where we compute results (PCA), or use a **heuristic approach** on the original space — select attributes with the highest relevance to the output and lowest redundancy among themselves (feature selection).

#### Principal Component Analysis (PCA)

**Goal**: find a **projection** (a new coordinate space) that captures the largest amount of variation in the data, so that distant objects stay separated while instances of the same class stay close — i.e., **minimize inter-class distance while maximizing intra-class variance**... actually more precisely: minimize *reconstruction/projection error* while maximizing captured variance (spread) of the projected points.
- Practically: the original data is projected onto a much smaller space. We find the **eigenvectors of the covariance matrix** — these eigenvectors define the new space (principal components).
- The projection error is smaller than in the original data representation, and the newly-projected points are **more widely spread** (more variance) than a naive/poor projection would give.

**PCA steps** — given $N$ data vectors of $f$ dimensions, find $k \le f$ orthogonal vectors (principal components) that best represent the data:

1. **Normalize the input data**: bring each attribute to the same range.
2. **Compute $k$ orthonormal (unit) vectors** — the principal components. Each input vector is a linear combination of the $k$ principal component vectors.
3. Principal components are **sorted by decreasing significance/strength** (i.e., by the amount of variance they explain).
4. **Reduce dimensionality** by discarding the weak components (those with low variance) — using only the strongest components still lets you reconstruct a good approximation of the original data.
5. **Works for numeric data only.**

**Detailed steps and formulas:**

**Step #1 — Calculate the adjusted dataset $A$**: subtract, from each dimension, the mean of that dimension across all $N$ samples (so the adjusted data has zero mean per dimension). $M$ = vector of per-dimension means.

**Step #2 — Calculate the covariance matrix $C$ from the adjusted dataset $A$**:

$$Cov(X,Y) = \frac{\sum_{i=1}^{N}(X_i-\bar{X})(Y_i-\bar{Y})}{N-1}$$

Since the adjusted dataset $A$ already has zero mean per dimension, the covariance matrix simplifies to:

$$C = \frac{A\,A^T}{N-1}$$

*Worked example* — $N=5$ people, attributes Height, test Score, Age ($f=3$):

| | S1 | S2 | S3 | S4 | S5 | Mean |
|---|---|---|---|---|---|---|
| Height | 64 | 66 | 68 | 69 | 73 | 68 |
| Score | 580 | 570 | 590 | 660 | 600 | 600 |
| Age | 29 | 33 | 37 | 46 | 55 | 40 |

$$Var(Height) = \frac{(64-68)^2+(66-68)^2+(68-68)^2+(69-68)^2+(73-68)^2}{5-1} = \frac{46.0}{4} = 11.50$$

$$Covar(Height,Score) = \frac{(64-68)(580-600)+(66-68)(570-600)+(68-68)(590-600)+(69-68)(660-600)+(73-68)(600-600)}{5-1} = \frac{200}{4} = 50.0$$

Resulting covariance matrix:

| | Height | Score | Age |
|---|---|---|---|
| Height | 11.5 | 50 | 34.75 |
| Score | 50 | 1250 | 205 |
| Age | 34.75 | 205 | 110 |

**Step #3 — Calculate eigenvectors and eigenvalues of $C$**, and order them by variability:
- **Large eigenvalue** → corresponding eigenvector has **high** variability (important direction).
- **Small eigenvalue** → corresponding eigenvector has **low** variability → introduces redundancy, can be **discarded**, reducing dimensionality of the new basis.

**Step #4 — Transform the dataset to the new basis**:

$$F = E^T A$$

where $F$ = transformed dataset (fewer dimensions than $A$), $E^T$ = transpose of the eigenvector matrix. Since $E$ is orthogonal ($E^{-1}=E^T$), you can recover (an approximation of) $A$ from $F$ via $A = EF$.

> **⚠️ Important pitfall**: the dimensionality reduction (discarding small eigenvalues) always happens **in the NEW space**. This means we **cannot** identify which *original* features introduced redundancy — every eigenvector is a linear combination of *all* the original features; the eigenvectors are entirely new features living in the new space.
>
> PCA is fundamentally **UNSUPERVISED** — applicable to classification or clustering problems without needing any labels.

#### Attribute Subset Selection

An alternative to PCA that, unlike PCA, operates **directly on the original feature space**, following a heuristic search approach. Goal: reduce redundant and irrelevant attributes.
- **Redundant attributes**: duplicate the information already contained in one or more other attributes (e.g., purchase price and sales tax paid).
- **Irrelevant attributes**: contain no information useful for the data mining task at hand (e.g., student ID is usually irrelevant for predicting GPA).

**Search space**: for $d$ attributes there are $2^d$ possible attribute subsets — so a **greedy** heuristic approach is used (pick what looks best at each step, without exhaustive search).

**Typical heuristic methods:**
1. **Best single attribute** under the attribute-independence assumption: choose via significance tests.
2. **Best step-wise (forward) feature selection**: pick the best single attribute first; then the next-best attribute conditional on the first; and so on.
3. **Step-wise (backward) attribute elimination**: repeatedly eliminate the worst attribute.
4. **Best combined selection and elimination**.
5. **Optimal branch and bound**: uses attribute elimination plus backtracking.

Decision trees are a **direct method** for dimensionality reduction, since building the tree implicitly performs feature selection.

**Mutual Information (MI)-based feature selection** (example heuristic approach): the mutual information between two discrete variables $X,Y$ (with joint pmf $p(x,y)$ and marginals $p(x),p(y)$):

$$I(X,Y) = \sum_{x\in X}\sum_{y\in Y} p(x,y)\cdot \log\frac{p(x,y)}{p(x)\,p(y)}$$

- $I(X,Y)$ is **minimal** (its log argument = 1, so the term is 0) when the variables are **independent**.
- $I(X,Y)$ is **highest** when the variables are strongly **correlated**.
- Feature selection with MI: select the feature with the highest mutual information with the output.

**Digression — Information and Entropy**: suppose we transmit symbols from $\{a,c,e,g\}$ with known probabilities $\left<\frac18,\frac18,\frac14,\frac12\right>$.
- Using a **fixed** number of bits (2 bits, since there are 4 symbols) is wasteful.
- We can instead assign **fewer bits to more probable symbols** — the optimal code length for a symbol of probability $p$ is $-\log_2 p$ bits, giving an **average of 1.75 bits/symbol** in this example (vs. 2 bits fixed).
- **Entropy** of a distribution $D$ with probabilities $\langle p_1,\dots,p_n\rangle$:

$$H(D) = -\sum_i p_i \log_2 p_i$$

  - If we only need to transmit one symbol (single class, classification problem already solved), $H(D)=0$.
  - Entropy (also called "information") is **higher the closer the distribution is to uniform**.

**MI's two key properties**: (1) can measure *any* kind of relationship between variables (not just linear); (2) invariant under transformations that preserve the order of elements (translations, rotations).
- A limitation: feature selection based on raw MI is **extremely sensitive to the estimation of the probability density functions**.

**NMIFS (Normalized Mutual Information Feature Selection)** — addresses a redundancy problem: greedily picking the feature that always maximizes $I$ with the class tends to repeatedly pick similar/redundant features. Solution: introduce a criterion $G$ balancing relevance (MI with the class) against redundancy (average normalized MI with already-selected features):

$$NI(f_i,f_s) = \frac{I(f_i;f_s)}{\min\big(H(f_i),H(f_s)\big)}$$

$$G_i = I(C,f_i) - \frac{1}{|S|}\sum_{f_s \in S} NI(f_i;f_s)$$

- The normalization by $\min(H(f_i),H(f_s))$ compensates for MI's bias toward multi-valued features and restricts values to $[0,1]$.
- **Algorithm**: (1) initialize the candidate set $F$ with all $N$ features, selected set $S=\emptyset$; (2) compute $I(f_i;C)$ for each candidate feature; (3) select the single feature maximizing $I(f_i;C)$ as the first member of $S$; (4) iteratively: compute $I(f_i;f_s)$ between remaining candidates and already-selected features, then select the candidate maximizing $G$, moving it from $F$ to $S$, repeating until $|S|=k$ (a target number of features, fixed beforehand); (5) output $S$.
- This approach is **SUPERVISED** — it requires labeled instances (it uses $I(C,f_i)$, the MI with the class).

> **Recap — PCA vs. heuristic (attribute subset selection) approach**: PCA creates a new space and selects features (dimensions) within it; the heuristic approach selects features in the **original** space based on relevance to the output and redundancy with other selected features. Both methods only **reduce** the number of already-extracted features — neither *extracts new* features from the raw underlying system.

#### Attribute Creation (Feature Generation)
Creating new attributes can capture the important information in the dataset more effectively than the raw original attributes. Three general methodologies:
- **Attribute extraction**: domain-specific.
- **Mapping data to a new space** (this is itself a form of data reduction): e.g., Fourier transform, wavelet transform, manifold approaches.
- **Attribute construction**: e.g., combining features, data discretization.

### 4.2 Numerosity Reduction

Aim: reduce data volume by choosing alternative, smaller *representations* of the data, minimizing storage and computation while preserving analytical usefulness. Two families:

- **Parametric methods**: assume the data fits a specific model, estimate the model's parameters, and store **only the parameters** (discarding the raw data, except possibly outliers). Classic example: **linear regression**.
- **Non-parametric methods**: don't assume any model — includes histograms, clustering, sampling.

#### Regression Analysis (parametric)
A collective name for techniques modeling a **dependent/response variable** as a function of one or more **independent/explanatory variables (predictors)**. Parameters are estimated to give the "best fit," typically via the **least-squares** method. Used for prediction (including time-series forecasting), inference, hypothesis testing, and modeling causal relationships.

**Simple linear regression**: $y = w_1 x + w_0$. Two coefficients $w_1$ (slope) and $w_0$ (intercept) fully specify the line — we can discard the raw data and work with just these two numbers. Least-squares formulas:

$$w_1 = \frac{\sum_{i=1}^{D}(x_i-\bar{x})(y_i-\bar{y})}{\sum_{i=1}^{D}(x_i-\bar{x})^2}, \qquad w_0 = \bar{y} - w_1\bar{x}$$

*Worked example*: aptitude test scores ($x$) vs. statistics grades ($y$). Computed $w_1 = 470/730 = 0.644$, $w_0 = 77 - 0.644\times78 = 26.768$. Final regression line: $y = 0.644x + 26.768$.

**Multiple linear regression**: $y = w_0 + w_1x_1 + w_2x_2$ (extends to more predictors). Closed-form solution:

$$W = (X^TX)^{-1}X^TY$$

**Polynomial (non-linear) regression**: $y = w_0 + w_1x + w_2x^2 + w_3x^3$. Even though the relationship is non-linear in $x$, it's still **linear in the weights $w_i$** — so, by defining $x_1=x,\ x_2=x^2,\ x_3=x^3$, the equation becomes $y=w_0+w_1x_1+w_2x_2+w_3x_3$, solvable with the same multiple-regression machinery.

#### Histogram Analysis (non-parametric)
Divides data into **buckets/bins** and stores summary statistics (count, sum, average, …) per bucket instead of every individual value. Two partitioning rules (same as in Chapter 2, applied here for data reduction purposes):
- **Equal-width**: buckets all have the same numerical range.
- **Equal-frequency (equal-depth)**: each bucket contains the same number of observations.

#### Clustering (non-parametric)
Partitions the dataset into clusters based on similarity, storing **only the cluster representation** (e.g., centroid + diameter) instead of every instance.
- Very effective when data is naturally clustered (similar instances group together); **not** as effective if data is "smeared" (no clear grouping).
- Can be hierarchical, stored in multi-dimensional index tree structures. Many clustering definitions/algorithms exist (covered later in the course).

#### Sampling (non-parametric)
Obtaining a small sample $s$ representative of the whole dataset $N$, letting mining algorithms run in **sub-linear** complexity relative to data size.
- **Key principle**: choose a representative subset so as not to bias the final result.
- **Simple random sampling** can perform very poorly with **skewed data** — in these cases, adaptive methods like **stratified sampling** are preferable.
- **Note**: sampling may **not** reduce database I/O (data is often read page-at-a-time regardless).

**Types of sampling:**
- **Simple random sampling**: every item has equal probability of selection.
- **Sampling without replacement**: once selected, an object is removed from the population — subsequent draws are **not independent**; the covariance between draws relates to the population variance $\sigma^2$.
- **Sampling with replacement**: a selected object is **not** removed — draws are **independent**, and the covariance between the two draws is **zero**.
- **Stratified sampling**: partition the dataset into strata (groups), then draw samples proportionally from each — used with skewed data to make sure smaller/minority groups are still represented in the reduced dataset.
- **Cluster sampling** (contrast with stratified): the population is divided into clusters; **entire clusters** are randomly selected and **all** their elements are used — this differs from stratified sampling, where instead a portion of elements is drawn from *every* stratum/group.

> **Worked intuition (imbalanced classes)**: if 99% of instances belong to a majority class and 1% to a minority class, plain random sampling risks losing the minority class entirely from a reduced sample. **Stratified sampling** preserves the original class proportions, ensuring the minority class stays represented — and can be paired with sampling *with* replacement within each stratum.

#### Data Cube Aggregation
The **base cuboid** is the lowest (most detailed) level of a data cube — e.g., individual customer phone-call records in a phone-calling data warehouse. Data cube aggregation summarizes data at **multiple levels of abstraction**, producing values that represent groups rather than individual records, significantly cutting the data volume to process. Best practice: use the **smallest representation sufficient** to answer the task — answer analytical queries using aggregated cube data whenever possible, rather than raw detailed data, to reduce computational complexity while preserving the information needed.

### 4.3 Data Compression

Reduces storage requirements by encoding information more compactly.
- **String compression**: many well-established algorithms; typically **lossless** (original data perfectly reconstructible), but often only limited manipulation is possible without decompression/expansion first.
- **Audio/video compression**: typically **lossy** — approximates the original signal, discarding "less important" information. Even though information is lost, the result is usually good enough for practical use; some algorithms even allow reconstructing small fragments without decompressing the whole file. Example: **JPEG** (lossy), widely used for images — we usually don't perceive the information loss.
- **Time sequences** differ from audio: they're typically shorter and vary more slowly over time, so they occupy less storage.
- Dimensionality reduction and numerosity reduction can themselves be viewed as **forms of data compression**, since they also shrink dataset size while trying to preserve essential information.
- **Summary**: compression can be lossless (original always perfectly recoverable) or lossy (discarded information is deemed non-useful for the analysis; what remains still gives a good approximation of the original data).

---

## 5. Data Transformation and Data Discretization

**Data transformation**: a function that maps the entire set of values of an attribute into a **new** set of values, such that every original value corresponds to a transformed value. Purpose: prepare data for analysis by improving its structure/representation.

**Methods:**
- **Smoothing**: removes noise (see §2).
- **Attribute/feature construction**: build new attributes from existing ones to add information or improve algorithm performance.
- **Aggregation**: summarize data, e.g., via data cube construction.
- **Normalization**: scale attribute values to fall within a specified (smaller) range.
- **Discretization**: convert continuous data into discrete categories, often via concept hierarchy construction.

### 5.1 Normalization

Especially important when attributes have very different scales — many mining algorithms rely on distance computations (e.g., Euclidean), and attributes with larger numeric ranges would otherwise **dominate** the distance, even if not actually more important (e.g., annual salary would dominate age). Normalization makes all attributes contribute more equally.

**1) Min-max normalization** — linearly maps values into a new range $[new\_min_A, new\_max_A]$:

$$v' = \frac{v - min_A}{max_A - min_A}(new\_max_A - new\_min_A) + new\_min_A$$

*Example*: income range $[\$12{,}000, \$98{,}000]$ mapped to $[0.0, 1.0]$. A value of $\$73{,}600$ maps to:
$$\frac{73{,}600-12{,}000}{98{,}000-12{,}000}(1.0-0)+0 = 0.716$$

- This does **not** change the distribution's shape — it's similar to a **compression** operation.
- **⚠️ Highly sensitive to outliers**: an extreme value stretches the range and compresses the rest of the data into a very small interval, causing loss of distinction between values. Mitigations: remove/cap outliers, or switch to **z-score normalization**.

**2) Z-score normalization** — uses the attribute's mean $\mu_A$ and standard deviation $\sigma_A$:

$$v' = \frac{v - \mu_A}{\sigma_A}$$

*Example*: $\mu=54{,}000$, $\sigma=16{,}000$. A value of $73{,}600$ maps to $\frac{73{,}600-54{,}000}{16{,}000}=1.225$.

- The normalized value expresses **how many standard deviations** the raw value is above/below the mean.
- If $A$ has a **large** standard deviation, its normalized range gets shrunk/compressed; if $\sigma$ is **small**, the range gets stretched/enlarged.
- Unlike min-max (which forces all values into an identical, fixed range), z-score's range is *similar* across attributes but not forced identical — because outliers' effect is mediated by the mean/std computed over *all* the data, z-score is **less sensitive to outliers** than min-max.

**3) Normalization by decimal scaling** — divide by a power of 10 so values fall in $[-1,1]$:

$$v' = \frac{v}{10^j}$$

where $j$ is the smallest integer such that $Max(|v'|)<1$.

*Example*: recorded values of $A$ range from $-986$ to $917$; max absolute value $=986$. We divide by $1000$ (i.e., $j=3$): $-986 \to -0.986$, $917 \to 0.917$.

### 5.2 Discretization

Recall the three attribute types (Chapter 2): **Nominal** (unordered set, e.g., color/profession), **Ordinal** (ordered set, e.g., military/academic rank), **Numeric** (real numbers).

**Discretization** converts a continuous attribute into categorical/discrete values by dividing its range into intervals:
- Interval labels can replace the actual raw data values.
- Reduces data size — **not** a random process, since the chosen discretization impacts the final result.
- Can be **supervised** (uses class-label information) or **unsupervised**. In classification, we can discretize so that the resulting intervals correspond to classes.
- Can be **split (top-down)** — starts from the full data range and recursively divides it into smaller intervals based on some criterion (e.g., reducing variance, improving class separation) — or **merge (bottom-up)** — starts with many small intervals (often one per unique value) and progressively merges adjacent ones until a stopping condition, resulting in broader/more general intervals.
- Can be applied **recursively** (repeating the process on smaller and smaller parts of the data).
- Prepares data for further analysis, e.g., classification.

**Typical discretization methods** (all can be applied recursively):

| Method | Supervision | Direction |
|---|---|---|
| Binning | Unsupervised | Top-down split |
| Histogram analysis | Unsupervised | Top-down split |
| Clustering analysis | Unsupervised | Either (top-down split or bottom-up merge) |
| Decision-tree analysis | **Supervised** | Top-down split |
| Correlation analysis (e.g., χ²/ChiMerge) | **Supervised** | Bottom-up merge |

#### Binning
- **Equal-width (distance) partitioning**: divides the range into $N$ intervals of equal size (a uniform grid). If $A,B$ are the min/max values of the attribute, interval width $W=\frac{B-A}{N}$. Simple, but **outliers can dominate** the representation and it doesn't handle skewed distributions well.
- **Equal-depth (frequency) partitioning**: divides the range into $N$ intervals each with approximately the same number of samples — interval widths may vary, but this is **more robust to skewed data** and generally gives better "data scaling," since each bin represents a similar portion of the dataset.

*Worked example* — sorted prices: `4, 8, 9, 15, 21, 21, 24, 25, 26, 28, 29, 34`. Equal-frequency partitioning into 3 bins: `{4,8,9,15}`, `{21,21,24,25}`, `{26,28,29,34}`.
- **Smoothing by bin means**: replace each value with the bin's mean → `{9,9,9,9}`, `{23,23,23,23}`, `{29,29,29,29}`.
- **Smoothing by bin boundaries**: replace each value with the **closest boundary** (min or max) of its bin → `{4,4,4,15}`, `{21,21,25,25}`, `{26,26,26,34}` — this preserves the range of each bin while still reducing noise.

**Binning vs. clustering for discretization** (without using class labels): equal-width binning can cut the same underlying class across two different intervals; equal-frequency does somewhat better but still may mix multiple classes in one interval. **K-means clustering** discretization performs **best**, because it actually finds groups of genuinely similar objects and separates class members into different intervals more cleanly.

#### Discretization by Classification (Decision Trees) — supervised, top-down split
Uses class labels (e.g., cancerous vs. benign) to guide splitting. Uses **entropy** to evaluate the "purity" of a candidate split (a split is *pure* if all resulting instances belong to the same class). Process: repeatedly divide the dataset, at each step choosing the split that **maximizes class separation**, until a stopping condition (e.g., minimum sample count, or no further improvement in purity).

#### Discretization by Correlation Analysis — ChiMerge (χ²-based, supervised, bottom-up merge)
Uses the χ² statistical test to decide whether adjacent intervals should be **merged**, aiming to define intervals containing instances of the **same class**.

**Algorithm:**
1. **Sort the data** by the attribute to discretize (call it $F$), keeping the associated class labels $K$.
2. **Initialize intervals**: start with **every distinct value in its own interval** (as many intervals as unique values).
3. **Compute χ² for every pair of adjacent intervals**: build a small contingency table of class frequencies for the two intervals, compute observed vs. expected frequencies, then $\chi^2 = \sum\frac{(O-E)^2}{E}$.
   - $\chi^2 = 0$ → perfect independence between feature value and class in those two intervals → the intervals are statistically indistinguishable and carry no discriminative information → **merge candidates**.
4. **Find the pair of adjacent intervals with the smallest χ² value** (smallest χ² = most similar class distributions = best merge candidate).
5. **Merge** that pair of intervals into one.
6. **Repeat** (recursively/iteratively): recompute χ² for the new set of adjacent intervals, again pick the smallest, merge, and so on.
7. **Stopping condition**: stop when either (a) **all** χ² values are above a chosen threshold (read off the χ² distribution table for the relevant degrees of freedom), or (b) a **desired number of intervals** is reached.
   - χ² below threshold → intervals are statistically similar → merge.
   - χ² above threshold → intervals are significantly different → **do not** merge (keep separate).

**Final result**: a set of intervals where values within each interval share similar class distributions, while different intervals are statistically distinguishable from each other. A **pure interval** is one where (almost) all instances share the same class — modifying even a single value inside it won't change the class distribution meaningfully.

*Worked walkthrough (from the slides, 12 samples with feature $F$ and class $K$)*: starting from 12 singleton intervals, χ² values are computed for each adjacent pair; using a threshold of **2.7024** (from the χ² table, $df=1$, significance level $\alpha=0.1$: *merge if $\chi^2 < 2.7024$*), the algorithm iteratively merges the lowest-χ² adjacent pairs across several rounds — from 12 intervals down to a small number of intervals (in the example, it converges to 3 final intervals: `{0,10}`, `{10,30}`, `{30,60}`) once no remaining adjacent pair has $\chi^2$ below the threshold.

### 5.3 Concept Hierarchy Generation

A **concept hierarchy** organizes attribute values into hierarchical levels of abstraction, typically associated with each dimension in a data warehouse. It lets analysts operate at different levels of granularity — e.g., raw numeric age values replaced by higher-level categories like *youth*, *adult*, *senior*. This enables **drill-down** (go to more detail) and **roll-up** (go to a more general level) in multidimensional data analysis.

**For nominal data, hierarchies can be defined:**
- **Explicitly, as a total/partial ordering of attributes**, specified by users/domain experts at the schema level — e.g., `street < city < state < country`.
- **Explicitly, by grouping specific values** into higher-level categories — e.g., `{Urbana, Champaign, Chicago} < Illinois`.
- As a **partial** hierarchy — only some levels specified (e.g., only `street < city`, nothing above).
- **Automatically**, by analyzing the **number of distinct values** per attribute.

**Automatic generation** relies on the heuristic: attributes with **more distinct values** tend to belong to **lower** levels of the hierarchy (finer granularity), while attributes with fewer distinct values sit at **higher** levels. Example distinct-value counts: `country`: 15, `province/state`: 365, `city`: 3567, `street`: 674,339 → hierarchy `street < city < province/state < country`.
- **Exceptions exist**: some attributes don't follow this simple "more distinct values = lower level" pattern — e.g., **temporal attributes** like weekday, month, quarter, year (weekday has 7 distinct values but isn't "lower" than month with ~30-31, in the natural calendar hierarchy sense).

**For numeric data**, concept hierarchies are generated using the discretization methods described above (§5.2).

---

## 6. Summary (from the final slides)

- **Data quality**: accuracy, completeness, consistency, timeliness, believability, interpretability.
- **Data cleaning**: e.g., missing/noisy values, outliers.
- **Data integration** from multiple sources: entity identification problem, removing redundancies, detecting inconsistencies.
- **Data reduction**: dimensionality reduction, numerosity reduction, data compression.
- **Data transformation and discretization**: normalization, concept hierarchy generation.

---

## 7. Key points / potential exam pitfalls

- **Missing-data strategies get progressively "smarter"**: ignore tuple (crude, only viable with huge data) → global constant (risky, can invent a fake class) → attribute mean → attribute mean *per class* (smarter — uses label info) → inference-based most-probable value (Bayesian/decision tree — most sophisticated).
- **Smoothing introduces bias at peaks**: moving-average smoothing is unbiased only where the underlying function is (locally) linear; at a local maximum/minimum (nonzero second derivative), smoothing **always** shrinks the peak. This is why smoothing ratio matters: use ratio $<0.2$ if you need the true peak height/width; larger ratios are fine if you only need peak *position*.
- **Never smooth before least-squares fitting** — smoothing is always slightly lossy, distorts residual-based fit diagnostics, and causes serious underestimation of propagated parameter errors.
- **Correlation ≠ causation**: the classic pitfall example is hospitals vs. car thefts — both driven by a hidden third variable (population).
- **χ² test is for NOMINAL/categorical data only** — it operates on discrete counts in a contingency table; it cannot be applied directly to continuous numeric attributes without first discretizing them.
- **Pearson's r only captures LINEAR relationships** — two variables can be strongly (non-linearly) related and still show $r \approx 0$. Use Spearman for non-linear/monotonic relationships.
- **Zero covariance is necessary but NOT sufficient for independence** — some dependent variable pairs still have $Cov=0$; the implication only holds under extra assumptions (e.g., joint normality).
- **PCA reduction always happens in the transformed space** — you can never say "this original feature was redundant" after PCA, since every principal component mixes *all* original features together. PCA is unsupervised; attribute subset selection (heuristic) can be supervised and works in the *original* feature space instead — this is the key contrast to remember.
- **Min-max vs. z-score normalization — outlier sensitivity**: min-max is very outlier-sensitive (a single extreme value compresses everything else into a tiny sub-range); z-score is comparatively robust because outliers' influence is "diluted" through the mean/std computed over the whole attribute.
- **Equal-width vs. equal-depth (frequency) binning**: equal-width is simple but outlier-dominated and bad with skewed data; equal-depth balances the number of samples per bin and handles skew better — but at the cost of variable interval widths.
- **Binning/histograms (unsupervised) underperform clustering (e.g., k-means) for discretization aimed at separating classes** — plain binning can slice a single class across two intervals; k-means explicitly groups similar instances.
- **ChiMerge merge rule can feel counter-intuitive at first**: a **low** χ² between two adjacent intervals means their class distributions are *statistically indistinguishable* → that's precisely the signal to **merge** them (not to treat them as unrelated). A **high** χ² means the intervals are meaningfully different in class composition → **keep them separate**.
- **Sampling with vs. without replacement**: with replacement → draws are independent, covariance between any two draws = 0; without replacement → draws are dependent (removing an item changes the population), covariance relates to the population variance $\sigma^2$.
- **Stratified vs. cluster sampling**: stratified sampling draws *some* elements from *every* group/stratum (preserves proportions, crucial with imbalanced/skewed classes); cluster sampling instead randomly picks a subset of *entire* clusters and uses *all* their elements.
- **Sampling may not reduce database I/O**: since data is often read page-at-a-time regardless of how many records you ultimately sample from each page.
- **Data compression can be lossless (strings, generally) or lossy (audio/video, generally)** — dimensionality/numerosity reduction techniques are themselves conceptually a form of (lossy) data compression.
- **Concept hierarchy "most distinct values → lowest level" heuristic has exceptions** — temporal attributes (weekday/month/quarter/year) don't follow the naive distinct-value-count ordering, since their natural hierarchy is calendar-based, not count-based.

---

*File auto-generated by merging `3-Preprocessing.pdf` (professor's slides) and `3 - Preprocessing sbobine.pdf` (lecture notes). For questions about this chapter, refer only to this file.*
