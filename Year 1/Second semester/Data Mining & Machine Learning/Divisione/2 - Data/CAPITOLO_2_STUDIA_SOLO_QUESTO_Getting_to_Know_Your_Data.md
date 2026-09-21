# Chapter 2 — Getting to Know Your Data

> **Course:** Data Mining and Machine Learning / Bioinspired Computational Methods — Biological Data Mining
> **Instructor:** Prof. Francesco Marcelloni (University of Pisa)
> **Sources merged into this file:** professor's slides (`2-Data.pdf`, 35 pages, based on the Han–Kamber–Pei textbook) + lecture notes from L02, 2025-09-25 (`2 - Data Sbobine.pdf`, 11 pages)
>
> **Instructions for Claude:** this is the single reference file for Chapter 2. When the user asks study questions about this chapter, use **only** this file as context (no need to re-read the original PDFs unless a detail is missing). The file merges the official slide content with the lecture's discursive explanations, keeping both levels of detail.

---

## Chapter outline (official slide structure)
1. Data Objects and Attribute Types
2. Basic Statistical Descriptions of Data
3. Data Visualization
4. Measuring Data Similarity and Dissimilarity
5. Summary

---

## 1. Types of Data Sets

- **Record**
  - Relational records
  - Data matrix (e.g., numerical matrix, crosstab)
  - Document data: text documents → term-frequency vector
  - Transaction data (e.g., purchase transactions: TID + list of items)
- **Graph and network**
  - World Wide Web
  - Social or information networks
  - Molecular structures
- **Ordered**
  - Video data: sequence of images
  - Temporal data: time-series
  - Sequential data: transaction sequences
  - Genetic sequence data
- **Spatial, image and multimedia**
  - Spatial data: maps
  - Image data
  - Video data

**Cross-tab**: a table in matrix format that displays the (multivariate) frequency distribution of the variables. Classic example: a term-document matrix (rows = documents, columns = words/terms, cells = term frequency in the document).

### Important characteristics of structured data
- **Dimensionality** → linked to the *curse of dimensionality*: as the number of attributes increases, data becomes more "sparse" and classic distance metrics (e.g., Euclidean) lose meaning.
- **Sparsity**: a dataset is sparse if it contains a high number of null/zero values → "only presence counts".
- **Resolution**: refers to the scale of the dataset — patterns depend on the chosen scale.
- **Distribution**: centrality and dispersion of the values.

---

## 2. Data Objects and Attributes

### Data Objects
- A dataset is made up of **data objects**, each representing an entity.
- Synonyms: *samples, examples, instances, data points, objects, tuples*.
- Examples:
  - sales database → customers, store items, sales
  - medical database → patients, treatments
  - university database → students, professors, courses
- Tabular convention: **rows → data objects**, **columns → attributes**.

### Attributes
An **attribute** (also called dimension, feature, variable) is a data field representing a characteristic/feature of a data object (e.g., customer_ID, name, address…).

#### Attribute types

| Type | Description | Examples |
|---|---|---|
| **Nominal** | categories, states, "names of things"; no order | Hair_color = {auburn, black, blond, brown, grey, red, white}, marital status, occupation, ID numbers, zip codes |
| **Binary** | special case of nominal with only 2 states (0/1) | — |
| ↳ Symmetric binary | both outcomes equally important | gender |
| ↳ Asymmetric binary | outcomes NOT equally important → convention: assign 1 to the most important/rare outcome | medical test (positive/negative, e.g. HIV positive = 1) |
| **Ordinal** | values have a meaningful order (ranking), but the magnitude between successive values is not known | Size = {small, medium, large}, grades, army rankings |
| **Numeric (quantitative)** | quantitative values, integer or real-valued | — |
| ↳ Interval-scaled | measured on a scale of equal-sized units, has order, **NO true zero-point** (division makes no sense) | temperature in °C/°F, calendar dates |
| ↳ Ratio-scaled | has an **inherent/true zero-point** → can speak of values as "twice as much" | temperature in Kelvin, length, counts, monetary quantities |

> Note from the lecture: 10K° is twice as high as 5K° (ratio-scaled, true zero), whereas it makes no sense to say 20°C is "twice" 10°C (interval-scaled, arbitrary zero).

#### Discrete vs Continuous
- **Discrete attribute**: has only a finite or countably infinite set of values (e.g., zip codes, profession, or the set of words in a collection of documents). Sometimes represented as integer variables. Binary attributes are a special case of discrete attributes.
- **Continuous attribute**: has real numbers as attribute values (e.g., temperature, height, weight). In practice, real values can only be measured and represented using a finite number of digits (floating-point).

---

## 3. Basic Statistical Descriptions of Data

**Motivation**: better understand the data through central tendency, variation and spread (median, max, min, quantiles, outliers, variance, etc.). Data dispersion can be analyzed with multiple granularities of precision (boxplot or quantile analysis).

### 3.1 Measures of Central Tendency

**1) Mean — algebraic measure**

$$\bar{x} = \frac{1}{N}\sum_{i=1}^{N} x_i$$

- **Weighted arithmetic mean**:
$$\bar{x} = \frac{\sum_{i=1}^{N} w_i x_i}{\sum_{i=1}^{N} w_i}$$

- **Trimmed mean**: mean obtained by chopping out extreme values (e.g., the top and bottom 2% before computing the mean). Used to **reduce the bias introduced by outliers** in the definition of the mean.

**2) Median**
- If the number of values is odd → the middle value.
- If even → the average of the middle two values.
- It is a **holistic measure**: must be computed on the entire dataset as a whole. Computing it exactly would require continuously updating/sorting the dataset → **computationally expensive**.
- **Estimated by interpolation** (for grouped data):

$$\text{median} = L_1 + \left(\frac{N/2 - (\sum freq)_l}{freq_{median}}\right)\cdot width$$

where:
  - $L_1$ = lower boundary of the median interval
  - $(\sum freq)_l$ = sum of the frequencies of all intervals **lower** than the median interval
  - $freq_{median}$ = frequency of the median interval
  - $width$ = width of the interval

> Intuition (from the lecture): find the interval in which the median lies, assume a uniform distribution within it, and estimate the median as the midpoint of that interval. This interpolation is used only when an estimated exact value is needed; otherwise, continuously re-sorting the whole dataset is avoided.

**3) Mode**
- The value that occurs most frequently in the data.
- Can be unimodal, bimodal, trimodal (one, two, three peaks).
- **Empirical formula** relating mean, median, and mode:

$$mean - mode = 3 \cdot (mean - median)$$

### 3.2 Symmetric vs Skewed Data

- **Symmetric distribution**: mean ≈ median ≈ mode.
- **Positively skewed** (right tail): the mode occurs at a value **smaller** than the median → generally `mean > median > mode`.
- **Negatively skewed** (left tail): the mode occurs at a value **greater** than the median → generally `mean < median < mode`.

### 3.3 Measuring the Dispersion of Data

- **kth percentile**: the value $x_i$ such that k percent of the data entries lie at or below $x_i$.
- The **median is the 50th percentile**.
- **Quartiles**: $Q_1$ = 25th percentile, $Q_3$ = 75th percentile.
- **IQR (Inter-Quartile Range)**: $IQR = Q_3 - Q_1$ → represents the range containing the middle 50 percent of the data.

**Variance and standard deviation** (algebraic measures, distributive/scalable computation):

$$\sigma^2 = \frac{1}{N}\sum_{i=1}^{N}(x_i - \bar{x})^2 = \frac{1}{N}\sum_{i=1}^{N}x_i^2 - \bar{x}^2$$

$$\sigma = \sqrt{\sigma^2}$$

- Standard deviation measures how much values typically deviate from the mean.
- Called **algebraic** because they can be computed from distributive measures.

**Normal (Gaussian) distribution** — the 68-95-99.7 rule:
- $[\bar{x}-\sigma,\ \bar{x}+\sigma]$ → contains about **68%** of the measurements
- $[\bar{x}-2\sigma,\ \bar{x}+2\sigma]$ → about **95%**
- $[\bar{x}-3\sigma,\ \bar{x}+3\sigma]$ → about **99.7%**

### 3.4 Graphic Displays of Basic Statistical Descriptions

- **Boxplot**: graphic display of the *five-number summary* (min, Q1, median, Q3, max).
  - The box covers the IQR (from Q1 to Q3); the median is a line inside the box.
  - The *whiskers* extend from the box to the minimum and maximum (excluding outliers).
  - **Outliers** are points beyond a threshold, typically **1.5 × IQR** beyond Q1/Q3, plotted individually.
  - Limitation: it tries to describe the whole distribution using only 5 values → **two very different distributions can have the same boxplot** (see histograms below).

- **Histogram**: x-axis = intervals of values, y-axis = frequency. The frequency is given by the **area** of the bar (AUC), not the height — a crucial distinction from bar charts when the categories are not of uniform width.
  - **Bar chart** ≠ histogram: in a bar chart, bars represent discrete categories (not necessarily adjacent) and the height is proportional to the value.
  - Histograms and bar charts coincide **only** with equi-spaced intervals (in that case, height = area).
  - With the right granularity, histograms are **more informative** than boxplots: two different distributions can share the exact same boxplot but have very different histograms.

- **Quantile plot**: displays all of the data and the corresponding cumulative proportion $f_i$ (each point is a quantile) → allows the user to assess both the overall behavior and unusual occurrences. Quantiles 0.25, 0.5, 0.75 correspond respectively to $Q_1$, the median, and $Q_3$.

- **Quantile-Quantile (Q-Q) plot**: graphs the quantiles of **one** univariate distribution against the corresponding quantiles of **another**, to identify shifts/similarities between them.
  - Key application: **normality testing** → the data distribution is visually compared to a normal one (which appears as a straight line) to determine whether it is Gaussian.

- **Scatter plot**: provides a first look at bivariate data — each pair of values is treated as a pair of coordinates and plotted as points in the plane. Useful to see clusters, trends, outliers, correlations → used in *feature redundancy analysis* and *feature selection*.
  - **Positively correlated**: points tend to increase together.
  - **Negatively correlated**: one increases while the other decreases.
  - **Uncorrelated**: no clear pattern.

- **3-D Boxplot**: extension of the classic boxplot to three dimensions to compare multiple distributions/variables at once.

---

## 4. Data Visualization

**Why data visualization?**
- Gain insight into an information space by mapping data onto graphical primitives.
- Provide a qualitative overview of large data sets.
- Search for patterns, trends, structure, irregularities, relationships among data.
- Help find interesting regions and suitable parameters for further quantitative analysis.
- Provide a visual proof of computer representations derived from data.

### Categories of visualization techniques

**1) Pixel-oriented visualization techniques**
- For a dataset of $m$ dimensions, create $m$ windows on the screen (one per dimension), each with as many pixels as instances in the dataset.
- The color of each pixel reflects the value of the corresponding attribute.
- **Qualitative** evaluation: allows comparing multiple features per instance to see how they are correlated.

**2) Geometric projection visualization techniques**
- Use mathematical projections to visualize multidimensional data.
- **Scatterplot matrices**: pairwise scatter plots between all features in the dataset, arranged in a matrix — always **symmetric**. Very useful when the number of instances is not too high.
- **Parallel coordinates**: $n$ equidistant parallel axes, each corresponding to an attribute, scaled to the [minimum, maximum] range of that attribute. Every data object is represented by a **polygonal line** that intersects each axis at the point corresponding to its value.
  - If many lines overlap on the same attribute → probably **redundant information**, not useful for classification.
- **Projection pursuit**: helps users find meaningful projections of multidimensional data.

**3) Icon-based visualization techniques**
- Visualize data values as features of graphical icons.
- **Chernoff faces**: each element of the face represents a different feature (up to 10 changeable parameters: head eccentricity, eye size, eye spacing, eye eccentricity, pupil size, eyebrow slant, nose size, mouth shape, mouth size, mouth opening).
- **Stick figures**: a very simple type of drawing made of lines and dots, often of the human form or other animals. Two attributes are mapped to the display axes, and the remaining attributes are mapped to the angle and/or length of the limbs. Texture patterns in the visualization reveal certain data characteristics.
- General techniques: **shape coding** (use shape to encode information), **color icons** (use color to encode additional information).
- **Tile bars**: used to represent a text as a numeric vector (encoding). Key terms (term set/topic) are fixed, and the tile bar represents the frequency with which each word occurs in a specific section of the document.
  - **Rectangles** correspond to documents.
  - The query is specified in terms of $k$ topics (one topic per line → *term set*).
  - **Columns** in rectangles correspond to document segments.
  - A **square** corresponds to a specific term set in a specific text segment.
  - The **darkness** of a square indicates the frequency of terms in that segment for that term set.

**4) Hierarchical visualization techniques**
- Organize data using a hierarchical partitioning into subspaces: dimensional stacking, tree-map, cone trees, InfoCube.
- **Dimensional Stacking**: partitions the n-dimensional attribute space into 2-D subspaces which are "stacked" into each other. Attribute value ranges are divided into classes; the most important attributes should be used on the outer levels.
  - Adequate for data with **ordinal attributes of low cardinality**.
  - Becomes difficult to display more than about nine dimensions; important to map dimensions appropriately.
- **Tree-Map**: a screen-filling method that recursively partitions the display area into rectangular regions based on attribute values.
  - Each rectangle = a node in the hierarchy; its size is typically proportional to a specific attribute (e.g., file size).
  - Higher-level directories/categories contain smaller rectangles (subcategories/files) → area proportional to the size of the subtrees.
  - Color can encode additional information (e.g., file type).
  - Classic example: visualization of a file system.
- **Cone Trees** and **InfoCube**: InfoCube is a 3-D technique where hierarchical information is displayed as nested semi-transparent cubes; the outermost cube = top level of the hierarchy, inner cubes = lower-level nodes, and so on, explorable interactively.

### Visualizing complex, non-numerical data
- **Tag cloud (word cloud)**: displays words/tags from a set of documents or user-generated content; the importance of a tag is represented by font size/color (more frequent words appear larger/more prominent).
- **Social networks / relationships**: nodes represent entities (people, organizations, web pages), edges represent relationships or interactions. Useful for identifying clusters, communities, or influential nodes in complex datasets (text, relationships) where traditional numerical analysis alone may not reveal meaningful insights.

---

## 5. Measuring Data Similarity and Dissimilarity

### Basic concepts
- **Similarity**: numerical measure of how alike two data objects are. Higher value = more alike. Often falls in the range [0,1].
- **Dissimilarity** (e.g., distance): numerical measure of how different two data objects are. Lower value = more alike. Minimum is often 0, upper limit **varies** (not necessarily 1).
- **Proximity**: refers to either a similarity or a dissimilarity.

### Data Matrix vs Dissimilarity Matrix
- **Data matrix**: $n$ data points × $p$ dimensions (attributes) → a **two-mode** matrix (rows and columns represent different entities: objects vs. attributes).
- **Dissimilarity matrix**: $n \times n$, registers only the distances between pairs of objects.
  - It is a **triangular, symmetric** matrix, with all zero entries on the diagonal (distance of an object from itself).
  - It is **single mode** (rows and columns represent the same set of entities).
  - Can be used as input for classifiers: in that case, the rows of the matrix are treated as objects and the columns as features.

### 5.1 Proximity Measure for Nominal Attributes

Generalization of a binary attribute; can take 2 or more states (e.g., colors: red, yellow, blue, green).

**Method 1 — Simple matching**:

$$d(i,j) = \frac{p - m}{p}$$

where $m$ = number of matches, $p$ = total number of variables.

- **Minimum distance** (perfect match): $p = m \Rightarrow d(i,j) = 0$.
- **Maximum distance**: $m = 0 \Rightarrow d(i,j) = 1$ (no matches).
- Limitation: only sees whether attributes match or not, **cannot give them an order**.

*Example (from the slides)*: variables eye color and hair color.
$i = (\text{green}, \text{blond})$, $j = (\text{green}, \text{black})$ → $m=1$, $p=2$ → $d(i,j) = (2-1)/2 = 0.5$.

**Method 2 — Use a large number of binary attributes**: create a new binary attribute for each of the $M$ nominal states (e.g., for "color" create attributes red, yellow, blue, green, …), turning the nominal attribute into a set of binary (Boolean true/false) attributes.

### 5.2 Proximity Measure for Binary Attributes

Build a **contingency table** for two objects $i,j$:

| | j=1 | j=0 | total |
|---|---|---|---|
| **i=1** | q | r | q+r |
| **i=0** | s | t | s+t |
| total | q+s | r+t | p |

- **Symmetric binary** (both states equally valuable, e.g., gender):

$$d(i,j) = \frac{r+s}{q+r+s+t}$$

- **Asymmetric binary** (one state much more probable/important than the other, e.g., medical test): $t$ is **removed** from the denominator, because otherwise (probability of 0 being much larger than 1) $t$ would be huge and the distance would always come out artificially small:

$$d(i,j) = \frac{r+s}{q+r+s}$$

- **Jaccard coefficient** (*similarity* measure for asymmetric binary variables):

$$sim_{Jaccard}(i,j) = \frac{q}{q+r+s}$$

  equivalent to the definition of **coherence**.

*Numeric example (Jack, Mary, Jim)* — gender is symmetric, the remaining attributes are asymmetric binary (Y/P = 1, N = 0):

$$D(Jack, Mary) = \frac{0+1}{2+0+1} = 0.33$$
$$D(Jack, Jim) = \frac{1+1}{1+1+1} = 0.67$$
$$D(Jim, Mary) = \frac{1+2}{1+1+2} = 0.75$$

### 5.3 Standardizing Numeric Data

- **Z-score**:

$$z = \frac{x - \mu}{\sigma}$$

where $x$ = raw score to be standardized, $\mu$ = mean of the population, $\sigma$ = standard deviation. Expresses the distance between the raw score and the population mean in units of standard deviation (negative when below the mean, positive when above).

- **Mean Absolute Deviation (MAD)** — an alternative that is more **robust to outliers** than the standard deviation:

$$s_f = \frac{1}{N}\left(|x_{1f}-m_f| + |x_{2f}-m_f| + \dots + |x_{Nf}-m_f|\right)$$

$$z_{if} = \frac{x_{if} - m_f}{s_f}$$

### 5.4 Distances on Numeric Data — Minkowski Distance

$$d(i,j) = \left(\sum_{k=1}^{p} |x_{ik}-x_{jk}|^h\right)^{1/h}$$

where $i,j$ are two $p$-dimensional data objects and $h$ is the **order** (this distance is also called the $L_h$-norm).

**Properties of a metric**:
- $d(i,j) > 0$ if $i \ne j$, and $d(i,i)=0$ (positive definiteness)
- $d(i,j) = d(j,i)$ (symmetry)
- $d(i,j) \le d(i,k) + d(k,j)$ (triangle inequality)

**Special cases**:
- **$h=1$ → Manhattan distance** (city block, $L_1$-norm):
$$d(i,j) = |x_{i1}-x_{j1}| + |x_{i2}-x_{j2}| + \dots + |x_{ip}-x_{jp}|$$
  - Special case: **Hamming distance** = number of bits that are different between two binary vectors.
- **$h=2$ → Euclidean distance** ($L_2$-norm):
$$d(i,j) = \sqrt{|x_{i1}-x_{j1}|^2 + |x_{i2}-x_{j2}|^2 + \dots + |x_{ip}-x_{jp}|^2}$$
- **$h \to \infty$ → Supremum distance** ($L_{max}$-norm, a.k.a. Chebyshev): the **maximum** difference between any component (attribute) of the vectors.

*Numeric example (data matrix with 4 points, 2 attributes)*:

| point | attr1 | attr2 |
|---|---|---|
| x1 | 1 | 2 |
| x2 | 3 | 5 |
| x3 | 2 | 0 |
| x4 | 4 | 5 |

Dissimilarity matrix (Euclidean, $L_2$):

| | x1 | x2 | x3 | x4 |
|---|---|---|---|---|
| x1 | 0 | | | |
| x2 | 3.61 | 0 | | |
| x3 | 2.24 | 5.1 | 0 | |
| x4 | 4.24 | 1 | 5.39 | 0 |

(the slides also report the Manhattan $L_1$ and Supremum $L_\infty$ versions with the same computation structure)

### 5.5 Ordinal Variables

- Similar to a categorical variable but with a **clear order** (e.g., economic status = low/medium/high).
- Can be discrete or continuous.
- Handling: convert rank $r_{if} \in \{1,\dots,M_f\}$ into a normalized value in [0,1]:

$$z_{if} = \frac{r_{if} - 1}{M_f - 1}$$

- After normalization, it can be treated as interval-scaled and distances computed with standard methods.
- **Caution**: this transformation (from semantic logic to numerical logic) can **never** be done with unordered nominal variables → it would impose a non-existing order.

### 5.6 Attributes of Mixed Type

A database may contain a mix of nominal, symmetric binary, asymmetric binary, numeric, and ordinal attributes. Combine their effects with a **weighted formula**:

$$d(i,j) = \frac{\sum_{f=1}^{p} w_{ij}^{(f)}\, d_{ij}^{(f)}}{\sum_{f=1}^{p} w_{ij}^{(f)}}$$

- If $f$ is binary or nominal: $d_{ij}^{(f)} = 0$ if $x_{if}=x_{jf}$, otherwise $1$.
- If $f$ is numeric: use the normalized distance.
- If $f$ is ordinal: compute ranks and apply $z_{if} = (r_{if}-1)/(M_f-1)$, then treat as interval-scaled.

### 5.7 Cosine Similarity

Typically used for **document/text mining**: a document can be represented by thousands of attributes, each recording the frequency of a particular word/phrase (term-frequency vector). Other uses: gene features in micro-arrays, information retrieval, biologic taxonomy.

$$\cos(d_1, d_2) = \frac{d_1 \cdot d_2}{\lVert d_1 \rVert \, \lVert d_2 \rVert}$$

where $\cdot$ is the vector dot product and $\lVert d \rVert$ is the length (norm) of vector $d$.

- A value close to 1 → the documents are very similar.
- **Why not Euclidean**: with very high-dimensional vectors (curse of dimensionality), objects appear sparser as their dimension increases, making the Euclidean distance unreliable. Cosine similarity only considers the **angle** between the compared vectors, overcoming this problem.

*Numeric example (from the slides)*:
$$d_1 = (5,0,3,0,2,0,0,2,0,0)$$
$$d_2 = (3,0,2,0,1,1,0,1,0,1)$$
$$d_1 \cdot d_2 = 5\cdot3+3\cdot2+2\cdot1+2\cdot1 = 25$$
$$\lVert d_1\rVert = \sqrt{25+9+4+4} = \sqrt{42} = 6.481$$
$$\lVert d_2\rVert = \sqrt{9+4+1+1+1+1} = \sqrt{17} = 4.12$$
$$\cos(d_1,d_2) = \frac{25}{6.481 \times 4.12} \approx 0.94$$

---

## 6. Summary (from the final slides)

- Data attribute types: nominal, binary, ordinal, interval-scaled, ratio-scaled.
- Many types of data sets, e.g., numerical, text, graph, Web, image.
- Gain insight into the data by: basic statistical data description (central tendency, dispersion, graphical displays), data visualization (map data onto graphical primitives), measuring data similarity.
- These steps are **the beginning of data preprocessing** (later chapters).
- Many methods have been developed but it is still an active area of research.

---

## 7. Key points / potential exam pitfalls

- **Interval- vs Ratio-scaled**: the key difference is the **true zero-point**. Without a true zero (interval), speaking of "ratios" is meaningless (e.g., 20°C is not twice 10°C); with a true zero (ratio) it is (e.g., 10K is twice 5K).
- **Simple matching (nominal)** only gives binary match/non-match information: it **introduces no order** among values.
- **Asymmetric binary**: $t$ (co-absence) is excluded from the denominator because otherwise, since state "0" is much more frequent, the distance would be artificially always close to 0.
- **Jaccard coefficient = coherence**: it is a **similarity** measure, not a distance (ranges 0 to 1, higher = more similar), used on asymmetric binary attributes.
- **Median as a holistic measure**: expensive to compute exactly on dynamic datasets → interpolation is used as an estimate.
- **Histogram vs Boxplot**: two datasets with the exact same five-number summary (hence the same boxplot) can have very different distributions, visible only via the histogram.
- **Histogram vs Bar chart**: in a histogram, the bar's **area** (AUC) matters, not its height; they coincide only with equi-spaced intervals.
- **Curse of dimensionality**: as dimensions increase, data becomes sparser → distances like Euclidean lose meaning → for high-dimensional vectors (e.g., term-frequency) **cosine similarity** is preferred, since it only looks at the angle.
- **Transforming ordinal variables into numeric ones** (via normalized ranks) is legitimate; doing so with **unordered nominal** variables is conceptually wrong (it would impose an order that doesn't exist).
- **Dissimilarity matrix**: always triangular, symmetric, zero diagonal, single-mode (rows and columns = same set of objects), unlike the data matrix which is two-mode (objects × attributes).
- **Trimmed mean and MAD**: both designed to be more **robust to outliers** compared respectively to the plain mean and the standard deviation.

---

*File auto-generated by merging `2-Data.pdf` (professor's slides) and `2 - Data Sbobine.pdf` (L02 lecture notes, 2025-09-25). For questions about this chapter, refer only to this file.*
