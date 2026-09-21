# Chapter 13 — Explainable AI (XAI)

> **Course:** Data Mining and Machine Learning / Bioinspired Computational Methods — Biological Data Mining
> **Instructor:** Prof. Francesco Marcelloni (University of Pisa) — slide deck authored by **Pietro Ducange, Francesco Marcelloni, Fabrizio Ruffini** (AI Group, Dept. of Information Engineering)
> **Sources merged into this file:** professor's slides (`13-ExplainableIA.pdf`, 20 pages) + lecture notes (`13-ExplainableIA sbobine.pdf`, 11 pages — lesson **L12, 16/04/2026**)
>
> **⚠️ Note**: unlike most of the course, this chapter is **NOT based on Han–Kamber–Pei**. It is an original deck built on the XAI survey literature (Arrieta et al., Bodria et al., Ali et al., Molnar).
>
> **Instructions for Claude:** this is the single reference file for Chapter 13. When the user asks study questions about this chapter, use **only** this file as context (no need to re-read the original PDFs unless a detail is missing). The file merges the official slide content with the lecture's discursive explanations and worked examples, keeping both levels of detail.

---

## Chapter outline (from the "Content of the lecture" slide)
1. XAI models — what the field is and where it came from
2. *"Ok, my AI model works. But do I understand why and how?"*
   - Real-life scenarios
   - Stakeholders' motivations
3. Trustworthiness → Transparency → **Explainability**
   - **Interpretable-by-design models** (white and grey boxes)
   - **Post-hoc techniques** to explain opaque models' predictions
4. Post-hoc taxonomies (human resemblance / by dataset / global–local / agnostic–specific)
5. Feature-importance intuition, **SHAP**, **LIME**, **counterfactuals**
6. Focus on images: saliency maps
7. A final note on nomenclature — and on transparency as *source traceability*

---

## 1. The two core definitions

> **⚠️ These two definitions open the lecture and are the backbone of the whole chapter. Learn the exact wording.**

> **EXPLAINABILITY** is the **ability to provide explanations for a model's predictions, EVEN IF THE MODEL ITSELF IS COMPLEX** — hence the ability to explain **WHY, given a certain input, the model produced a specific output.**
>
> **INTERPRETABILITY** is the **ability to explain HOW decisions have been taken** — i.e. **the degree to which a human can DIRECTLY UNDERSTAND HOW A MODEL WORKS INTERNALLY.** It answers: *how does this model make decisions?*

> **The relationship between them** (lecture): with **explainability** we are **generally NOT able to fully describe the internal workings** of the model, but we can **provide insights into how and why it makes decisions** — for example **by identifying which features influence the predictions the most**, and **by explaining input–output relationships**.
>
> **⚠️ Of course, IF WE HAVE INTERPRETABILITY, THEN WE ALSO HAVE EXPLAINABILITY.** *(The implication goes one way only — this is the classic exam trap.)*

**Formal definition of explainability used in the deck** *(A. B. Arrieta et al.)*:
> **"Given a certain AUDIENCE, explainability refers to the details and reasons a model gives to make its functioning clear or easy to understand."**

> **⚠️ Note the word AUDIENCE**: explainability is **not an absolute property of a model** — it is relative to **who is asking**. The same model may be explainable to a data scientist and opaque to an end user.

**Three classes of models**, differing in **how much we understand about the internal workings of the system itself**: **white-box**, **grey-box**, **black-box**.

## 2. A little history

- The research field of **explainable artificial intelligence (XAI)** addresses explainability problems and methodologies — **probably first introduced by McCarthy in 1958** *("Programs with common sense")*.
- **Explainability at that time** was mostly related to the **requirements for a system "able to learn"**.
- **Now**: different definitions, similar nuances of meaning — **the underlying idea is to understand how a system works.**
- **The acronym XAI was first used in 2004 by van Lent et al.** *("An explainable artificial intelligence system for small-unit tactical behavior")*.

## 3. Why should I understand how my system works? (real-life scenarios)

| Domain | Application | Why explainability matters |
|---|---|---|
| **Healthcare** | **Diagnostic systems** | helps doctors **understand why a particular diagnosis or prediction was made**, improving trust and aiding decision-making |
| | **Drug discovery** | understanding how the model reaches conclusions about **potential drug compounds is vital for validation** and for optimizing research effort |
| **Finance** | **Credit scoring** | ensures **fairness and transparency**, letting individuals **understand why they were granted or denied credit** |
| | **Risk assessment** | helps comprehend **the factors influencing the assessment** |
| **Autonomous vehicles** | **Safety and decision-making** | understanding the reasoning is **crucial for safety and for building public trust** |
| **Criminal justice** | **Recidivism prediction** | crucial in predictive policing **to ensure fairness and prevent biases** in law enforcement and sentencing |
| **Ethical AI deployment** | **Bias mitigation** | transparent models enable **identification and mitigation of biases** (hiring processes, resource allocation) |

### The stakeholders and what each one gains

**Stakeholders**: **(domain) expert user, data scientist, end user, regulators, high-tech companies, lawyers, insurance companies.**

**What understanding AI buys you** (the summary slide):
- **Goodness of results** — **performance**, **bug fixing**, **augmenting results via expert-user domain knowledge**;
- **New opportunities** — private and public services, research fields;
- **Accountability**;
- **Increased trust**, and **winning over reluctance**;
- **Laws are respected**; **ethics and fundamental rights are respected**; **bias fixing**;
- **Increased interactivity → increased interaction → increased trust.**

### ⚠️ "Thinking out of the human"

> **It has been shown that AI-based systems are able not only to MIMIC the way humans reach a result, but also to CREATE METHODS UNLIKELY FOR HUMANS TO FOLLOW.**
> **AI is built by looking at human intelligence, but it is NOT identical to it.**
>
> **Wittgenstein's lion**: *"If an intelligent lion could talk, we would not understand him"* — **because it could perceive the world in a qualitatively different way with respect to humans.**

> **Why this matters for the exam**: it is the deepest argument for XAI. The problem is not merely that models are *complicated*; it is that a model may be solving the task **in a way that has no human analogue at all**, so we cannot recover its reasoning by introspection or analogy.

## 4. Trustworthy AI and regulation

> **Life-impacting decisions based on AI applications REQUIRE TRUST** ⇒ **regulators set rules to ensure wellbeing, preservation of fundamental rights, and societal development.**

**The world map of approaches:**

| Region | Orientation |
|---|---|
| **EU** | **commercial- AND fundamental-rights-oriented**, trying to be **at the forefront of legislation worldwide** |
| **USA** | **commercial- and rights-oriented**; **different laws for different states** |
| **China** | **commercial-oriented**; **focus on data privacy** |
| **Brazil** | *(also legislating)* |

### The EU path

- **April 2019**: the **high-level expert group on AI** set up by the European Commission presented the **"Ethics Guidelines for Trustworthy AI"**, according to which AI should be:
  - **LAWFUL** — respecting all applicable laws and regulations;
  - **ETHICAL** — respecting ethical principles and values;
  - **ROBUST** — both from a **technical** and a **social** perspective.
- **The AI Act** — the **EU's first law on AI**, based on the 2019 guidelines:
  - **April 2021**: first proposal;
  - **11 May 2023**: press release on the draft negotiating mandate;
  - **15 June 2023**: parliament vote for endorsement;
  - **often referred to as the first law on AI.**

> **Does it matter?** **Next-generation AI will have to comply with this type of legislation — a gradual transition and advance planning is advisable.**

**The expert group identified SEVEN key ethical requirements.** *(The lecture focuses on one of them: **TRANSPARENCY**.)*

### Transparency: three aspects

> 1. **COMMUNICATION** — **humans have the right to be informed that they are interacting with an AI system.**
> 2. **TRACEABILITY** — **data gathering/labelling and algorithms should be documented to the best possible standard.**
> 3. **EXPLAINABILITY** — **systems and decisions should be explained in a manner ADAPTED TO THE STAKEHOLDER CONCERNED**, with **different explainability levels for different kinds of model.**

**The four assessment questions on explainability** (from the EU Ethics Guidelines) — *did you assess:*
1. **to what extent the decisions, and hence the outcome, made by the AI system can be UNDERSTOOD?**
2. **to what degree the system's decision INFLUENCES the organisation's decision-making processes?**
3. **WHY was this particular system deployed in this specific area?**
4. **what the system's BUSINESS MODEL is** (for example, how does it create value for the organisation)?

## 5. The model spectrum: white, grey and black boxes

**The ordering used in the deck** (from Ali et al.), from most to least interpretable:

```
INTERPRETABLE BY DESIGN
   Linear models, Case-based reasoning models (KNN), Decision trees
   Fuzzy rule-based models, Bayesian networks
NEED POST-HOC TECHNIQUES
   (traditional) ANN-based models (also DNN), ensemble models (e.g. Random Forest)
```

### 5.1 White (glass) boxes

> **Those models that are easily interpretable for a human.** They are characterized by **COMPLETE TRANSPARENCY**, which allows us to **fully understand the inner workings of the model** — for this reason they are also called **GLASS BOXES**.
> **Interpretability here = the ability to explain HOW a prediction has been taken**, or to provide the meaning in a way understandable by a human.

| Model | Why it is interpretable | Caveats |
|---|---|---|
| **Linear / logistic regression** — $f(x) = a_0 + a_1x_1 + a_2x_2 + \dots + a_nx_n$ | **intuitively understandable: the larger the coefficient, the larger the impact of that feature on the output** | **models are not complex ⇒ LOW PERFORMANCE if the underlying phenomenon is complex** |
| **Case-based reasoning** | **humans intuitively look in the past for similar input situations and use past outputs for new inference** | typical methods involve **k-NN** (**computationally expensive**) or **defining PROTOTYPES** (*and: which is a good prototype?*) |
| **Decision trees** | **similar to human reasoning as a sequence of simple decisions in the form "if–then"** (logical models); **simple univariate thresholds**; **usually good performance** | **PRONE TO OVERFITTING** |

> **The lecture's addition on decision trees**: **each ROOT-TO-LEAF PATH corresponds to a specific rule that can be fully explained**, so we can understand **how the system is performing the inference process.**

### 5.2 Grey boxes

> **Those models that users can interpret TO SOME DEGREE IF THEY ARE CAREFULLY DESIGNED** (Ali et al.).
>
> **The lecture's framing**: grey-box models **lie between white boxes** (internal mechanisms fully known) **and black boxes** (inference process not accessible). **We have PARTIAL knowledge: some components are interpretable, others remain unclear** — so these models are **understandable only up to a certain level.**

**(a) Fuzzy rule-based systems (FRBS)** — **if–then rules**, where **linguistic rules and fuzziness help interpretability**:

```
If temperature is "low" AND humidity is "medium"
Then tourist_number = 0.25 + 0.4*temperature - 0.2*humidity      (or: tourist_number is "high")
```

> **⚠️ Where the "greyness" comes from** (lecture): **the general structure is based on well-defined if–then rules, but UNCERTAINTY is introduced in the definition of the linguistic variables** such as *"low"*, *"medium"*, *"high"*.
>
> **In fuzzy logic this uncertainty is handled through FUZZY SETS, which allow GRADUAL TRANSITIONS between categories.** Consequently **values can belong to multiple sets with different degrees of membership, leading to OVERLAPPING REGIONS.**
>
> **Why overlapping is a feature, not a bug**: **if these rules were handled with CLASSICAL (crisp) SETS, which do not allow overlapping regions, a BIAS OF PERCEPTION would be introduced into the model** — **due to the impossibility of giving an absolute definition of what is "low" or "high".**
>
> **Fuzzy sets were introduced by Lotfi A. Zadeh in 1965**, as a derivation of **multi-valued logic**.

**(b) Bayesian models** — usually take the form of a **probabilistic DIRECTED ACYCLIC graphical model whose links represent the CONDITIONAL DEPENDENCIES between a set of variables.** *For example, a Bayesian network could represent the probabilistic relationships between **diseases and symptoms**.*

### 5.3 Focus on decision trees and FRBS: how to EVALUATE interpretability

> **Both are "LOGICAL MODELS": both can be expressed by a set of "if–then" rules.**

**Two complementary evaluations:**

**1. QUANTITATIVE evaluation — complexity of a system as a PROXY for interpretability:**
- **number of rules**, **number of antecedents**, **number of coefficients** (for TSK-FRBS), *and many others.*

> **The lecture's phrasing**: **interpretability is INVERSELY related to model complexity** — **simpler models are easier to understand.**

**2. QUALITATIVE evaluation — SEMANTIC clarity:**
- *"for me, low temperature is from 0 → 5 °C; but for you?"*
- **⚠️ If I use many fuzzy sets, I can end up with many linguistic labels such as "very very very hot" ≠ "very very hot"… — is that really an interpretable system?**

> **The lecture's synthesis**: **using a LIMITED number of well-defined linguistic labels ("low", "medium", "high") improves understanding, whereas TOO MANY labels introduce ambiguity.**

**NOTE ON SPARSITY — Miller's law:**
> **Humans can (?) handle $7 \pm 2$ cognitive entities at the same time** [G. A. Miller, *"The magical number seven, plus or minus two"*, 1956; T. L. Saaty, 2003].
> ⇒ **"Less is okay(?)": sparse scenarios could be well handled by "not complex" logical models, allowing MAXIMUM interpretability.**
> **Sparse models with few rules and conditions are preferable, as they reduce COGNITIVE LOAD.**

### 5.4 Black (opaque) boxes

> **Those models that are NOT ESPECIALLY DESIGNED to be interpretable by humans — still, WE WANT THEM TO BE EXPLAINABLE.**

> **The lecture's description**: black-box models are those **whose internal workings are not accessible or interpretable**, so **we cannot clearly understand how they perform the inference process**. **Typical examples: NEURAL NETWORKS**, which — **despite their high predictive performance — lack transparency, especially when designed with a high number of neurons.**

**The general workflow for opaque models:**
> **Focus PRIMARILY on maximizing PERFORMANCE during design and training, WITHOUT explicitly considering interpretability. Then, ONCE THE MODEL HAS BEEN DEVELOPED, apply additional techniques to explain its behaviour.**
> **This leads to the concept of POST-HOC EXPLAINABILITY**, where **an auxiliary model or method is used to APPROXIMATE and interpret the behaviour of the original black-box model.**

**The scheme on the slide:**
```
Dataset X ──► Opaque model m ──► prediction  ŷ = m(x) ──► USER
                    │                                       ▲
                    └──► Explanator model e ──► expl = e(m, x)
                         (explanation given an input and a model)
```

> **NOTE on the slide**: **nomenclature is not universally accepted — we are in the era of "naming" things.**

> **⚠️ And the requirement that governs everything**: when applying post-hoc explainability, **the interpretation must be clear and given in such a way that DIFFERENT USERS/STAKEHOLDERS understand why the model is behaving in a specific way.** This is why **there are different types of explainability and different post-hoc methods**, handling different data types.

### 5.5 ⚠️ The accuracy–interpretability trade-off

> **A CENTRAL PROBLEM of explainable AI is the INVERSE RELATIONSHIP between ACCURACY and INTERPRETABILITY/EXPLAINABILITY.**
> **To have high accuracy it is necessary to INCREASE THE COMPLEXITY of the models, which are consequently LESS easily interpretable/explainable.**
> **Hence: if we want high accuracy, we have to PENALIZE explainability — and vice versa.**
>
> **Where post-hoc methods live**: they **are generally applied to quite complex models — in the region of ENSEMBLES and SVM methods — that have low explainability and interpretability.**

## 6. How do we achieve explainability? Two routes and four taxonomies

> **How to achieve explainability?**
> 1. **DESIGN OF INHERENTLY INTERPRETABLE MODELS** — *white and grey boxes* (§5.1–5.3);
> 2. **POST-HOC EXPLAINABILITY TECHNIQUES for black boxes (opaque models)**, which have higher levels of complexity.

**The taxonomy of post-hoc methods has FOUR points of view** *(Bodria et al., "Benchmarking and survey of explanation methods for black box models")*:
1. **By resemblance with the human approach**
2. **By dataset (data type)**
3. **Global / local**
4. **Model agnostic / model specific**

### 6.0 The broader taxonomy (from the lecture)

**Dimension 1 — the PURPOSE of interpretability:**
- **to create WHITE BOXES**, i.e. intrinsically interpretable models;
- **to EXPLAIN BLACK BOXES** and complex models using post-hoc methods;
- **to enhance the FAIRNESS of a model**;
- **to TEST THE SENSITIVITY of predictions** — *if the model is interpretable, we are also able to test how sensitive the model is in producing the desired output.*

**Dimension 2 — model specific vs. model agnostic** (see §6.3).

### 6.1 Post-hoc methods resembling the human approach

**Schematically** *(Arrieta et al.)*:
- **Text explanations**
- **Visual explanations** *(e.g. highlighting relevant regions in images)*
- **Local explanations** *(focused on individual predictions)*
- **Explanations by example** *(predictions justified by similarity to known instances)*
- **Explanations by simplification → SURROGATE MODELS**
- **Feature relevance explanations**
- *… (more incoming)*

> **On SURROGATE MODELS** (lecture): a **complex model is approximated by a simpler, more interpretable one** in order to explain how a specific output is obtained from a given input. **The most famous example is the DECISION TREE**, since it is **inherently interpretable and lets us clearly trace the mapping from input to output.** In post-hoc use, **we build a decision tree to simplify and explain a more complex, black-box model.**
> **⚠️ Since the surrogate is a SIMPLIFICATION of the original model, it can only provide an APPROXIMATE representation of its behaviour.**

> **On FEATURE RELEVANCE** (lecture): it identifies **which variables most influence the model's decisions**, playing a key role **not only in interpretability but also in FEATURE SELECTION**. It is **especially used with TABULAR data**: **if a feature is particularly relevant, a variation of its value leads to a proportional variation in the performance of the model.**
> **⚠️ But these explanations are ONLY APPROXIMATIONS and do NOT fully capture the true internal workings of the model.**

### 6.2 Post-hoc methods by dataset (data type)

> **Explainability methods must be ADAPTED to the type of dataset, as different data modalities require different explanation strategies.** The four/five major data types: **tabular, text, image, graph** *(plus time series)*.

| Data type | Goal of the explanation | Typical techniques |
|---|---|---|
| **Tabular** | **which VARIABLES have the greatest influence on the output** | **feature relevance/importance**; **RULE-BASED approaches** — interpretable if–then rules extracted to describe the model's behaviour |
| **Image** | **highlight the REGIONS or PIXELS that contribute most to a prediction** | **SALIENCY MAPS** and related visualizations *(e.g. identifying pathological regions — nodules — in CT scans)*; **CONCEPT ATTRIBUTION**, associating a concept with the image *(e.g. classifying zebras based on the concept of having stripes)* |
| **Text** | **identify the most relevant WORDS or TOKENS influencing the decision** | **sentence highlighting** (contribution of each word); **attention-based representations** — a matrix of scores revealing how words relate to each other |
| **Time series** | **highlight the most informative SEGMENTS of the signal** | **series highlighting** (a score per point, based on its contribution); **temporal attention** mechanisms |
| **Graph** | **identify the most relevant COMPONENTS of the graph** | **node highlighting** (importance scores to nodes) or **edge highlighting** (to edges) |

**Example-based approaches** (cutting across data types):
- **PROTOTYPES** — **representative examples that characterize a class learned by the model**, allowing the user to understand predictions **by comparing them with typical instances**;
- **COUNTERFACTUAL EXPLANATIONS** — **alternative examples very similar to the input but leading to a DIFFERENT prediction**, highlighting **the minimal changes required to alter the decision** and thereby **providing an implicit indication of the DECISION BOUNDARY between classes.**

> **The principle is always the same** (lecture): **provide a clear explanation to the end users/stakeholders, and communicate how the models arrive at the prediction.**

### 6.3 Model agnostic vs. model specific

| | **Model SPECIFIC** | **Model AGNOSTIC** |
|---|---|---|
| **Definition** | **meant to explain a SPECIFIC AI model** (or a specific group of models) | **NOT tied to any specific AI model** |
| **PROS** | **tailored for a specific model** ⇒ **more precise and detailed** explanations | **high FLEXIBILITY**; easily generalizable to all model typologies |
| **CONS** | **cannot be applied to all models**; loses generalizability | **possibly HIGH COMPUTATIONAL EFFORT**; tends to have lower performance, **or needs more computation to reach the same level of explainability** |

### 6.4 Global vs. local — for BOTH post-hoc and inherently interpretable models

> **⚠️ This slide makes a distinction students routinely blur: the global/local axis applies to explainABILITY (post-hoc) AND to interpretABILITY (by design), but it means different things in each case.**

**Post-hoc EXPLAINABILITY:**
- **LOCAL explanators**: **explain INDIVIDUAL PREDICTIONS of the AI model** — considering a test set, we try to explain **every single instance**;
- **GLOBAL explanators**: **provide global insights into the ENTIRE model**;
- **⚠️ Global explanators OFTEN COME FROM THE AGGREGATION OF LOCAL EXPLANATIONS.**

> **The standard workflow** (lecture): **focus on the explainability of the single instances, then give a global comprehension of the overall model behaviour by AGGREGATING the single instances together.**

**Inherently INTERPRETABLE models:**
- **LOCAL interpretability**: **refers to how the model is defined for a SPECIFIC REGION** — e.g. **for a rule-based system, the ACTIVATED regions and their rules**; **for a decision tree, the ROOT-TO-LEAF PATH** containing the rule that locally explains the prediction;
- **GLOBAL interpretability**: **refers to the STRUCTURE of the model** ⇒ **a COMPLEXITY notion** (e.g. the **number of rules extracted from a decision tree**).

> **The lecture's sharpening**: **the higher the complexity of the model — the larger its size (e.g. the more roots and leaves a decision tree has) — the LOWER the interpretability**, because such a large structure becomes harder and harder to explain.
> **⚠️ So for a decision tree the SINGLE PATH remains interpretable (local), while GLOBAL interpretability depends on the total number of paths/rules and the number of antecedents per rule.**

## 7. Post-hoc feature-importance: the basic intuition

> **The intuition**: given a dataset with $j = 1, \dots, J$ features and a learned model $f$, compute a **score metric** $s$ of $f$ (e.g. **accuracy, MAE**). **To evaluate the importance of a single feature $j$, see how the metric changes while the feature value changes. If the score changes a lot, that feature is important.**

**The procedure — PERMUTATION IMPORTANCE:**
1. **For each feature $j$, randomly SHUFFLE its values $K$ times** ($k = 1, 2, \dots, K$) ⇒ we obtain a **"corrupted version of the data"**;
2. **compute the score on the corrupted data**, $s(k,j)$;
3. **compute the importance of feature $j$ as**
$$I(j) \;=\; s \;-\; \frac{1}{K}\sum_{k=1}^{K} s(k,j)$$

> **What the shuffling does** (lecture): it **BREAKS THE RELATIONSHIP between that feature and the target variable**. **If performance significantly degrades, the feature plays an important role; if performance remains unchanged, the feature has little or no importance.** Repeating it $K$ times and averaging gives a **reliable estimate**.

> **⚠️ The question asked on the slide — and its answer:**
> *"If your feature is CONSTANT — the values are all the same — what is the importance $I(j)$ of this feature?"*
> **⇒ It would be $= 0$… NO IMPORTANCE.** *(Shuffling identical values changes nothing, so the score does not move.)*

**This intuition is the basis of the most popular post-hoc method: SHAP.**

## 8. SHAP — SHapley Additive exPlanations

> **SHAP** *(Lundberg et al., "A unified approach to interpreting model predictions", 2017)* **is a post-hoc method for estimating the SHAPLEY VALUES** *(introduced by **Lloyd Shapley**, Nobel Memorial Prize in Economics, 2012)*.

### 8.1 The intuition: cooperative game theory

> **The soccer team analogy.** Imagine a soccer team wins a match **3–0**. **How much did each player contribute to the victory?**
> **NOT ONLY THE GOAL SCORER MATTERS: defenders, midfielders and pass creators all play a role.**

**The dictionary:**

| Game theory | XAI |
|---|---|
| **The team** | **the model** |
| **The players** | **the features** |
| **The victory / profit** | **the model prediction** |

**Shapley values consider:**
- **ALL possible orders (COALITIONS) in which players (features) are added**;
- **the CHANGE in team performance (prediction) with each addition**;
- **the AVERAGE contribution of each player across all combinations.**

> **⇒ It fairly attributes the prediction to each feature, just like crediting players in a team win.**

### 8.2 From game theory to XAI: how a coalition is evaluated

> **Replace players with FEATURES; replace profit with the MODEL PREDICTION. Shapley values quantify the impact of each feature on the model prediction.**
>
> **To evaluate each coalition, we:**
> - **keep ONLY the selected features from the instance $x$**;
> - **REPLACE the other features using values from a BACKGROUND (reference) DATASET**;
> - **this SIMULATES what the model would predict WITHOUT ACCESS to certain features.**

**The additive decomposition** (lecture): **the prediction for a given instance can be expressed as the SUM OF A BASE VALUE AND THE CONTRIBUTIONS OF ALL FEATURES.**

> - **The BASE VALUE represents the AVERAGE MODEL OUTPUT over the background (reference) dataset;**
> - **each SHAP value quantifies HOW MUCH A SPECIFIC FEATURE SHIFTS THE PREDICTION FROM THIS BASELINE.**
>
> **⚠️ On the background dataset** (lecture): it is **typically a SUBSET OF THE TRAINING SET** and can be constructed **through SAMPLING or CLUSTERING techniques (e.g. using cluster prototypes)** in order to **reduce computational complexity and the number of instances considered.**

### 8.3 The worked example: apartment prices

> **You have trained an AI model to predict apartment prices. For an individual instance it predicts €300,000 and you need to explain this prediction.**
> **The apartment has an AREA of 50 m², is on the 2nd FLOOR, has a PARK NEARBY and CATS ARE BANNED.**
> **The average prediction for all apartments is €310,000.**
>
> **Question: how much has each feature value contributed to the prediction COMPARED TO THE AVERAGE PREDICTION?**

**Definition applied**: **the Shapley value is the AVERAGE MARGINAL CONTRIBUTION of a feature value across ALL POSSIBLE COALITIONS** (here a coalition = a group of players = features).

**Computing the Shapley value of the feature `cat_banned`:**

| | Coalition | Model prediction |
|---|---|---|
| **Coalition 1** | same as the instance to be predicted, **except 1st floor → 2nd floor** | **€310,000** |
| **Coalition 2** | same as the instance, **except 1st floor → 2nd floor AND cat is ALLOWED** | **€320,000** |

$$\text{marginal contribution of } \texttt{cat\_allowed} \;=\; 320{,}000 - 310{,}000 \;=\; \textbf{€10,000}$$

> **We repeat this computation for ALL possible coalitions → then we take the AVERAGE.** *(Molnar, "Interpretable Machine Learning".)*

> **⚠️ What a Shapley value tells you** (lecture): **both the MAGNITUDE and the DIRECTION (positive or negative) of that feature's contribution to the final prediction.** **In general, the higher the ABSOLUTE VALUE of the Shapley value, the greater the impact of the feature on the final prediction.**

### 8.4 The SHAP package — tabular example (breast cancer)

*(0 = diseased, 1 = healthy)*

```python
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import shap

seed = 42
cancer = load_breast_cancer()
X = pd.DataFrame(cancer.data, columns=cancer.feature_names)
y = pd.DataFrame(cancer.target, columns=['target'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)

rf_classifier = RandomForestClassifier(random_state=seed)
rf_classifier.fit(X_train, y_train.values.ravel())

# Explain the model's predictions using SHAP
explainer = shap.TreeExplainer(rf_classifier)
shap_values = explainer.shap_values(X_test)     # SHAP values for the entire test set
```
*Installation: `pip install shap` or `conda install -c conda-forge shap` — docs at shap.readthedocs.io*

**The two plots, and what each is for:**

| Plot | Code | Scope |
|---|---|---|
| **Force plot** — **single prediction explanation** | `shap.force_plot(explainer.expected_value[1], shap_values[1][0,:], X_test.iloc[0,:])` | **LOCAL explanation** |
| **Beeswarm / summary plot** — **many single-prediction explanations aggregated** | `shap.summary_plot(shap_values[1], X_test, show=False)` | **GLOBAL overview (global explanation)** |

> **⚠️ Note the pattern**: the global view is literally **the aggregation of many local views** — exactly the principle stated in §6.4.

### 8.5 ⚠️ How to READ a beeswarm (summary) plot

This is the most exam-likely practical skill in the chapter. From the lecture:

- **the $x$-axis represents the SHAPLEY VALUES**, indicating each feature's contribution to the prediction, **both positively and negatively**;
- **the COLOUR encodes the FEATURE VALUE** — **RED for high magnitudes, BLUE for low ones**;
- **each POINT corresponds to a specific INSTANCE.**

**Two contrasting readings given in class:**

| Pattern | Interpretation |
|---|---|
| **First feature**: **low feature values (blue) ⇒ HIGH Shapley values; high feature values (red) ⇒ LOW Shapley values** | **the feature has a STRONG impact on the output** — a variation in the feature value corresponds to a variation in the prediction: **a clear, monotonic (here inverse) relationship** |
| **Second feature**: **HIGH Shapley values for BOTH high and low feature values** | **the feature still significantly affects the output** (high variability in Shapley values), **BUT the absence of a clear separation between red and blue points suggests NO SIMPLE OR MONOTONIC relationship** between the feature value and its impact |

> **General rule**: **features with large positive or negative SHAP values have a strong influence**, while **MIXED COLOUR distributions indicate more complex, NON-LINEAR relationships.**

### 8.6 SHAP in one paragraph (the lecture's own summary)

> **SHAP is based on cooperative game theory, where features are interpreted as PLAYERS contributing to a final outcome, which corresponds to the MODEL PREDICTION. The contribution of each feature is computed by considering ALL POSSIBLE COALITIONS and evaluating how the prediction changes when a feature is added. This is typically done using a BACKGROUND (reference) dataset, which simulates the absence of certain features. In this way, SHAP provides a FAIR ATTRIBUTION of the prediction to each feature, capturing both the MAGNITUDE and the DIRECTION of their impact.**

## 9. LIME — Local Interpretable Model-agnostic Explanations

> **We provide a SIMPLER MODEL trying to MIMIC THE MODEL LOCALLY.**
> **Instead of explaining the entire model, LIME explains the prediction for a SPECIFIC INSTANCE by approximating the model locally with a simpler, interpretable model — the SURROGATE MODEL.**
> **The key idea: IN A SMALL REGION AROUND THE INSTANCE, THE COMPLEX MODEL CAN BE APPROXIMATED BY A SIMPLER ONE.**

**The four steps:**
1. **Consider the model $m$ (usually opaque) we want to explain, and an input instance $I$.**
2. **Create PERTURBED SAMPLES ("fictitious points") CLOSE to the instance $I$.** On this **synthetic** sample, **use the model $m$ to get the predictions.**
3. **Train a SIMPLER model, interpretable by design, on the couple: synthetic inputs → model $m$'s predictions** — **with a WEIGHT BASED ON THE DISTANCE from $I$.**
4. **The simpler model (the SURROGATE), being interpretable by design (a linear model or a decision tree), is used to explain the model $m$ LOCALLY.**

### PROS and CONS

**PRO: EASY TO UNDERSTAND** — both in its application and in the explanation it provides.

**CONS:**
- **Synthetic data generation**: **if not careful, it can create "not accurate" local models.**
- **The notion of DISTANCE**: **how to choose it? How do we decide when to STOP generating synthetic data?** *(The lecture calls this the major difficulty: defining a distance around the input instance and setting how many synthetic samples to generate.)*
- **It is a LOCAL model** — **you could "patch" many local models together… but is that feasible?** *(Different instances require different local models.)*
- **⚠️ DOUBLE ERROR**: when evaluating the performance of the full strategy **you must take into account the errors related to model $m$ AND the errors related to the surrogate model.**

> **⚠️ The consequence that is easy to get wrong** (lecture): **the quality of the surrogate is NOT assessed on its accuracy with respect to the true ground truth, but on HOW WELL IT APPROXIMATES THE BEHAVIOUR OF THE ORIGINAL MODEL.** **The goal is to measure the FIDELITY of the surrogate — how closely its predictions match those of the model being explained.**

*Package: `pip install lime` — github.com/marcotcr/lime*

## 10. Counterfactual explanations

> **The main idea: determine the MINIMAL CHANGES required in the input to obtain a DIFFERENT output.**
> **Given the output $y_0$ and the corresponding input $x_0$, what should be different in $x_0$ to get another output $y_1$? — in general, we are very interested in the SMALLEST change in $x_0$.**

**The exam example from the slides:**

> *You are taking an exam, but you do not pass.* **Intuitive consequent questions: → Why? → What change should have been made so that my request would have been accepted?**
>
> **Student $x_0$**: *I studied **20 hours** + read **2 books** + answered last year **10 questions** + **99 aperitifs**.*
>
> **Student $x_0$ + some small changes**: *I studied **40,000 hours** + read **10 books** + answered last year **20 questions** + went to **2 aperitifs**.*

> **What the method is looking for** (lecture): **the minimal change of these variables that leads to a change in the predicted output class** — here **from not passing the exam to actually passing it.**

**Why counterfactuals are valuable:**
- **they are ACTIONABLE and align well with human reasoning**, answering the question **"What should be different to change the outcome?"**;
- **⚠️ the explanation is quite INTUITIVE, and EASIER TO UNDERSTAND than Shapley values, which might be difficult for an untrained user;**
- they **implicitly indicate the DECISION BOUNDARY between classes** (§6.2).

**Python packages**: **Alibi** (`docs.seldon.io/projects/alibi`), **MACE** (`github.com/charmlab/mace`), **DiCE** (`github.com/interpretml/DiCE`).

## 11. Focus on images: saliency maps

> **A SALIENCY MAP (SM) is an image in which a pixel's BRIGHTNESS represents HOW SALIENT (important) the pixel is for the model decision.**
> **In a classification task, a POSITIVE value means the pixel contributed POSITIVELY to the classification, while a NEGATIVE one means it contributed NEGATIVELY.**
> **Intuition from the plots: RED = positive, BLUE = negative** *(and, per the lecture, **GREY pixels highlight the zones that do NOT contribute** to the classification).*

> **⚠️ The conceptual bridge**: **saliency maps for images are based on the SAME INTUITION of "feature importance" used for tabular and text data** — **pixels (or groups of pixels) simply play the role of features.**

**How the two main methods adapt to images:**

| Method | Adaptation | Nature |
|---|---|---|
| **LIME** | **segments the image into SUPERPIXELS** (groups of pixels / segments of the image) and evaluates which regions are most relevant | **local, post-hoc, MODEL-AGNOSTIC explainer** |
| **SHAP** (in the **GRAD-SHAP** variant) | **produces a saliency map FOR EACH CLASS in the input** | **can explain SIMULTANEOUSLY DIFFERENT LABELS**, allowing a class-specific interpretation |

## 12. A final note — nomenclature, and the OTHER meaning of transparency

### 12.1 Nomenclature is not fixed

> **In the literature (and in the public debate) you will find that the nomenclature is NOT YET FIXED.** Terms such as **interpretability, explainability, transparency** — and, per the lecture, also **intelligibility and comprehensibility** — **are sometimes used as SYNONYMS.**
> **We have to wait for the community to reach "nomenclature maturity".**

### 12.2 ⚠️ Transparency has TWO meanings

This closing part comes from the lecture and is not on the slides — **it is the most "current" material in the chapter.**

> **(a) Transparency OF THE MODEL** — closely related to explainability. **A transparent model is one able to explain how it produces an output from a given input, ideally in a way understandable to humans.** This corresponds to the **model interpretability/explainability** discussed for white boxes, grey boxes and post-hoc methods.
>
> **(b) Transparency as TRACEABILITY OF THE SOURCES of information used by the model** — a **legal and practical** perspective. **This has become particularly relevant with the widespread use of GENERATIVE AI systems**, capable of producing text, images and other content. **In these cases it is NOT SUFFICIENT to obtain an output: it is also important to understand WHERE THE INFORMATION COMES FROM and whether it is RELIABLE.**

**Why (b) matters — HALLUCINATIONS:**
- **Generative AI systems may produce HALLUCINATIONS**, i.e. **outputs that are PLAUSIBLE but INCORRECT or unsupported by real sources.**
- **To ensure trustworthiness it is essential to VERIFY the information**, especially in **scientific and legal contexts where accuracy is critical.**
- **A well-known example: a legal case in which a lawyer relied on AI-generated content that included FABRICATED REFERENCES, leading to serious consequences in court.**
- **⚠️ Even when generative models PROVIDE references, these may not be accurate or correctly interpreted.** In many cases **the generated content does not faithfully reflect the original sources**, particularly with **non-public or restricted-access material.**
- **Studies have shown that modern AI systems introduce a NON-NEGLIGIBLE level of hallucination, even in tasks such as SUMMARIZATION.**
- **⚠️ Increasing model complexity does NOT necessarily eliminate this issue — and may in some cases EXACERBATE it.**

> **The conclusion**: **transparency in the sense of SOURCE TRACEABILITY and RELIABILITY is a key requirement, explicitly emphasized in regulations such as the European AI Act.** **More generally, users must adopt a CRITICAL approach when interacting with AI systems, treating their outputs as POTENTIALLY USEFUL BUT NOT INHERENTLY TRUSTWORTHY, and always verifying the underlying sources when accuracy is required.**

---

## Key points / potential exam pitfalls

### The two definitions
- **⚠️ EXPLAINABILITY answers "WHY this output for this input" (works even for complex models); INTERPRETABILITY answers "HOW does the model work internally".** Do not swap them.
- **Interpretability ⇒ explainability, but NOT vice versa.** This one-way implication is stated explicitly in the notes.
- **Explainability is relative to an AUDIENCE** (Arrieta's definition) — it is not an intrinsic property of the model.
- **The nomenclature is NOT standardized**: interpretability / explainability / transparency / intelligibility / comprehensibility are often used interchangeably. Being able to *say* this is itself an exam point.

### The model spectrum
- **White boxes = complete transparency ("glass boxes")**: **linear/logistic regression, case-based reasoning (k-NN, prototypes), decision trees.** Know each one's **caveat**: linear models ⇒ **low performance on complex phenomena**; k-NN ⇒ **computationally expensive** (and *"which is a good prototype?"*); decision trees ⇒ **prone to overfitting**.
- **Grey boxes = interpretable TO SOME DEGREE IF CAREFULLY DESIGNED**: **fuzzy rule-based systems** and **Bayesian networks**.
- **⚠️ Why FRBS are GREY and not white**: the rule structure is clear, but **uncertainty enters through the LINGUISTIC VARIABLES**. **Fuzzy sets (Zadeh, 1965) allow gradual transitions and OVERLAPPING regions**; **crisp sets would introduce a BIAS OF PERCEPTION**, because there is no absolute definition of "low" or "high".
- **A Bayesian network is a probabilistic DIRECTED ACYCLIC graph whose links are CONDITIONAL DEPENDENCIES** (diseases ↔ symptoms).
- **Black boxes = not designed to be interpretable, but we still want them EXPLAINABLE.** Typical case: **neural networks with many neurons**; the workflow is **maximize performance first, explain afterwards** ⇒ **post-hoc explainability**.
- **⚠️ THE ACCURACY–INTERPRETABILITY TRADE-OFF is an INVERSE relationship**: higher accuracy needs higher complexity needs lower interpretability. **Post-hoc methods live in the ensemble/SVM region.**

### Evaluating interpretability
- **Decision trees and FRBS are both "LOGICAL MODELS"** — both expressible as **if–then rules**.
- **QUANTITATIVE evaluation uses COMPLEXITY AS A PROXY**: number of rules / antecedents / coefficients. **QUALITATIVE evaluation is about SEMANTICS**: *"low temperature is 0–5 °C for me — but for you?"*, and **too many fuzzy labels ("very very very hot") destroy interpretability.**
- **⚠️ Miller's law, $7 \pm 2$**: humans handle about seven cognitive entities at once ⇒ **SPARSE models are preferable** because they **reduce cognitive load**.

### Taxonomies
- **Two routes to explainability: (1) design inherently interpretable models; (2) post-hoc techniques for opaque models.**
- **The FOUR post-hoc points of view: human resemblance / by dataset / global–local / agnostic–specific.**
- **Purposes of interpretability (lecture's first taxonomy dimension): create white boxes, explain black boxes, enhance FAIRNESS, test SENSITIVITY of predictions.**
- **⚠️ Model specific = precise but not portable; model agnostic = portable but computationally heavier / less precise.**
- **⚠️ Global vs. local applies to BOTH axes but means different things**: post-hoc **local explanators explain single predictions**, **global explanators explain the whole model and are OFTEN BUILT BY AGGREGATING LOCAL ONES**; for interpretable models, **local interpretability = the fired rule / root-to-leaf path**, **global interpretability = the model's STRUCTURE, i.e. its COMPLEXITY.**
- **Know the data-type table**: tabular ⇒ feature relevance + rule extraction; image ⇒ **saliency maps + concept attribution**; text ⇒ **sentence highlighting + attention**; time series ⇒ **series highlighting + temporal attention**; graph ⇒ **node/edge highlighting**.
- **Prototypes = representative examples of a class; counterfactuals = minimally different examples with a DIFFERENT prediction** (⇒ they reveal the **decision boundary**).
- **Surrogate models (explanation by simplification): the canonical choice is a DECISION TREE**, and the surrogate is **necessarily an APPROXIMATION.**

### Permutation importance
- **$I(j) = s - \frac{1}{K}\sum_k s(k,j)$**: shuffle the feature $K$ times, recompute the score, average, subtract from the original score. **Shuffling BREAKS the feature–target relationship.**
- **⚠️ The slide's question: a CONSTANT feature has importance EXACTLY 0** — shuffling identical values changes nothing.

### SHAP
- **SHAP estimates SHAPLEY VALUES from cooperative game theory** (Lundberg 2017; Shapley, Nobel 2012). **Team = model, players = features, victory = prediction.**
- **⚠️ The Shapley value is the AVERAGE MARGINAL CONTRIBUTION of a feature ACROSS ALL POSSIBLE COALITIONS** — not the contribution in one scenario. *"Not only the goal scorer matters."*
- **The prediction decomposes additively: BASE VALUE + sum of SHAP values**, where the **base value = average model output over the BACKGROUND dataset**.
- **⚠️ A coalition is evaluated by KEEPING the selected features and REPLACING the others with values from the background dataset** — this **simulates the absence** of those features. **The background dataset is a subset of the training set, built by sampling or clustering (e.g. cluster prototypes) to cut computation.**
- **Be able to redo the apartment example**: prediction €300,000, baseline €310,000; for `cat_banned`, coalition 1 gives €310,000 and coalition 2 (cat allowed) gives €320,000 ⇒ **marginal contribution €10,000**; then **average over all coalitions**.
- **A Shapley value carries MAGNITUDE and DIRECTION; the larger its ABSOLUTE value, the greater the impact.**
- **⚠️ Force plot = LOCAL (one prediction); beeswarm/summary plot = GLOBAL (aggregation of many local explanations).**
- **Reading a beeswarm**: **$x$-axis = SHAP value; colour = feature value (RED high, BLUE low); each point = one instance.** **Clean blue/red separation ⇒ strong, monotonic relationship. Mixed colours with wide spread ⇒ strong influence but NON-LINEAR / non-monotonic relationship.**

### LIME
- **LIME = Local Interpretable Model-agnostic Explanations: perturb around the instance → label the synthetic points with the black box → fit a simple surrogate WEIGHTED BY DISTANCE → explain locally.**
- **⚠️ The three hard problems: (1) generating meaningful synthetic data; (2) defining the notion of DISTANCE / "locality" and when to stop sampling; (3) the explanation is only valid LOCALLY, so global understanding requires "patching" many local models.**
- **⚠️ DOUBLE ERROR: the black box's error PLUS the surrogate's approximation error.** And crucially: **the surrogate is judged on FIDELITY to the black box, NOT on accuracy against the ground truth.**

### Counterfactuals and images
- **Counterfactual = the SMALLEST change in $x_0$ that flips the output to $y_1$.** **Actionable, human-aligned, and EASIER for untrained users than Shapley values.** Packages: **Alibi, MACE, DiCE**.
- **A saliency map encodes per-pixel importance: RED = positive contribution, BLUE = negative, GREY = no contribution.** It is **feature importance with pixels as features**.
- **⚠️ LIME on images works via SUPERPIXELS (image segments); GRAD-SHAP produces ONE SALIENCY MAP PER CLASS**, so it can explain several labels simultaneously.

### Regulation and trustworthiness
- **The 2019 EU Ethics Guidelines: AI must be LAWFUL, ETHICAL, ROBUST** (technically *and* socially). **Seven key requirements**; this lecture focuses on **TRANSPARENCY**.
- **The AI Act**: proposal **April 2021**, negotiating mandate **May 2023**, parliament endorsement **June 2023** — **"the first law on AI"**.
- **⚠️ Transparency has THREE aspects: COMMUNICATION (right to know you are talking to an AI), TRACEABILITY (document data gathering/labelling and algorithms), EXPLAINABILITY (adapted to the stakeholder).**
- **⚠️ And transparency has TWO senses**: **of the MODEL** (explainability) and **of the SOURCES** (traceability) — the latter now dominant because of **generative AI and HALLUCINATIONS** (the fabricated-citations court case; **more complexity does not fix hallucination and may worsen it**).
- **Wittgenstein's lion** is the memorable argument that **AI may reason in ways with no human analogue**, so explainability cannot be assumed from familiarity.

### Cross-chapter connections
- **Decision trees** (Chapter 4) are the recurring white-box example here — and **their overfitting tendency**, noted in Chapter 4, is repeated as their interpretability caveat. **Root-to-leaf path = local interpretability.**
- **k-NN** (Chapter 4) reappears as **case-based reasoning**, with **its computational cost** as the known drawback; **PROTOTYPES** connect to **cluster representatives / medoids** (Chapter 5), and the lecture explicitly proposes **clustering prototypes to build SHAP's background dataset**.
- **Random forests and ensembles** (Chapter 4) are the canonical opaque models needing post-hoc explanation.
- **Feature relevance doubles as FEATURE SELECTION** — the attribute-subset-selection problem of **Chapter 3**.
- **Class-imbalance and fairness concerns** (Chapters 4 and 7) return here as **bias mitigation** and the **criminal-justice/credit-scoring** scenarios.
- **The interpretability-vs-accuracy trade-off** mirrors the **model-complexity / overfitting trade-off** running through Chapter 4.

---

*File auto-generated by merging `13-ExplainableIA.pdf` (professor's slides, 20 pages — deck by Ducange, Marcelloni, Ruffini) and `13-ExplainableIA sbobine.pdf` (lecture notes, 11 pages — lesson L12 of 16/04/2026). Formulas and figures that the deck stored as images (the permutation-importance formula, the SHAP additive decomposition, the taxonomy diagrams, the beeswarm and force plots) have been reconstructed or described from the surviving text and the lecture's commentary. The final section on transparency as source traceability and on hallucinations comes from the lecture notes only. For questions about this chapter, refer only to this file.*
