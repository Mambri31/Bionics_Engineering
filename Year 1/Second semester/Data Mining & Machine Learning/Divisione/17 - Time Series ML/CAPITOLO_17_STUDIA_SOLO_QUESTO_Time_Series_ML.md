# Chapter 17 — Time Series: Machine Learning Approaches

> **Course:** Data Mining and Machine Learning / Bioinspired Computational Methods — Biological Data Mining
> **Instructor:** Prof. Francesco Marcelloni (University of Pisa)
> **Sources merged into this file:** professor's slides (`17-TimeSeries_MachineLearningApproaches.pdf`, 15 pages) + lecture notes (`17- Time series ML.pdf`, 13 pages)
>
> **⚠️ Note on the material**: as for Chapter 16, the slides are **extracted from Marco Peixeiro, *Time Series Forecasting in Python*, Manning Publications Co., 2022.**
>
> **⚠️ Prerequisite**: this chapter continues directly from **Chapter 16 (Time Series: Traditional Approaches)** — ARIMA/SARIMA, stationarity and the chronological train/test split are assumed known.
>
> **Instructions for Claude:** this is the single reference file for Chapter 17. When the user asks study questions about this chapter, use **only** this file as context (no need to re-read the original PDFs unless a detail is missing). The file merges the official slide content with the lecture's discursive explanations and worked examples, keeping both levels of detail.

---

## Chapter outline
1. When and why to use Machine Learning instead of statistical models
2. The three model types: single-step, multi-step, multi-output
3. Encoding time: the sine/cosine transformation
4. Preparing the dataset: split, scaling, data windowing, batches
5. Baseline models
6. Deep learning models: linear and deep neural networks
7. Recurrent Neural Networks (RNN)
8. Long Short-Term Memory (LSTM) and its three gates
9. Results and a final note on foundational models

---

## 1. When and why use Machine Learning?

> **The key question** (lecture): *when and why should we use machine learning approaches instead of classical statistical models?*

> **STATISTICAL METHODS WORK PARTICULARLY WELL WHEN YOU HAVE SMALL DATASETS (LESS THAN 10,000 DATA POINTS). OTHERWISE, THEY BECOME VERY SLOW.**
>
> **⚠️ MACHINE LEARNING IS USED EITHER:**
> 1. **when STATISTICAL MODELS TAKE TOO MUCH TIME TO FIT** *(computationally prohibitive for large datasets)*, **OR**
> 2. **when they result in CORRELATED RESIDUALS THAT DO NOT APPROXIMATE WHITE NOISE** *(the model does not fully capture the underlying structure of the series — the data exhibit complex patterns not adequately captured by classical methods).*

> **⚠️ Note how criterion (2) links back to Chapter 16**: it is exactly the **Ljung-Box failure** — if the residuals of your ARIMA/SARIMA remain **autocorrelated**, the model is inadequate, and that is a signal to move to ML rather than to keep grid-searching $(p,d,q)$.

**On foundational models** (lecture):
> **Modern ML approaches for time series are often built upon FOUNDATIONAL MODELS trained on large amounts of historical data.** These **generalize well to NEW time series that are STRUCTURALLY SIMILAR to the series used in their training datasets**, allowing for improved predictive performance **while still retaining statistical rigor.**

## 2. The three types of model

> **Machine learning models for time series forecasting can be categorized according to the TYPE OF PREDICTION they generate:**

| Model | Definition | When it is used |
|---|---|---|
| **SINGLE-STEP** | **outputs a SINGLE VALUE representing the prediction for the NEXT timestep.** **⚠️ The INPUT CAN BE OF ANY LENGTH, but the OUTPUT REMAINS A SINGLE PREDICTION one step ahead.** | applications requiring **high-frequency, short-term predictions**, where each forecast is **updated continuously** |
| **MULTI-STEP** | **the output is a SEQUENCE OF VALUES representing predictions for MANY TIMESTEPS into the future.** *E.g. if the model predicts the next 6 hours or 24 hours, it is multi-step.* | scenarios requiring **planning over a longer horizon** — **resource allocation, energy demand forecasting** |
| **MULTI-OUTPUT** | **generates predictions for MORE THAN ONE TARGET.** *E.g. forecasting temperature AND wind speed.* | when **multiple interdependent time series** must be predicted simultaneously — the model **can leverage CORRELATIONS BETWEEN THE OUTPUTS to improve accuracy** |

> **⚠️ Do not confuse MULTI-STEP with MULTI-OUTPUT.** **Multi-step = many TIMESTEPS of ONE variable. Multi-output = one (or more) timestep of SEVERAL variables.** The lecture notes that **the multi-step case is "of course the most complicated"**.

## 3. ⚠️ Encoding time: the sine/cosine transformation

> **The question: how can we encode time expressed as `YYYY-MM-DD HH:MM:SS`?**

**The problem with raw time encoding** (lecture):
> - **If we use raw timestamps in seconds, the model perceives that TIME IS ALWAYS INCREASING LINEARLY.**
> - **Consequently, observations at THE SAME HOUR ON DIFFERENT DAYS are represented by VERY DIFFERENT numerical values.**
> - **This encoding FAILS TO TRANSMIT the CYCLICAL or SEASONAL relationships** in the data — critical for daily traffic patterns, hourly bandwidth usage, electricity demand.

**The solution — map time onto the UNIT CIRCLE:**

> **First, transform the date into SECONDS.** For daily cycles:
> $$\text{day} = 24 \times 60 \times 60 = 86{,}400 \text{ seconds}$$
>
> **Then apply a SINE transformation:**
> ```python
> day = 24*60*60
> df['day_sin'] = (np.sin(timestamp_s * (2*np.pi/day))).values
> ```
> **⚠️ But using this transformation ALONE, 12 p.m. is equivalent to 12 a.m.** — the sine takes the same value twice per cycle.
>
> **Second, apply a COSINE transformation to avoid the problem:**
> ```python
> df['day_cos'] = (np.cos(timestamp_s * (2*np.pi/day))).values
> ```

**The resulting encoding:**

| Time | (sin, cos) |
|---|---|
| **00:00** | **(0, 1)** |
| **06:00** | **(1, 0)** |
| **12:00** | **(0, −1)** |
| **18:00** | **(−1, 0)** |

> **⚠️ THE POINT: neither transformation works ALONE.** *"The sine has value 0 both at 00:00 and at 12:00… Using the cosine, we have the same problem for 6:00 and 18:00. However, COMBINING THE TWO, we get the representation of a cycle for the hours of the day."*
> **Each UNIQUE PAIR of sine and cosine values represents a specific time of day, and this pattern REPEATS EVERY 24 HOURS, naturally encoding the daily seasonality.**
> **⇒ With ONLY TWO FEATURES we can encode all possible times in a day while preserving the periodic repetition across multiple days.**
>
> **And times that are temporally CLOSE in the cycle (e.g. 23:00 and 01:00) are also CLOSE in the transformed space** — which is precisely what the raw encoding destroys.

> **⚠️ Changing the period changes the seasonality modelled**: *"if we want to model a different type of seasonality — weekly or yearly — the same circular encoding can be applied, but THE PERIOD OF THE CYCLE MUST BE ADJUSTED accordingly. The circle remains conceptually the same; only the length of the period changes."* (E.g. use one week in seconds instead of 86,400.)

**The application** (lecture): with hourly measurements of traffic or bandwidth, **if we want the model to predict usage at 2 a.m., it should recognize that 2 a.m. today is SIMILAR to 2 a.m. yesterday.** Sine/cosine encoding **ensures the model perceives these points as close in the input space.**

## 4. Preparing the dataset

### 4.1 Split and scaling

> **We use a simple 70:20:10 split for the TRAINING, VALIDATION and TEST sets.**
> **We SCALE all the values between 0 and 1 — this DECREASES THE TIME FOR TRAINING deep learning models.**

> **Why a VALIDATION set specifically** (lecture): the big problem is **OVERTRAINING** — *"the model is too specific on the training set and unable to generalize."* To detect it, **compare performance on training vs. test**. In practice **we perform a SECOND SPLIT on the training data**: one part is the actual training set, the other the **VALIDATION SET, used DURING the learning phase to test the model's capability to generalize.**
> **⚠️ This is especially useful for NEURAL NETWORKS, which tend to suffer from overtraining**: while iterating to learn the weights, **the validation set tells us early whether we are heading toward overfitting.**

### 4.2 Data windowing

> **DATA WINDOWING PROCESS: we define a SEQUENCE of data points on the time series and define WHICH ARE INPUTS and WHICH ARE LABELS.**
> *(Lecture: in classification the labels are the classes; here **the labels are the values to be predicted**.)*

**The basic example:**
> **We can adopt a window of 24 HOURS to predict the next 24 HOURS.** The total dataset is split into **windows of 48 hours**, where **the first 24 hours form the INPUT** and **the next 24 hours form the TARGET OUTPUT.** *(First instance: hours 1–24 as input, 25–48 as output.)*
> **Obviously, the overall training set is separated into multiple windows.**

**⚠️ The improvement — MOVING WINDOWS with step 1:**
> **With the approach above, WE WASTE A LOT OF TRAINING DATA** — every instance consumes 48 fresh hours.
> **⇒ We can exploit MOVING WINDOWS WITH STEP 1**: *"if the first instance has input from $t=0$ to $t=23$ and label from $t=24$ to $t=47$, the second instance is just shifted by one and has input from $t=1$ to $t=24$…"*
> **⇒ The advantage: we create MANY OVERLAPPING SEQUENCES, which SIGNIFICANTLY INCREASES the number of training samples available.**

### 4.3 Batches, epochs and shuffling

> - **Deep learning models are trained with BATCHES. A BATCH is a COLLECTION OF DATA WINDOWS fed to the model for training** — the **parameters are UPDATED AFTER EACH BATCH.** *For example, batch size 32.*
> - **For a training set with 12,285 rows, we have 384 batches.**
> - **Training the model on ALL the batches is called ONE EPOCH.**
> - **A LOT OF EPOCHS are normally necessary to improve the accuracy of the predictions.**
> - **SHUFFLING is used AT THE BATCH LEVEL.**

> **⚠️ Why shuffling does NOT break the time series** (lecture — an important subtlety): *"it's not a problem if we change the ORDER OF THE INSTANCES, because we still have the time associated with the values."* **Each instance already carries its input sequence and its label in the correct internal order; shuffling only changes the order in which whole windows are presented.**

**The two metrics:**
> - **LOSS FUNCTION: MEAN SQUARED ERROR (MSE)** — computed **after each batch**;
> - **EVALUATION METRIC: MEAN ABSOLUTE ERROR (MAE)** — computed **after each epoch.**

## 5. Baseline models

> **As usual, we use baseline models to have a METRIC FOR COMPARISON.** *(Lecture: "if we want to test a complex model, we usually spend a lot of time training it and then we might realize that the output is comparable to the baseline, so it's not useful.")*

### 5.1 Single-step baseline

> **The input is ONE timestep and the output is the prediction of the NEXT timestep.**
> **Window: input width 1, label width 1, shift 1.**
> **THE SIMPLEST PREDICTION WE CAN MAKE IS THE LAST OBSERVED VALUE — the prediction is simply the input data point.**

> *(Lecture: in the plot "the red dots are just the green ones shifted". **If the variations are slow, this model is quite good, but it strongly depends on the time interval used between timesteps.**)*

### 5.2 Multi-step baseline

> **For a multi-step of $N$, the input is $N$ timesteps and the output is the prediction of $N$ timesteps** (for instance $N = 24$ hours).
> **Window: input width $N$, label width $N$, shift $N$.**
>
> **TWO possible baselines:**
> 1. **Predict the LAST KNOWN VALUE for the next 24 timesteps** — *simply takes the input and REPEATS THE LAST VALUE of the input sequence over $N$ timesteps.*
> 2. **Predict the LAST 24 TIMESTEPS for the next 24 timesteps** — *the prediction for the next $N$ timesteps is simply the last known $N$ timesteps of data.*

> **⚠️ Which works better, and why** (lecture): **baseline 1 "doesn't work very well, especially if there are fluctuations — the error between prediction and actual value is very high."** **Baseline 2 (repeating the input sequence) works BETTER — but only BECAUSE THERE IS A REPETITION OF BEHAVIOUR DAY TO DAY.** *(I.e. it is really exploiting daily seasonality — the same insight as the naïve seasonal baseline of Chapter 16.)*

### 5.3 Multi-output baseline

> **A single-step model used for predicting DIFFERENT OUTPUTS.** *Assuming two outputs, we apply a single-step model for predicting both* — **i.e. we process them SEPARATELY.**

## 6. Deep Learning models

### 6.1 The linear model

> **A SIMPLE LINEAR MODEL — not actually a deep learning model: A SIMPLE MULTIVARIATE LINEAR REGRESSION.**
> Declined in all three variants: **single-step**, **multi-step linear model**, **multi-output linear model.**

> *(Lecture: "we use several inputs at instant $t$ to predict the traffic volume at instant $t+1$… By using this single-step model the approximation can work quite well IF THE MODEL HAS A LINEAR BEHAVIOUR." For the multi-step version the input is **not a single value at $t$ but a whole window from $t=0$ to $t=23$.**)*

### 6.2 The deep neural network

> **ACTIVATION FUNCTION: in each neuron of the hidden layers, responsible for generating the output.**
> **⚠️ IF A LINEAR ACTIVATION FUNCTION IS USED, THE MODEL WILL ONLY MODEL LINEAR RELATIONSHIPS. THEREFORE, TO MODEL NONLINEAR RELATIONSHIPS IN THE DATA, WE MUST USE A NONLINEAR ACTIVATION FUNCTION.**
> **Example: the RECTIFIED LINEAR UNIT (ReLU).**

Again declined as **single-step**, **multi-step** and **multi-output** models.

> **Result** (lecture): **"if we use deep neural network models, they can predict the time series better, and on the test set the MEAN ABSOLUTE ERROR DECREASES A LOT."**

## 7. Recurrent Neural Networks (RNN)

> **A RECURRENT NEURAL NETWORK (RNN) is a deep learning architecture ESPECIALLY ADAPTED TO PROCESSING SEQUENCES OF DATA.**
> **It uses a HIDDEN STATE THAT IS FED BACK INTO THE NETWORK, so it can use PAST INFORMATION AS AN INPUT when processing the next element of a sequence.**
> **⇒ THIS IS HOW IT REPLICATES THE CONCEPT OF MEMORY.**
>
> **⚠️ RNNs SUFFER FROM SHORT-TERM MEMORY, meaning that information from an EARLY element in the sequence WILL STOP HAVING AN IMPACT further into the sequence.**

**The analogy given in class:**
> **An RNN works by maintaining the memory of what has been seen so far to process a sequence. This is similar to what we do WHEN WE READ A SENTENCE: we start reading from left to right and follow the sentence, making sense of it as we go. For example, SOME WORDS MIGHT HAVE DIFFERENT MEANINGS and we can understand the right one by reading the whole sentence.**
>
> **KEY IDEA: at each word, I use what I learned from previous words to make sense of the current one.**
> **RNNs keep memory in the HIDDEN STATE that SUMMARIZES EVERYTHING SEEN SO FAR. These memories are UPDATED AT EACH STEP, and the output is generated using BOTH the current input AND the previously learned memories.**

## 8. Long Short-Term Memory (LSTM)

> **LSTM (controlled and selective memory) is a deep learning architecture that is a SUBTYPE OF RNN.**
> **⚠️ LSTM ADDRESSES THE PROBLEM OF SHORT-TERM MEMORY BY ADDING THE CELL STATE. This allows PAST INFORMATION TO FLOW THROUGH THE NETWORK FOR A LONGER PERIOD OF TIME, meaning the network STILL CARRIES INFORMATION FROM EARLY VALUES in the sequence.**

> **The problem restated** (lecture): *"in the standard RNN information is passed through time BUT IT TENDS TO FADE AWAY; when the time between two important events is large, THE EARLY INFORMATION MIGHT BE LOST."* **LSTM is a smart memory system because it introduces a cell with GATES that control the flow of information.**

### 8.1 The three gates

> **The LSTM is made up of THREE GATES:**
> 1. **The FORGET GATE determines WHAT INFORMATION FROM PAST STEPS IS STILL RELEVANT** (and what can be forgotten because irrelevant).
> 2. **The INPUT GATE determines WHAT INFORMATION FROM THE CURRENT STEP IS RELEVANT** (so must be kept).
> 3. **The OUTPUT GATE determines WHAT INFORMATION IS PASSED ON to the next element of the sequence or as a result to the output layer** (which part of the cell state to output as hidden state).

**How each gate works (the slide detail):**

| Gate | Mechanism |
|---|---|
| **FORGET** | **The present element $x_t$ and past information $h_{t-1}$ are first COMBINED. They are DUPLICATED — one copy is sent to the input gate, the other goes through the SIGMOID activation function. The sigmoid outputs a value between 0 and 1: CLOSE TO 0 ⇒ information must be FORGOTTEN; CLOSE TO 1 ⇒ the information is KEPT. The output is then combined with the past cell state using POINTWISE MULTIPLICATION, generating an updated cell state $C'_{t-1}$.** |
| **INPUT** | **The past hidden state and current element are duplicated again and sent through a SIGMOID and a HYPERBOLIC TANGENT (tanh). Again, THE SIGMOID DETERMINES WHAT INFORMATION IS KEPT OR DISCARDED, while THE TANH REGULATES THE NETWORK TO KEEP IT COMPUTATIONALLY EFFICIENT.** |
| **OUTPUT** | **$[h_{t-1} + x_t]$ are passed through the SIGMOID to determine if information will be kept or discarded. Then THE CELL STATE is passed through the TANH and combined with the sigmoid output using POINTWISE MULTIPLICATION. This is the step where PAST INFORMATION IS USED TO PROCESS THE CURRENT ELEMENT. We then output a NEW HIDDEN STATE $h_t$, passed to the next LSTM neuron or to the output layer. THE CELL STATE IS ALSO OUTPUT.** |

*(The **candidate $c_t$** creates the new content that could be added to the cell state.)*

**The analogy given in class:**
> **Consider an LSTM as a WAREHOUSE MANAGER: the CELL STATE is the WAREHOUSE that stores information, and the GATES are the MANAGER'S DECISIONS about the boxes. The manager OBSERVES what is present and what is new, PREPARES the new information, and DECIDES what to show right now.**
>
> **KEY IDEA: to LEARN AUTOMATICALLY WHAT TO FORGET, WHAT TO REMEMBER, AND WHAT TO USE RIGHT NOW. This allows LSTM to hold important information for a long time, SOLVING THE VANISHING MEMORY PROBLEM of standard RNNs.**

> **📌 EXAM-RELEVANT REMARK FROM THE LECTURE**: *"In the slides there are the MATHEMATICAL SCHEMES of the three gates, but **THE PROFESSOR DID NOT WANT TO PUT FOCUS ON THEM**. What he wanted to show was just that by using an LSTM we are able to REDUCE THE MEAN ABSOLUTE ERROR with respect to the baseline, the linear model and the dense neural networks — in the single-step, multi-step and multi-output models alike."*
> **⇒ Understand the ROLE of each gate and the cell state; do not memorise the gate equations.**

### 8.2 The three LSTM configurations

> - **SINGLE-STEP model: we use $N$ timestamps to predict THE NEXT timestamp.**
> - **MULTI-STEP model: we use $N$ timestamps to predict $N$ timestamps.**
> - **MULTI-OUTPUT model: we use $N$ timestamps to predict ONE timestamp** (for several targets).

## 9. Results and closing note

> **The bottom line**: across **single-step, multi-step and multi-output** settings, the **LSTM reduces the MAE** with respect to **the baseline, the linear model, and the dense neural network.**

> **A final observation from the lecture:**
> **"Today we don't actually use only these types of NN, but can also find FOUNDATIONAL MODELS, trained on millions and millions of data."**
> **⚠️ "It's still always good to PERFORM SOME ANALYSIS ON THE TIME SERIES, because EXPLAINABILITY IS STILL VERY IMPORTANT."**
> **"All these neural networks can be found in libraries."**

---

## Key points / potential exam pitfalls

### When ML instead of statistics
- **⚠️ TWO triggers, and only two: (1) statistical models are TOO SLOW / computationally prohibitive (roughly, beyond ~10,000 data points); (2) their RESIDUALS ARE CORRELATED and do not approximate white noise.**
- **Trigger (2) is exactly a FAILED LJUNG-BOX TEST from Chapter 16** — that is the concrete signal to abandon ARIMA/SARIMA rather than keep tuning $(p,d,q)$.
- **Statistical methods are NOT obsolete — they work particularly well on SMALL datasets.** Do not answer "ML is always better".
- **FOUNDATIONAL MODELS** generalize to **new series that are STRUCTURALLY SIMILAR** to their training data.

### The three model types
- **⚠️ SINGLE-STEP: input of ANY length, output ALWAYS ONE value.** The length of the input is not what defines it — **the OUTPUT is.**
- **MULTI-STEP = many TIMESTEPS. MULTI-OUTPUT = many TARGETS.** Multi-step is the hardest.
- **Multi-output models can exploit CORRELATIONS BETWEEN THE OUTPUTS.**

### Encoding time
- **⚠️ THE CENTRAL IDEA OF THIS CHAPTER'S FEATURE ENGINEERING: raw time in seconds INCREASES LINEARLY FOREVER, so the same hour on different days gets completely different values, and cyclicality is lost.**
- **The fix is a PAIR of features: $\sin(2\pi t/T)$ and $\cos(2\pi t/T)$, with $T = 86{,}400$ s for a daily cycle.**
- **⚠️ NEITHER ALONE SUFFICES**: sine alone confuses **00:00 with 12:00**, cosine alone confuses **06:00 with 18:00**. **Only the PAIR identifies each instant uniquely** — a point on the unit circle.
- **Know the four anchor points: 00:00 → (0,1); 06:00 → (1,0); 12:00 → (0,−1); 18:00 → (−1,0).**
- **To model weekly or yearly seasonality, CHANGE THE PERIOD $T$** — the construction is otherwise identical.

### Data preparation
- **70:20:10 train/validation/test; scale to $[0,1]$ to SPEED UP training.**
- **⚠️ The VALIDATION set exists to detect OVERTRAINING DURING learning** — especially important for neural networks.
- **⚠️ MOVING WINDOWS WITH STEP 1 instead of disjoint windows: the disjoint approach WASTES TRAINING DATA; overlapping windows GREATLY INCREASE the number of training samples.**
- **BATCH = a collection of data windows (parameters updated after each batch); EPOCH = one pass over ALL batches; many epochs are normally needed.**
- **⚠️ SHUFFLING IS AT THE BATCH LEVEL AND DOES NOT BREAK THE TIME SERIES**, because each window already carries its own internal temporal order and its label. Be ready to explain why this does not contradict Chapter 16's "never split randomly" rule — **the SPLIT is still chronological; only the PRESENTATION ORDER of windows is shuffled.**
- **LOSS = MSE (per batch); EVALUATION METRIC = MAE (per epoch).** Know which is which.

### Baselines
- **Single-step baseline = predict the LAST OBSERVED VALUE** (window 1/1/1). **Good when variations are slow; strongly dependent on the timestep interval.**
- **⚠️ TWO multi-step baselines: (1) REPEAT THE LAST VALUE $N$ times; (2) REPEAT THE LAST $N$ TIMESTEPS.** **(2) performs better — but ONLY BECAUSE the behaviour repeats day to day**, i.e. it is implicitly exploiting seasonality.
- **Multi-output baseline = apply a single-step model to EACH output SEPARATELY.**

### Models
- **The "linear model" is NOT deep learning — it is multivariate linear regression**, available in all three variants.
- **⚠️ A LINEAR ACTIVATION FUNCTION ⇒ ONLY LINEAR RELATIONSHIPS. Nonlinearity REQUIRES a nonlinear activation (e.g. ReLU).**
- **RNN: a HIDDEN STATE FED BACK into the network reproduces MEMORY.** The **sentence-reading analogy** is the professor's own.
- **⚠️ THE RNN's DEFECT IS SHORT-TERM MEMORY — early elements stop influencing the sequence (fading / vanishing memory).**
- **⚠️ LSTM FIXES IT BY ADDING THE CELL STATE**, letting information flow for longer.
- **THE THREE GATES: FORGET (what past information is still relevant) · INPUT (what current information is relevant) · OUTPUT (what is passed on / becomes the new hidden state).** The **warehouse-manager analogy** captures it: *learn what to forget, what to remember, what to use now.*
- **Mechanism vocabulary: SIGMOID decides KEEP (≈1) vs. FORGET (≈0); TANH regulates the network / keeps it computationally efficient; gates combine with the cell state by POINTWISE MULTIPLICATION.**
- **📌 The professor explicitly DE-EMPHASISED the gate equations** — know the roles, the cell state, and the fact that **LSTM lowers the MAE versus baseline, linear and dense models in all three configurations.**
- **⚠️ Even with foundational models, ANALYSING the time series remains important — because EXPLAINABILITY still matters** (a direct callback to Chapter 13).

### Cross-chapter connections
- **Chapter 16 is the prerequisite**: the **correlated-residuals trigger** for using ML is a **failed Ljung-Box test**; the **repeat-the-last-$N$-timesteps baseline** is the **naïve seasonal forecast**; and the **chronological split** discipline carries over (with the batch-shuffling nuance above).
- **Overfitting, train/validation/test splits, activation functions and neural networks** come from **Chapter 4 (Classification)** — here applied to a regression-over-time task.
- **Feature scaling to $[0,1]$** is **normalization from Chapter 3 (Data Preprocessing)**; the **sine/cosine encoding is FEATURE CONSTRUCTION**, also a Chapter 3 idea, applied to a cyclical attribute.
- **The black-box nature of RNNs/LSTMs and the closing remark on explainability** point straight to **Chapter 13 (XAI)** — deep models buy accuracy at the cost of interpretability, the same trade-off stated there.
- **Sequences and temporal order** connect to **Chapter 12 (Sequential Pattern Mining)** and **Chapter 15 (Data Streams)**: RNN/LSTM are the neural answer to the same "order carries information" problem, and a **single-step model updated continuously** is conceptually a stream learner.

---

*File auto-generated by merging `17-TimeSeries_MachineLearningApproaches.pdf` (professor's slides, 15 pages, extracted from Peixeiro, "Time Series Forecasting in Python", Manning 2022) and `17- Time series ML.pdf` (lecture notes, 13 pages). Figures that the deck stored as images (the unit-circle time encoding, the windowing and batching diagrams, the RNN/LSTM architecture schemes and the gate equations) have been described from the surviving text and the lecture's commentary; per the professor's explicit remark, the gate equations are not the focus. For questions about this chapter, refer only to this file.*
