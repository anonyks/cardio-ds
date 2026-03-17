# info / cheat sheet for heart disease prediction notebook

ok so this is basically everything thats needed for this project. 

---

## 0. foundational terms (basics)

### what is supervised learning

this entire project is **supervised learning**. "supervised" = the dataset comes with the correct answers already (the `target` column). the model learns the relationship between input features and the known output, then tries to predict the output for new unseen data.

the other type is **unsupervised learning** = no labels, the model just tries to find patterns on its own (like clustering). not used here.

### what is a model

a **model** is basically a math function that takes input data and spits out a prediction. "training" the model = feeding it data so it can learn the best parameters (weights) that minimize errors. once trained, the model can be used to predict on new data it hasnt seen before.

### features vs labels

- **features** = the input columns used for prediction (age, cholesterol, etc). also called independent variables or predictors.
- **label / target** = the output column thats being predicted. also called dependent variable or response variable.
- in this project: 13 features, 1 target (heart disease yes/no)

### parameters vs hyperparameters

- **parameters** = values the model learns during training (like weights and biases). cant be set manually.
- **hyperparameters** = values set BEFORE training (like learning rate, number of layers, C in logistic regression). the model doesnt learn these -- they come from the human or from tuning.

### what is a DataFrame

a **DataFrame** (from pandas) is a 2D table of data with labeled rows and columns -- basically an excel spreadsheet in python. each column can have a different data type. most data science work in python revolves around manipulating DataFrames.

---

## 1. the libraries (and why each one is needed)

### numpy (`np`)

the math library for python. does fast operations on arrays (lists but faster and more memory efficient). not used directly much here but pandas and sklearn depend on it internally so it needs to be imported.

```python
import numpy as np
```

### pandas (`pd`)

THE library for working with tabular data in python. provides the **DataFrame** object. can load csv files, filter rows, drop columns, get stats, etc. used everywhere in this notebook.

```python
import pandas as pd
```

### matplotlib (`plt`)

the OG plotting library. somewhat verbose (lots of code for a simple graph) but it works. `plt.plot()`, `plt.show()`, `plt.savefig()` etc. used here for bar charts, line plots, and the roc curve.

```python
import matplotlib.pyplot as plt
%matplotlib inline  # makes plots show up directly in the notebook instead of a separate window
```

### seaborn (`sns`)

built on top of matplotlib but makes prettier plots with less code. used mainly for the heatmap and confusion matrix visualizations. good for statistical plots.

```python
import seaborn as sns
```

### sklearn (scikit-learn)

THE machine learning library for python. has literally everything -- splitting data, scaling, models, metrics, tuning. a bunch of stuff got imported from it:

```python
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_auc_score, roc_curve
```

### tensorflow / keras

for building neural networks. keras is the high-level api that makes it easy to stack layers. tensorflow is the engine running underneath. Sequential means layers go one after another (like a sandwich).

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
```

### warnings

not a data science thing, just hides annoying warning messages that clutter the output. doesnt affect the actual code at all.

```python
import warnings
warnings.filterwarnings('ignore')
```

---

## 2. loading and looking at the data

### `pd.read_csv()`

reads a csv file and turns it into a DataFrame. csv = **Comma Separated Values**, basically a text file where each line is a row and commas separate the columns.

```python
df = pd.read_csv('heart.csv')
```

### `df.shape`

gives the dimensions of the dataframe as (rows, columns). so like `(303, 14)` means 303 patients and 14 columns. no parentheses after shape btw -- its a property not a method.

```python
print(df.shape)
```

### `df.head()`

shows the first 5 rows. good for a quick peek at what the data looks like. can also do `df.head(10)` for first 10 rows.

```python
df.head()
```

### `df.info()`

shows each column name, how many non-null values it has, and the data type (int64, float64 etc). useful to check for missing data or if a column has the wrong type.

```python
df.info()
```

### `df['target'].value_counts()`

counts how many times each unique value appears in a column. used on the target column to see how many patients have heart disease (1) vs dont (0). important to check if the classes are **balanced** (roughly equal counts) or **imbalanced** (one class way more than the other).

```python
df['target'].value_counts()
```

---

## 3. data cleaning

### missing values / `dropna()`

sometimes datasets have empty cells (**NaN** = Not a Number). models cant handle those so they need to be dealt with. options:
- **drop them** (`dropna()`) -- just remove rows with missing values. easy but data gets lost.
- **fill them** (`fillna()`) -- replace with mean, median, mode, or some other value. not needed here.

**NaN** stands for "Not a Number." its pythons way of saying "this cell is empty / missing." numpy and pandas both recognize it. can check for it with `df.isnull()` or `pd.isna()`.

```python
before = df.shape[0]
df = df.dropna()
after = df.shape[0]

print("dropped", before - after, "rows")
print("left with", after)
```

in this case it dropped 1 row. not a big deal.

---

## 4. EDA (Exploratory Data Analysis)

fancy name for "looking at the data before doing anything with it." the goal is to understand whats going on -- are there outliers? are the classes balanced? which features seem related to the target? etc.

### `df.describe()`

gives basic stats for each numeric column: count, mean, std (standard deviation), min, max, and the quartiles (25%, 50%, 75%). the 50% is the **median** (middle value). helps spot weird stuff like if the max cholesterol is 564 -- thats pretty high.

```python
df.describe()
```

### mean, median, standard deviation (the math behind describe)

**mean (average):**

$$\bar{x} = \frac{1}{n} \sum_{i=1}^{n} x_i$$

just add up all the values and divide by how many there are. problem: gets pulled by outliers. if 9 people earn 50k and 1 person earns 10 million, the mean income is misleading.

**median:**

the middle value when all values are sorted. if theres an even number of values, its the average of the two middle ones. not affected by outliers -- thats why its sometimes better than mean.

example: [1, 2, 3, 100] → mean = 26.5, median = 2.5. median is more representative here.

**standard deviation (std):**

$$\sigma = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (x_i - \bar{x})^2}$$

measures how spread out the values are from the mean. low std = values are clustered close to the mean. high std = values are all over the place.

the thing inside the square root is called **variance** ($\sigma^2$). std is just the square root of variance. variance is in squared units (hard to interpret), so std brings it back to the original units.

**quartiles and percentiles:**

- **25th percentile (Q1)** = 25% of values are below this
- **50th percentile (Q2)** = the median
- **75th percentile (Q3)** = 75% of values are below this
- **IQR (Interquartile Range)** = Q3 - Q1. measures the spread of the middle 50% of data. used to detect outliers -- anything below Q1 - 1.5×IQR or above Q3 + 1.5×IQR is often considered an outlier.

### bar plot (target distribution)

a simple bar chart showing how many 0s and 1s are in the target column. if one class has way more samples the model might just predict that class all the time and still get high accuracy (thats why accuracy alone can be misleading).

```python
df['target'].value_counts().plot(kind='bar')
plt.title('Target Distribution')
plt.xlabel('Target')
plt.ylabel('Count')
plt.savefig('obtained_fig/target_distribution.png', bbox_inches='tight')
plt.show()
```

### correlation & Pearson correlation coefficient

**correlation** = how much two variables move together. value goes from -1 to 1:
- **1** = perfect positive correlation (one goes up, other goes up)
- **-1** = perfect negative correlation (one goes up, other goes down)
- **0** = no linear relationship at all

`df.corr()` calculates the **Pearson correlation coefficient** for every pair of columns. the actual formula:

$$r = \frac{\sum (x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\sum (x_i - \bar{x})^2 \cdot \sum (y_i - \bar{y})^2}}$$

in english: for each pair of values, multiply how far each is from its mean, sum those products, and divide by a normalizing factor. the result is always between -1 and 1.

important note: correlation measures LINEAR relationships only. two variables can be strongly related in a non-linear way and still have near-zero correlation. also, **correlation ≠ causation** (just because two things move together doesnt mean one causes the other).

then plotted as a **heatmap** to visually see which features are related. red = positive correlation, blue = negative.

```python
plt.figure(figsize=(10, 8))
corr = df.corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.1f')
plt.title('Correlation Heatmap')
plt.savefig('obtained_fig/correlation_heatmap.png', bbox_inches='tight')
plt.show()
```

- `annot=True` -- shows the actual numbers on each cell
- `cmap='coolwarm'` -- the color scheme (red/blue)
- `fmt='.1f'` -- format numbers to 1 decimal place
- `figsize=(10, 8)` -- makes the plot bigger so its readable

---

## 5. preprocessing

### features (X) vs target (y)

in ML, **X** is the input data (the features/columns the model uses to make predictions) and **y** is the output (whats being predicted). they get split before training.

`axis=1` means drop a column. `axis=0` would drop a row.

```python
X = df.drop('target', axis=1)  # everything except target
y = df['target']               # just the target column
```

### train/test split

NEVER train and test on the same data. thats like giving students the exam answers and then testing them on the same questions -- obviously they'll do well but nothing was actually learned. so the data gets split:

- **training set** (usually 80%) -- model learns from this
- **test set** (usually 20%) -- model is evaluated on this, never seen during training

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,           # 20% for testing
    random_state=42,         # seed so the split is the same every time the code runs
    stratify=y               # keeps same ratio of 0s and 1s in both sets
)
```

- **`test_size=0.2`** -- 20% goes to test, 80% to train
- **`random_state=42`** -- a seed number. doesnt matter what number gets picked, its just so the "random" split is reproducible. 42 is a common convention
- **`stratify=y`** -- super important for classification. if the dataset has 55% disease and 45% no disease, stratify makes sure the train and test sets also have roughly 55/45. without it, the split might be unlucky and put all the disease cases in one set

### StandardScaler (feature scaling / standardization)

some features are big numbers (like cholesterol = 200-500) and some are tiny (sex = 0 or 1). models like logistic regression and neural networks are sensitive to the **scale** of input features. if one feature has way bigger numbers itll dominate the others.

**StandardScaler** fixes this by transforming each feature so that:
- **mean = 0** (centered)
- **standard deviation = 1** (same spread)

the formula is called **z-score normalization**:

$$z = \frac{x - \mu}{\sigma}$$

where $\mu$ is mean and $\sigma$ is standard deviation. after this transformation, a value tells how many standard deviations it is from the mean. e.g. z = 2 means the value is 2 standard deviations above average.

```python
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)  # learns mean/std from training data AND scales it
X_test_scaled = scaler.transform(X_test)         # uses the SAME mean/std to scale test data
```

**IMPORTANT**: `fit_transform` is used on train and just `transform` on test. if `fit` gets called on test data too, thats **data leakage** -- the model indirectly gets info about the test set, which is cheating. super common mistake.

### normalization vs standardization

people mix these up all the time:
- **standardization** (what StandardScaler does) -- shifts to mean=0, std=1. no fixed range. good when data is roughly normally distributed.
- **normalization** (MinMaxScaler) -- scales values to a fixed range like [0, 1]. formula: $x' = \frac{x - x_{min}}{x_{max} - x_{min}}$. good when the distribution isnt gaussian.

this project uses standardization.

### data leakage

**data leakage** = when information from the test set accidentally "leaks" into the training process. the model seems to perform great during evaluation but fails on truly new data because it already "saw" test info.

common causes:
- fitting the scaler on test data (using `fit_transform` on test instead of just `transform`)
- including the target variable as a feature by accident
- using future data to predict the past (in time-series)

bottom line: the test set must be completely isolated. pretend it doesnt exist until final evaluation.

---

## 6. logistic regression

### what it is

despite having "regression" in the name, its actually a **classification** algorithm (yeah confusing). it predicts the probability that something belongs to a class (like disease or no disease).

it works by:
1. takes a **weighted sum** of all input features: $z = w_1 x_1 + w_2 x_2 + ... + w_n x_n + b$
2. passes it through the **sigmoid function**: $\sigma(z) = \frac{1}{1 + e^{-z}}$
3. sigmoid squishes any number into a value between 0 and 1
4. if the output > 0.5 → predict class 1 (disease), else predict class 0 (no disease)

the model learns the weights ($w$) and bias ($b$) during training by minimizing the error.

#### what is the bias term ($b$)

the **bias** (also called intercept) is an extra learnable number added to the weighted sum. without it, the decision boundary always has to pass through the origin (0,0). the bias lets the model shift the boundary to fit the data better. think of it like the y-intercept in $y = mx + b$ -- same idea.

#### euler's number ($e$)

$e \approx 2.71828$. its a mathematical constant that shows up everywhere in calculus, probability, and ML. in the sigmoid formula $\frac{1}{1+e^{-z}}$, the $e^{-z}$ part is what creates the S-shaped curve. as z gets very large, $e^{-z}$ approaches 0 so the output approaches 1. as z gets very negative, $e^{-z}$ blows up so the output approaches 0.

#### logarithm (log)

a **logarithm** is the inverse of exponentiation. $\log_b(x) = y$ means $b^y = x$.

in data science, "log" almost always means the **natural logarithm** ($\ln$, base $e$). key properties:
- $\log(1) = 0$
- $\log(0)$ is undefined (goes to $-\infty$)
- $\log(ab) = \log(a) + \log(b)$

logs show up in the loss function (binary crossentropy) and in many ML formulas. the reason: multiplying small probabilities together makes the numbers insanely tiny. taking the log converts products into sums, which is numerically more stable.

### regularization (in depth)

regularization = adding a penalty to the model to prevent it from fitting the training data too closely (overfitting). it discourages the model from learning overly complex patterns.

the model normally minimizes the **loss** (how wrong the predictions are). with regularization, it minimizes:

$$\text{Total Cost} = \text{Loss} + \lambda \times \text{Penalty}$$

where $\lambda$ controls how much penalty is applied. in sklearn, $C = \frac{1}{\lambda}$, so:
- high C = low penalty = model fits training data more closely → risk of overfitting
- low C = high penalty = model is more constrained → risk of underfitting

**L1 regularization (Lasso):**

$$\text{Penalty} = \sum |w_i|$$

adds the sum of **absolute values** of all weights. can push some weights to exactly 0, effectively performing **feature selection** (removing useless features).

**L2 regularization (Ridge):**

$$\text{Penalty} = \sum w_i^2$$

adds the sum of **squared** weights. shrinks all weights towards zero but none become exactly 0. generally smoother and more stable.

### hyperparameters

hyperparameters = settings chosen before training (the model doesnt learn these).

- **C** -- inverse regularization strength. higher C = less regularization. too high → **overfitting**. too low → **underfitting**.
- **penalty** -- type of regularization (l1 or l2, explained above)
- **solver** -- the algorithm used internally to optimize. `liblinear` works for both l1 and l2 on small datasets.

### GridSearchCV (hyperparameter tuning)

instead of guessing which hyperparameters are best, **GridSearchCV** tries every combination given to it and picks the winner.

**"Grid"** = it creates a grid of all possible combos. so if C = [0.01, 0.1, 1, 10] and penalty = ['l1', 'l2'], thats 4 × 2 = 8 combos to try.

**"CV" = Cross-Validation** (explained below)

```python
lr_params = {
    'C': [0.01, 0.1, 1, 10],
    'penalty': ['l1', 'l2'],
    'solver': ['liblinear']
}

lr = LogisticRegression(random_state=42)
lr_grid = GridSearchCV(lr, lr_params, cv=5)
lr_grid.fit(X_train_scaled, y_train)

best_lr = lr_grid.best_estimator_      # the model that performed best
print("best params:", lr_grid.best_params_)
```

### cross-validation (cv=5)

instead of splitting training data into one train and one validation set, **k-fold cross-validation** splits it into k parts (here k=5). it trains 5 times, each time using a different part as the "validation" set and the rest for training. then it averages the results.

why? one split can be lucky or unlucky. averaging over 5 splits gives a more reliable estimate of how the model will perform on unseen data.

```
Fold 1: [VAL] [train] [train] [train] [train]
Fold 2: [train] [VAL] [train] [train] [train]
Fold 3: [train] [train] [VAL] [train] [train]
Fold 4: [train] [train] [train] [VAL] [train]
Fold 5: [train] [train] [train] [train] [VAL]
```

### feature importance (logistic regression coefficients)

after training, `coef_` gives the weight for each feature. **positive weight** = feature pushes prediction towards disease, **negative weight** = pushes towards no disease. bigger absolute value = more important.

```python
coefs = pd.Series(best_lr.coef_[0], index=X.columns)
coefs = coefs.sort_values()

coefs.plot(kind='barh', figsize=(8, 5))
plt.title('Feature Importance (LR Coefficients)')
plt.savefig('obtained_fig/feature_importance.png', bbox_inches='tight')
plt.show()
```

---

## 7. evaluation metrics

theres a bunch of ways to measure how good a model is. accuracy alone aint enough.

### accuracy

$$\text{Accuracy} = \frac{\text{correct predictions}}{\text{total predictions}}$$

sounds great but its misleading when classes are imbalanced. like if 95% of patients are healthy, a dumb model that always predicts "healthy" gets 95% accuracy but misses every sick person. thats bad. thats called the **accuracy paradox**.

### precision

$$\text{Precision} = \frac{TP}{TP + FP}$$

"of all the people predicted as having heart disease, how many actually have it?"

high precision = few false alarms. important when false positives are costly (like telling a healthy person they have a disease -- causes unnecessary stress and tests).

### recall (aka sensitivity / true positive rate)

$$\text{Recall} = \frac{TP}{TP + FN}$$

"of all the people who actually have heart disease, how many did the model catch?"

high recall = fewer missed sick people. super important in medical stuff -- missing a person with heart disease could literally kill them.

### precision-recall tradeoff

precision and recall are always in tension. increasing one usually decreases the other:
- lower the threshold (e.g. predict disease if probability > 0.3 instead of 0.5) → more people get flagged → recall goes up, but precision drops (more false alarms)
- raise the threshold → fewer people get flagged → precision goes up, but recall drops (more missed cases)

the right balance depends on the problem. in healthcare, recall is usually prioritized (better to have false alarms than miss a sick person).

### f1-score

$$F1 = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$

the **harmonic mean** of precision and recall. a single number that balances both. useful when its not clear which one matters more.

**why harmonic mean and not regular (arithmetic) mean?**

arithmetic mean of precision=0.9 and recall=0.1 would be 0.5 -- sounds decent. but having 0.1 recall means the model misses 90% of sick people, which is terrible. the harmonic mean gives 0.18 instead, which better reflects how bad that situation is. basically the harmonic mean punishes extreme imbalances.

$$\text{Arithmetic mean} = \frac{a + b}{2} \quad \quad \text{Harmonic mean} = \frac{2ab}{a + b}$$

### where TP, FP, TN, FN come from

- **TP (True Positive)** -- model says disease, patient actually has disease ✓
- **FP (False Positive)** -- model says disease, patient is actually healthy ✗ (false alarm)
- **TN (True Negative)** -- model says healthy, patient is actually healthy ✓
- **FN (False Negative)** -- model says healthy, patient actually has disease ✗ (missed it)

### specificity (bonus metric, not coded but good to know)

$$\text{Specificity} = \frac{TN}{TN + FP}$$

"of all the healthy people, how many were correctly identified as healthy?" its the opposite of recall -- recall is about catching sick people, specificity is about correctly leaving healthy people alone. also called **True Negative Rate (TNR)**.

### code for all metrics

```python
y_pred_lr = best_lr.predict(X_test_scaled)           # 0 or 1
y_pred_lr_proba = best_lr.predict_proba(X_test_scaled)[:, 1]  # probability of class 1

lr_accuracy = accuracy_score(y_test, y_pred_lr)
lr_precision = precision_score(y_test, y_pred_lr)
lr_recall = recall_score(y_test, y_pred_lr)
lr_f1 = f1_score(y_test, y_pred_lr)
lr_auc = roc_auc_score(y_test, y_pred_lr_proba)
```

`predict_proba(X)[:, 1]` -- gives probabilities. its a 2D array where column 0 = prob of class 0, column 1 = prob of class 1. column 1 gets grabbed with `[:, 1]`.

### classification_report

prints all the metrics in a nice formatted table. shows precision, recall, f1 for each class and also the averages.

the averages section shows:
- **macro avg** -- just the plain average of each class's metric. treats both classes equally regardless of size.
- **weighted avg** -- weighted by number of samples in each class. more representative when classes are imbalanced.
- **support** -- how many samples are in each class.

```python
print(classification_report(y_test, y_pred_lr))
```

---

## 8. confusion matrix

a 2x2 table that shows exactly where the model got it right and where it messed up.

```
              Predicted 0    Predicted 1
Actual 0        TN              FP
Actual 1        FN              TP
```

- top-left (TN) and bottom-right (TP) = correct predictions
- top-right (FP) and bottom-left (FN) = mistakes

all the metrics (accuracy, precision, recall, f1) can be calculated from just these 4 numbers. its the source of truth.

```python
cm_lr = confusion_matrix(y_test, y_pred_lr)
sns.heatmap(cm_lr, annot=True, fmt='d', cmap='Blues')
plt.title('LR Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.savefig('obtained_fig/lr_confusion_matrix.png', bbox_inches='tight')
plt.show()
```

- `fmt='d'` -- format as integers (whole numbers)
- `cmap='Blues'` -- blue color scheme. `'Greens'` was used for the ANN one

---

## 9. ANN (Artificial Neural Network)

### what is a neural network

inspired by the brain (loosely). basically layers of "neurons" connected to each other. each connection has a **weight**. data goes in, gets multiplied by weights, goes through an activation function, and comes out the other side as a prediction.

for this problem:
- **input layer**: 13 features (age, cholesterol, etc.)
- **hidden layers**: where the actual learning happens
- **output layer**: 1 number between 0 and 1 (probability of heart disease)

### weights and biases in a neural network

each connection between neurons has a **weight** (a number). each neuron also has a **bias** (another number). for a single neuron:

$$\text{output} = \text{activation}(w_1 x_1 + w_2 x_2 + ... + w_n x_n + b)$$

at the start of training, weights are initialized randomly (usually small random numbers). the training process adjusts them to minimize the loss. the total number of learnable parameters = all weights + all biases. `model.summary()` shows this count.

### Sequential model

means layers are stacked linearly -- output of one layer is input of the next. thats the simplest architecture.

```python
model = Sequential()
```

### Dense layer

a **Dense** (or fully connected) layer = every neuron in this layer is connected to every neuron in the previous layer. the number is how many neurons.

how many parameters does a Dense layer have?

$$\text{params} = (\text{inputs} \times \text{neurons}) + \text{neurons}$$

the first part is weights, the second part is biases. for the first layer: $(13 \times 64) + 64 = 896$ parameters.

```python
model.add(Dense(64, activation='relu', input_shape=(13,)))  # first hidden layer, 64 neurons
model.add(Dense(32, activation='relu'))                       # second hidden layer, 32 neurons
model.add(Dense(1, activation='sigmoid'))                     # output layer, 1 neuron
```

`input_shape=(13,)` -- only needed on the first layer. tells the network there are 13 input features.

### activation functions

an activation function decides whether a neuron should "fire" or not. without it, the network is just doing linear math (stacking linear transformations) and cant learn complex patterns. this is called the **linearity problem** -- no matter how many layers, without activation functions its all equivalent to a single linear layer.

**ReLU (Rectified Linear Unit):**
$$f(x) = \max(0, x)$$

if the value is negative, make it 0. if positive, keep it. simple and works great. used in hidden layers. advantages: fast to compute, doesnt have the vanishing gradient problem (unlike sigmoid in deep networks).

**Sigmoid:**
$$\sigma(x) = \frac{1}{1 + e^{-x}}$$

squishes any number into range (0, 1). used in the output layer for binary classification because a probability output is needed. the curve is S-shaped:
- very negative input → output near 0
- input = 0 → output = 0.5  
- very positive input → output near 1

### Dropout

a **regularization technique** to prevent overfitting. during training, it randomly "turns off" a percentage of neurons in that layer. this forces the network to not rely on any single neuron too much -- the knowledge gets distributed.

`Dropout(0.3)` = randomly ignore 30% of neurons in the previous layer during each training step. during prediction (testing), all neurons are used (but outputs are scaled down to compensate).

```python
model.add(Dropout(0.3))
```

### compiling the model

before training, the model needs to be compiled -- basically configuring how it learns.

```python
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)
```

**optimizer = 'adam'**
- the algorithm that updates the weights during training
- adam = **Ada**ptive **M**oment estimation
- it automatically adjusts the learning rate for each weight. basically the "smart" version of gradient descent
- almost always just used as the default go-to

**loss = 'binary_crossentropy' (in depth)**
- the function that measures how wrong the predictions are
- "binary" because there are 2 classes (disease / no disease)
- the model tries to minimize this number during training
- formula:

$$L = -\frac{1}{n}\sum_{i=1}^{n} [y_i \log(p_i) + (1 - y_i) \log(1 - p_i)]$$

where $y_i$ is the actual label (0 or 1) and $p_i$ is the predicted probability.

breaking it down:
- when actual = 1: loss = $-\log(p)$. if the model predicts p=0.99, loss = $-\log(0.99) \approx 0.01$ (good!). if p=0.01, loss = $-\log(0.01) \approx 4.6$ (terrible).
- when actual = 0: loss = $-\log(1-p)$. if model predicts p=0.01, loss ≈ 0.01 (good). if p=0.99, loss ≈ 4.6 (terrible).

so the loss function heavily punishes confident wrong predictions. thats the key idea.

why log? because probabilities are between 0 and 1, and multiplying small probabilities makes numbers vanishingly tiny. the log converts products to sums and stretches the scale so differences between 0.01 and 0.001 actually show up.

**metrics = ['accuracy']**
- just tracks accuracy during training so its visible. doesnt affect the training itself

### `model.summary()`

prints out the architecture -- how many layers, how many parameters (weights + biases) each layer has. useful to see the size of the model.

```python
model.summary()
```

### training the model (`.fit()`)

```python
history = model.fit(
    X_train_scaled, y_train,
    validation_split=0.2,
    epochs=50,
    batch_size=16
)
```

**epochs = 50**
- one epoch = one complete pass through the entire training dataset
- 50 epochs = the model sees all the training data 50 times
- more epochs = more learning but too many can cause overfitting (the model starts memorizing the training data instead of learning patterns)

**batch_size = 16**
- instead of updating weights after seeing ALL training samples (slow, called **batch gradient descent**), or after each single sample (noisy, called **stochastic gradient descent / SGD**), weights get updated after every 16 samples (called **mini-batch gradient descent**)
- its a compromise. smaller batch = noisier updates but can escape local minima. bigger batch = smoother updates but uses more memory
- common batch sizes: 16, 32, 64, 128

**local minima**: the loss function is a complex surface with hills and valleys. gradient descent tries to find the lowest point (global minimum) but can get stuck in a "dip" thats not actually the lowest point (local minimum). smaller batches add noise that can help "jump" out of local minima.

**validation_split = 0.2**
- takes 20% of the training data and uses it as a validation set
- the model trains on the remaining 80% and after each epoch checks performance on the validation 20%
- this lets the training progress be monitored -- if train loss goes down but val loss goes up, thats overfitting

**history**
- the `.fit()` function returns a history object that stores the loss and accuracy for each epoch
- these can be plotted to visualize training progress

### loss curves (training vs validation)

plotting how the loss changes over epochs. ideally both should go down together.

- if **training loss goes down but validation loss goes up** → **overfitting** (model memorized training data, doesnt generalize)
- if **both are high** → **underfitting** (model is too simple or needs more training)
- if **both go down and converge** → good (thats the goal)

**convergence** = when the loss stops changing significantly between epochs. the model has "settled" -- more training wont help much. if the losses are still dropping at epoch 50, more epochs might be beneficial.

```python
plt.plot(history.history['loss'], label='train loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Training vs Validation Loss')
plt.savefig('obtained_fig/training_loss.png', bbox_inches='tight')
plt.show()
```

### making predictions with the ANN

```python
y_pred_ann_proba = model.predict(X_test_scaled)
y_pred_ann_proba = y_pred_ann_proba.flatten()        # from 2D array to 1D
y_pred_ann = (y_pred_ann_proba > 0.5).astype(int)    # if prob > 0.5 → 1, else → 0
```

- `.predict()` gives probabilities (like 0.73 or 0.21)
- `.flatten()` -- the output is shape (61, 1) but (61,) is needed so its easier to work with
- `> 0.5` gives True/False, `.astype(int)` converts to 1/0

### threshold (the 0.5 cutoff)

the **threshold** is the cutoff probability for classification. default is 0.5:
- probability ≥ 0.5 → predict class 1 (disease)
- probability < 0.5 → predict class 0 (no disease)

but 0.5 isnt always the best choice. in medical diagnosis, maybe the threshold should be 0.3 -- better to flag more people and get some false positives than to miss someone who actually has heart disease. lowering the threshold increases recall but decreases precision (the precision-recall tradeoff thing from earlier).

---

## 10. AUC-ROC

### ROC curve (Receiver Operating Characteristic)

a graph that shows the tradeoff between:
- **True Positive Rate (TPR)** = recall = how many sick people were correctly identified
- **False Positive Rate (FPR)** = how many healthy people were wrongly flagged as sick

$$TPR = \frac{TP}{TP + FN} \quad \quad FPR = \frac{FP}{FP + TN}$$

the curve is made by trying EVERY possible threshold value (not just 0.5) and plotting TPR vs FPR for each. so its like asking "if the threshold was 0.1, what would TPR and FPR be? what about 0.2? 0.3?" and plotting all those points.

- a model that hugs the **top-left corner** is great (high TPR, low FPR)
- the **diagonal dashed line** represents random guessing (coin flip)
- anything below the diagonal is worse than random

### AUC (Area Under the Curve)

literally the area under the ROC curve. single number summary:
- **AUC = 1.0** → perfect model (all patients classified correctly at some threshold)
- **AUC = 0.5** → random guess (useless)
- **AUC < 0.5** → worse than random (the model is somehow backwards)

higher is better. generally: >0.9 = excellent, >0.8 = good, >0.7 = acceptable, <0.7 = meh

mathematically, AUC is the probability that the model ranks a randomly chosen positive sample higher than a randomly chosen negative sample. so AUC = 0.9 means theres a 90% chance the model assigns a higher probability to a person who actually has heart disease than to a healthy person.

```python
fpr_lr, tpr_lr, thresholds_lr = roc_curve(y_test, y_pred_lr_proba)
fpr_ann, tpr_ann, thresholds_ann = roc_curve(y_test, y_pred_ann_proba)

plt.plot(fpr_lr, tpr_lr, label='LR (AUC=' + str(round(lr_auc, 2)) + ')')
plt.plot(fpr_ann, tpr_ann, label='ANN (AUC=' + str(round(ann_auc, 2)) + ')')
plt.plot([0, 1], [0, 1], color='black', linestyle='--', label='random guess')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve Comparison')
plt.legend()
plt.savefig('obtained_fig/roc_curve_comparison.png', bbox_inches='tight')
plt.show()
```

`roc_curve()` returns three things: the FPR values, TPR values, and the thresholds used. only fpr and tpr are needed for plotting.

---

## 11. overfitting vs underfitting

this comes up everywhere so heres a proper explanation

### overfitting
- model learns the training data TOO well, including the noise and random fluctuations
- great accuracy on training data, bad on test data
- like memorizing answers for a specific practice exam instead of actually understanding the material
- common with complex models (deep neural nets) or small datasets
- **fixes**: dropout, regularization (l1/l2), more data, simpler model, early stopping

### underfitting
- model is too simple to capture the patterns in the data
- bad accuracy on BOTH training and test data
- like not studying at all
- **fixes**: more complex model, more features, train longer, remove too much regularization

the sweet spot is in between -- model learns the actual patterns without memorizing noise. this is called the **bias-variance tradeoff**:
- **high bias** = underfitting (model makes too many assumptions, oversimplifies)
- **high variance** = overfitting (model is too sensitive to training data, every little fluctuation changes predictions)

### early stopping (not used in this notebook but worth knowing)

a technique to stop training when the validation loss stops improving. instead of running for a fixed number of epochs, training stops early when the model starts overfitting. keras has a callback for this: `EarlyStopping(monitor='val_loss', patience=5)` -- stops if val_loss doesnt improve for 5 epochs.

---

## 12. gradient descent and backpropagation

### gradient descent

this is HOW models actually learn. during training:

1. model makes predictions (forward pass)
2. loss function calculates how wrong they are
3. **gradient descent** adjusts the weights slightly in the direction that reduces the loss
4. repeat

its like being on a mountain blindfolded trying to get to the lowest point. feeling the slope of the ground (the gradient) and taking a step downhill. keep doing this until (hopefully) the bottom is reached.

mathematically, the **gradient** is the partial derivative of the loss with respect to each weight. it tells the direction and steepness of the slope. the weight update rule:

$$w_{new} = w_{old} - \alpha \cdot \frac{\partial L}{\partial w}$$

where $\alpha$ is the **learning rate** (how big each step is) and $\frac{\partial L}{\partial w}$ is the gradient.

- learning rate too big → overshoots the minimum, bounces around, might diverge
- learning rate too small → takes forever to converge, might get stuck in local minima
- adam optimizer handles this automatically by adapting the learning rate for each parameter -- thats why its nice

### backpropagation

**backpropagation** (short for "backward propagation of errors") is the algorithm that calculates the gradients in a neural network. it works backwards from the output layer to the input layer using the **chain rule** from calculus.

the chain rule says: if $f(g(x))$ is a composition of functions, then $\frac{df}{dx} = \frac{df}{dg} \cdot \frac{dg}{dx}$

in a neural network, each layer is like a nested function. backpropagation applies the chain rule layer by layer, from the output back to the input, to figure out how much each weight contributed to the error. then gradient descent uses those gradients to update the weights.

so the training loop is:
1. **forward pass** -- data goes through the network, predictions come out
2. **compute loss** -- compare predictions to actual labels
3. **backward pass (backpropagation)** -- calculate gradients for every weight
4. **update weights** -- gradient descent adjusts weights to reduce loss
5. repeat for each batch in each epoch

---

## 13. other stuff used in the code

### `bbox_inches='tight'`

used in `plt.savefig()`. makes sure nothing gets cut off when saving the plot to a file. without it sometimes axis labels or titles get cropped. just always use it.

```python
plt.savefig('obtained_fig/something.png', bbox_inches='tight')
```

### `random_state=42`

a seed that makes "random" operations reproducible. using the same seed gives the same "random" result every time the code runs. useful for consistent results and reproducibility. 42 is from hitchhikers guide to the galaxy (the answer to life, the universe and everything).

### `.fit()` vs `.transform()` vs `.fit_transform()`

- **`.fit(data)`** -- learn the parameters from data (like calculating mean and std)
- **`.transform(data)`** -- apply those learned parameters to data
- **`.fit_transform(data)`** -- does both in one step (learn + apply)

rule: `fit_transform` on training data, only `transform` on test data. NEVER fit on test data.

### binary classification

classification = predicting a category (not a number). binary = only 2 categories. in this case: heart disease (1) or no heart disease (0).

vs **multiclass classification** = more than 2 categories (like predicting which type of animal).
vs **regression** = predicting a continuous number (like predicting house price).

### the dataset columns (for reference)

| column | what it is | type |
|--------|-----------|------|
| age | age in years | continuous |
| sex | 1 = male, 0 = female | categorical (binary) |
| cp | chest pain type (0-3) | categorical |
| trestbps | resting blood pressure | continuous |
| chol | cholesterol level | continuous |
| fbs | fasting blood sugar > 120 mg/dl (1 = yes, 0 = no) | categorical (binary) |
| restecg | resting ECG results (0, 1, 2) | categorical |
| thalach | max heart rate during exercise | continuous |
| exang | exercise induced chest pain (1 = yes, 0 = no) | categorical (binary) |
| oldpeak | ST depression from exercise vs rest | continuous |
| slope | slope of ST segment during exercise | categorical |
| ca | number of major blood vessels colored by fluoroscopy (0-3) | discrete |
| thal | blood disorder type (1 = normal, 2 = fixed defect, 3 = reversible) | categorical |
| target | 1 = heart disease, 0 = no disease | categorical (binary) |

**continuous** = can be any value in a range (like 120.5 blood pressure). **categorical** = fixed set of categories (like chest pain type 0, 1, 2, 3). **discrete** = countable whole numbers.

---

## 14. comparing models

a DataFrame with all the metrics for both models side by side. thats the proper way to compare -- dont just look at accuracy, compare everything.

```python
results = pd.DataFrame({
    'Model': ['Logistic Regression', 'Neural Network'],
    'Accuracy': [lr_accuracy, ann_accuracy],
    'Precision': [lr_precision, ann_precision],
    'Recall': [lr_recall, ann_recall],
    'F1': [lr_f1, ann_f1],
    'AUC': [lr_auc, ann_auc]
})
```

### why LR did as well as / better than the ANN here

- dataset is tiny (302 rows). neural networks need a LOT of data to shine. with small data they often overfit or just cant learn anything useful beyond what a simpler model already captures
- logistic regression is the go-to baseline for binary classification. its simple, interpretable, and works surprisingly well on small/medium datasets
- ANNs are better when theres complex non-linear patterns and lots of data
- this is actually a version of **Occam's Razor** -- simpler models are preferred when they perform just as well as complex ones

---

## 15. normal distribution (gaussian distribution)

worth knowing because StandardScaler works best when the data is roughly normally distributed, and a lot of statistical tests assume normality.

a **normal distribution** is the classic bell curve. most values cluster around the mean, and the further from the mean, the fewer values there are.

$$f(x) = \frac{1}{\sigma\sqrt{2\pi}} e^{-\frac{(x-\mu)^2}{2\sigma^2}}$$

where $\mu$ = mean (center of the bell) and $\sigma$ = standard deviation (width of the bell).

the **68-95-99.7 rule**: in a normal distribution:
- ~68% of values fall within 1 std of the mean
- ~95% fall within 2 stds
- ~99.7% fall within 3 stds

after StandardScaler, the data has mean=0 and std=1, which is called a **standard normal distribution**. so a z-score of 2 means "2 standard deviations above average."

---

ok thats basically everything. reading all of this should be enough to get through a presentation on this notebook.
