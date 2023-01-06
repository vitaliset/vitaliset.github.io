---
layout: post
title: Hyperparameter search with threshold-dependent metrics
featured-img: sleek
category: [basic, imbalanced learning]
mathjax: true
summary: It can be dangerous to do hyperparameter tunning with threshold-dependent metrics directly. Here we discuss why and how to work around it.
---

<p><div align="justify">In a binary classification problem, you probably shouldn&#39;t ever use the <code>.predict</code> method from scikit-learn (and consequently from libraries that follow <a href="https://scikit-learn.org/stable/developers/develop.html">its design pattern</a>). In scikit-learn, the implementation of <code>.predict</code>, in general, follows the logic <a href="https://github.com/scikit-learn/scikit-learn/blob/98cf537f5c538fdbc9d27b851cf03ce7611b8a48/sklearn/ensemble/_forest.py#L800-L837">implemented</a> for <a href="https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html"><code>sklearn.ensemble.RandomForestClassifier</code></a>:</div></p>

```python
def predict(self, X):
    ...
    proba = self.predict_proba(X)
    ...
    return self.classes_.take(np.argmax(proba, axis=1), axis=0)
```

<p><div align="justify">In the case where we only have two classes (0 or 1), the <code>.predict</code>, when picking the class with the highest &quot;probability&quot;, is equivalent to the rule &quot; if <code>.predict_proba &gt; 0.5</code>, then predict <code>1</code>; otherwise, predict <code>0</code>&quot;. That is, under the hood, we are using a threshold of <code>0.5</code> without having visibility.</div></p>

<p><div align="justify">Up to now, nothing new. However, I will show in an example how this can be harmful to superficial analyzes that don&#39;t take this into account.</div></p>

___

## Optimizing f1 in a naive way

<p><div align="justify">To exemplify this issue, I will use a dataset from <a href="https://imbalanced-learn.org/stable/">imbalanced-learn</a>, a library with several implementations of techniques that deal with imbalanced problems, from the <a href="https://github.com/scikit-learn-contrib">scikit-learn-contrib</a> environment. And let&#39;s build a model that ideally has the best possible <a href="https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html"><code>sklearn.metrics.f1_score</code></a>.</div></p>

```python
from imblearn.datasets import fetch_datasets

dataset = fetch_datasets()["coil_2000"]
X, y = dataset.data, (dataset.target==1).astype(int)

print(f"Percentage of y=1 is {np.round(y.mean(), 5)*100}%.")
print(f"Number of rows is {X.shape[0]}.")
```

    Percentage of y=1 is 5.966%.
    Number of rows is 9822.

<p><div align="justify">I&#39;m going to divide the dataset (taking care of the stratification because we are in an imbalanced problem) into a part for training the model, a second for choosing the threshold, and a last one for validation. We will not be dealing with the second part for now, but I will show some ways of optimizing the threshold that will need this extra set.</div></p>

```python
from sklearn.model_selection import train_test_split

X_train_model, X_test, y_train_model, y_test = train_test_split(X, y, random_state=0, stratify=y)
X_train_model, X_train_threshold, y_train_model, y_train_threshold = \
train_test_split(X_train_model, y_train_model, random_state=0, stratify=y_train_model)
```

<p><div align="justify">Suppose we want to optimize the hyperparameters of a <a href="https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html"><code>sklearn.ensemble.RandomForestClassifier</code></a> getting the best possible <a href="https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html"><code>sklearn.metrics.f1_score</code></a> (as we anticipated just now).</div></p>

<p><div align="justify">I&#39;m going to create an auxiliary function to run this search for hyperparameters because we&#39;re going to do this sometimes (using a <a href="https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html"><code>sklearn.model_selection.GridSearchCV</code></a>, but it could be any other way to search for hyperparameters).</div></p>

```python
from sklearn.model_selection import StratifiedKFold

params = {
    "max_depth": [2, 4, 10, None],
    "n_estimators": [10, 50, 100],
}

skfold = StratifiedKFold(n_splits=3,
                         shuffle=True,
                         random_state=0)
```

```python
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

def run_experiment(estimator, scoring, X, y, params, cv):
    gscv = (
        GridSearchCV(estimator=estimator,
                     param_grid=params,
                     scoring=scoring,
                     cv=cv)
        .fit(X, y)
    )

    return (
        pd.DataFrame(gscv.cv_results_)
        .pipe(lambda df:
              df[list(map(lambda x: "param_" + x,  params.keys())) + ["mean_test_score", "std_test_score"]])
    )
```

<p><div align="justify">With this auxiliary function built, we run our search trying to optimize <code>scoring=&quot;f1&quot;</code>.</div></p>

```python
run_experiment(estimator=RandomForestClassifier(random_state=0),
               scoring="f1", X=X_train_model, y=y_train_model, params=params, cv=skfold)
```

<p><center><img src="{{ site.baseurl }}/assets/img/threshold_dependent_opt/output_10_0.png"></center></p>

<p><div align="justify">Some combinations of hyperparameters seem to have an <a href="https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html"><code>sklearn.metrics.f1_score</code></a> of 0. Weird.</div></p>

<p><div align="justify">This happens because as <a href="https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html"><code>sklearn.metrics.f1_score</code></a> is a threshold-dependent metric (in the sense that it needs hard predictions instead of predicted probabilities), scikit-learn understands that it needs to use <code>.predict</code> instead of <code>.predict_proba</code> (and consequently &quot;uses the threshold of <code>0.5</code>&quot;, as we discussed the equivalence earlier).</div></p>

<p><div align="justify">As our problem is imbalanced, a threshold of <code>0.5</code> usually is suboptimal. And that&#39;s the case. We will have a considerable accumulation of <code>.predict_proba</code> close to 0 in almost any model, and, probably, a threshold closer to <code>0</code> in our problem seems more reasonable.</div></p>

```python
from collections import Counter
out_of_the_box_model = RandomForestClassifier(random_state=0).fit(X_train_model, y_train_model)

predict_proba = out_of_the_box_model.predict_proba(X_train_threshold)[:, 1]
predict = out_of_the_box_model.predict(X_train_threshold)

# Just to check. ;)
assert ((predict_proba > 0.5).astype(int) == predict).all()

fig, ax = plt.subplots(ncols=2, figsize=(6, 2.5))

ax[0].hist(predict_proba, bins=np.linspace(0, 1, 26))
ax[0].set_title("Histogram of .predict_proba(X)", fontsize=10)

count_predict = Counter(predict)
ax[1].bar(count_predict.keys(), count_predict.values(), label=".predict(X)", width=0.4)
count_y = Counter(y_train_threshold)
ax[1].bar(np.array(list(count_y.keys())) + 0.4, count_y.values(), label="y", width=0.4)
ax[1].set_xticks([0.2, 1.2])
ax[1].set_xticklabels([0, 1])
ax[1].tick_params(bottom = False)
ax[1].set_yscale("log")
ax[1].set_title("Count of 0's and 1's", fontsize=10)
ax[1].legend(fontsize=7)

plt.tight_layout()
```

<p><center><img src="{{ site.baseurl }}/assets/img/threshold_dependent_opt/output_13_0.png"></center></p>

<p><div align="justify">Very few examples pass the <code>0.5</code> threshold, a significantly lower amount than the actual number of class 1 samples. This tells us that a softer threshold (less than <code>0.5</code>) makes more sense in this problem.</div></p>

<p><div align="justify">This is often the case in imbalanced learning scenarios. For instance, if you have 1% of people with some disease in your population and your model predicts that this person has a 10% chance of having that disease, then chances are that you should treat him as someone with a high likelihood of being ill.</div></p>

___

## Tuning the threshold

<p><div align="justify">To find the optimal threshold, we can <a href="https://hastie.su.domains/ISLR2/ISLRv2_website.pdf">bootstrap</a> a set separate from the one used in training to find the best threshold for that model by optimizing some metric (threshold-dependent) of interest, such as, in our case, <a href="https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html"><code>sklearn.metrics.f1_score</code></a>.</div></p>

```python
from tqdm import tqdm

def optmize_threshold_metric(model, X, y, metric, threshold_grid, n_bootstrap=20):
    metric_means, metric_stds = [], []
    for t in tqdm(threshold_grid):
        metrics = []
        for i in range(n_bootstrap):
            ind_bootstrap = np.random.RandomState(i).choice(len(y), len(y), replace=True)
            metric_val = metric(y[ind_bootstrap],
                          (model.predict_proba(X[ind_bootstrap])[:, 1] > t).astype(int))
            metrics.append(metric_val)
        metric_means.append(np.mean(metrics))
        metric_stds.append(np.std(metrics))

    metric_means, metric_stds = np.array(metric_means), np.array(metric_stds)
    best_threshold = threshold_grid[np.argmax(metric_means)]

    return metric_means, metric_stds, best_threshold
```

<p><div align="justify">For each threshold value, we estimate the mean of <a href="https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html"><code>sklearn.metrics.f1_score</code></a> that we expect to obtain with that choice if we run the experiment different times through the bootstrap and the standard deviation to get an idea of the variance of the <a href="https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html"><code>sklearn.metrics.f1_score</code></a> we got. We chose the final threshold as the one with the best-estimated <a href="https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html"><code>sklearn.metrics.f1_score</code></a>.</div></p>

```python
threshold_grid = np.linspace(0, 1, 101)
from sklearn.metrics import f1_score

f1_means_ootb, f1_stds_ootb, best_threshold_ootb = \
optmize_threshold_metric(out_of_the_box_model, X_train_threshold, y_train_threshold, f1_score, threshold_grid)

fig, ax = plt.subplots(figsize=(5, 2.5))
ax.plot(threshold_grid, f1_means_ootb)
ax.fill_between(threshold_grid, f1_means_ootb - 1.96 * f1_stds_ootb, f1_means_ootb + 1.96 * f1_stds_ootb, alpha=0.5)
ax.vlines(best_threshold_ootb, min(f1_means_ootb - 1.96 * f1_stds_ootb), max(f1_means_ootb + 1.96 * f1_stds_ootb), "k", label="Chosen threshold")
ax.set_xticks(np.linspace(0, 1, 11))
ax.set_xlabel("Threshold")
ax.set_ylabel("f1_score")
ax.legend()
plt.tight_layout()
```

    100%|██████████| 101/101 [02:00<00:00,  1.19s/it]

<p><center><img src="{{ site.baseurl }}/assets/img/threshold_dependent_opt/output_18_1.png"></center></p>

```python
f1_score(y_test, (out_of_the_box_model.predict_proba(X_test)[:, 1] > best_threshold_ootb).astype(int))
```

    0.1878453038674033

```python
f1_score(y_test, out_of_the_box_model.predict(X_test))
```

    0.043478260869565216

<p><div align="justify">With the threshold chosen through optimization, we ended up with a much better <a href="https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html"><code>sklearn.metrics.f1_score</code></a> than the one we get with <code>.predict</code>, with the <code>0.5</code> threshold.</div></p>

<p><div align="justify">$\oint$ <em>Here we are directly choosing the threshold that, on average, has the best metric value of interest, but there are other possibilities [<a href="#bibliography">1</a>]. We could, for example, play with the &quot;confidence interval&quot; (which, in this case, I&#39;m just plotting to give an order of magnitude of the variance), optimizing for the upper or lower limit, or even use the threshold that maximizes <a href="https://en.wikipedia.org/wiki/Youden%27s_J_statistic">Youden&#39;s J statistic</a> (which is equivalent to taking the threshold that gives the most significant separation of the KS curves between the <code>.predict_proba(X[y==0])</code> and <code>.predict_proba(X[y==1])</code>.</em></div></p>

___

## Back to hyperparameters search

<p><div align="justify">But what to do now? How can we get around this if optimizing the <a href="https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html"><code>sklearn.metrics.f1_score</code></a> directly doesn&#39;t look like a good idea since scikit-learn will use <code>.predict</code>? We will discuss three possibilities of how to get around this issue. One case is not necessarily better than the other, and the idea is to show some options for facing the problem.</div></p>

### 1. Optimizing a metric that works and is related to the desired metric

<p><div align="justify">The most applied way in the market is, even if you are interested in the threshold-dependent metric, to use a threshold-independent metric to do this optimization and only, in the end, use something like <code>optmize_threshold_metric</code> to optimize the metric of genuine interest.</div></p>

<p><div align="justify">$\oint$ <em>This sounds sub-optimal, but we do this all the time in Machine Learning. Even if you&#39;re interested in optimizing <a href="https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html"><code>sklearn.metrics.roc_auc_score</code></a> on a credit default classification problem, your <a href="https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html"><code>sklearn.ensemble.RandomForestClassifier</code></a> will be optimizing for <code>criterion=&quot;gini&quot;</code> or something related to <a href="https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html"><code>sklearn.metrics.roc_auc_score</code></a>, but that is different. Here the idea is the same. Optimizing for <a href="https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html"><code>sklearn.metrics.roc_auc_score</code></a> or <a href="https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html"><code>sklearn.metrics.average_precision_score</code></a> is not the same as optimizing for <a href="https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html"><code>sklearn.metrics.f1_score</code></a>, for example, but models that are good at the former will be good at the latter too.</em></div></p>

```python
run_experiment(estimator=RandomForestClassifier(random_state=0),
               scoring="roc_auc", X=X_train_model, y=y_train_model, params=params, cv=skfold)
```

<p><center><img src="{{ site.baseurl }}/assets/img/threshold_dependent_opt/output_24_0.png"></center></p>

### 2. Leak the threshold search

<p><div align="justify">But what if we want to explicitly optimize our interest metric within the grid search for some reason? In that case, we need to make a bigger workaround. A reasonable proxy of how your model will perform when you optimize the threshold is to optimize the threshold on your test set. In this case, as you will choose the threshold that will optimize the metric in the validation set, your metric will be the best possible, and you can directly take the <code>max</code> or the <code>min</code>.</div></p>

```python
from sklearn.metrics import make_scorer

def make_threshold_independent(metric, threshold_grid=np.linspace(0, 1, 101), greater_is_better=True):
    opt_fun = {True: max, False: min}
    opt = opt_fun[greater_is_better]
    def threshold_independent_metric(y_true, y_pred, *args, **kwargs):
        return opt([metric(y_true, (y_pred > t).astype(int), *args, **kwargs) for t in threshold_grid])
    return threshold_independent_metric

f1_threshold_independent_score = make_threshold_independent(f1_score)
f1_threshold_independent_scorer = make_scorer(f1_threshold_independent_score, needs_proba=True)
```

<p><div align="justify">As this is a threshold-independent metric (because we passed <code>needs_proba=True</code>), we will no longer have the problem of scikit-learn using <code>.predict</code>.</div></p>

```python
df_best_f1 = run_experiment(estimator=RandomForestClassifier(random_state=0),
                            scoring=f1_threshold_independent_scorer,
                            X=X_train_model, y=y_train_model, params=params, cv=skfold)

df_best_f1
```

<p><center><img src="{{ site.baseurl }}/assets/img/threshold_dependent_opt/output_28_0.png"></center></p>

<p><div align="justify">On the other hand, we are leaking our model and consequently overestimating our metric since we are choosing the best threshold in the cross-validation validation set.</div></p>

### 3. Tuning the threshold during gridsearch on a chunk of the training set

<p><div align="justify">A better way to do this (in terms of correctly evaluating the performance during cross-validation) is to modify our estimator&#39;s training function so that it also calculates the best threshold.  To clarify what we are doing without having to look at the class details I will implement, it is worth comparing the difference between the method 2 and method 3.</div></p>

<p><div align="justify">In each step of our cross-validation, we will have a training set and a set validation that we will use to evaluate the performance of the classifier trained in that training set. That is what we were doing during method 1, for instance.</div></p>

<p><center><img src="{{ site.baseurl }}/assets/img/threshold_dependent_opt/output_30_0.png"></center></p>

<p><div align="justify">In solution 2, we optimize the threshold on the validation set by taking the best possible metric value for the different thresholds of our threshold grid. But, as we are leaking the threshold search, we will overestimate our metric, which can be harmful.</div></p>

<p><center><img src="{{ site.baseurl }}/assets/img/threshold_dependent_opt/output_32_0.png"></center></p>

<p><div align="justify">In the solution we are discussing, during the training stage, we will do a hold-out to have a set that we will use to optimize the threshold, and the optimal threshold will be used in the validation evaluation.</div></p>

<p><center><img src="{{ site.baseurl }}/assets/img/threshold_dependent_opt/output_34_0.png"></center></p>

<p><div align="justify">A rough implementation of a class that does this logic is as follows:</div></p>

```python
import inspect
def dic_without_keys(dic, keys):
    return {x: dic[x] for x in dic if x not in keys}

class ThresholdOptimizerRandomForestBinaryClassifier(RandomForestClassifier):

    def __init__(self, n_bootstrap=20, metric=f1_score, threshold_grid=np.linspace(0, 1, 101), *args, **kwargs,):

        kwargs_without_extra = dic_without_keys(kwargs, ("n_bootstrap", "metric", "threshold_grid"))
        super().__init__(*args, **kwargs_without_extra)
        self.metric = metric
        self.threshold_grid = threshold_grid
        self.n_bootstrap = n_bootstrap

    @classmethod
    def _get_param_names(cls):
        init = getattr(super().__init__, "deprecated_original", super().__init__)
        init_signature = inspect.signature(init)
        parameters = [p for p in init_signature.parameters.values() if p.name != "self" and p.kind != p.VAR_KEYWORD]
        return sorted([p.name for p in parameters] + ["n_bootstrap", "metric", "threshold_grid"])

    def fit(self, X, y, sample_weight=None):

        X_train_model, X_train_threshold, y_train_model, y_train_threshold = \
        train_test_split(X, y, random_state=self.random_state, stratify=y)

        super().fit(X_train_model, y_train_model, sample_weight=sample_weight)
        _, _, self.best_threshold_ = self.optmize_threshold_metric(X_train_threshold, y_train_threshold)

        return self

    def optmize_threshold_metric(self, X, y):
        metric_means, metric_stds = [], []
        for t in self.threshold_grid:
            metrics = []
            for i in range(self.n_bootstrap):
                ind_bootstrap = np.random.RandomState(i).choice(len(y), len(y), replace=True)
                metric_val = self.metric(y[ind_bootstrap],
                                         (self.predict_proba(X[ind_bootstrap])[:, 1] > t).astype(int))
                metrics.append(metric_val)
            metric_means.append(np.mean(metrics))
            metric_stds.append(np.std(metrics))

        metric_means, metric_stds = np.array(metric_means), np.array(metric_stds)
        best_threshold = self.threshold_grid[np.argmax(metric_means)]

        return metric_means, metric_stds, best_threshold

    def predict(self, X):
        preds = self.predict_proba(X)[:, 1]
        return (preds > self.best_threshold_).astype(int)
```

<p><div align="justify">$\oint$ <em><a href="https://scikit-learn.org/stable/developers/develop.html#instantiation">scikit-learn doesn&#39;t like you using <code>args</code> and <code>kwargs</code> on your estimator&#39;s <code>init</code></a> because of how they designed the way they deal with hyperparameter optimization. But as I didn&#39;t want my <code>init</code> to <a href="https://github.com/vitaliset/blog-notebooks/blob/700ce6ed6f61952db3f54c2f149fda5fc36175e9/DataLab_Blog_Boruta_2022_09_05/shap_feature_importances_.py#L52-L71">look like this</a>, I decided to change the <a href="https://github.com/scikit-learn/scikit-learn/blob/98cf537f5c538fdbc9d27b851cf03ce7611b8a48/sklearn/base.py#L122-L151"><code>_get_param_names</code></a> from the <a href="https://scikit-learn.org/stable/modules/generated/sklearn.base.BaseEstimator.html"><code>sklearn.base.BaseEstimator</code></a> to call only the parameters of the class I&#39;m inheriting from (<a href="https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html"><code>sklearn.ensemble.RandomForestClassifier</code></a>, a.k.a. <code>super()</code>). If you want to design it properly, you should do <a href="https://github.com/vitaliset/blog-notebooks/blob/700ce6ed6f61952db3f54c2f149fda5fc36175e9/DataLab_Blog_Boruta_2022_09_05/shap_feature_importances_.py#L52-L71">this</a>.</em></div></p>

<p><div align="justify">$\oint$ <em>Note that although I&#39;m inheriting from <a href="https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html"><code>sklearn.ensemble.RandomForestClassifier</code></a>, I don&#39;t use any <a href="https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html"><code>sklearn.ensemble.RandomForestClassifier</code></a>-specific logic here, and actually, you can do the same with any scikit-learn estimator.</em></div></p>

<p><div align="justify">We are basically using the same optimization function we had discussed earlier on the part of the set that is given in <code>.fit</code> by doing a <a href="https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html"><code>sklearn.model_selection.train_test_split</code></a>. This implementation is computationally expensive, mainly because of bootstrap. So I lowered the number of bootstrap samples to make it faster.</div></p>

```python
%%time

df_best = run_experiment(
    estimator=ThresholdOptimizerRandomForestBinaryClassifier(random_state=0, n_bootstrap=5,
                                                             metric=f1_score, threshold_grid=threshold_grid),
    scoring="f1", X=X_train_model, y=y_train_model, params=params, cv=skfold)

df_best
```

    CPU times: total: 5min 25s
    Wall time: 5min 28s

<p><center><img src="{{ site.baseurl }}/assets/img/threshold_dependent_opt/output_39_1.png"></center></p>

___

## Tuning the threshold for the best hyperparameters combination

<p><div align="justify">With this best combination of hyperparameters of method 3 chosen, we can do the procedure we discussed earlier to find the best threshold for this model.</div></p>

```python
best_params_values = df_best.sort_values("mean_test_score", ascending=False).iloc[0][list(map(lambda x: "param_" + x,  params.keys()))].values
best_params = dict(zip(params.keys(), best_params_values))
best_params
```

    {'max_depth': 4, 'n_estimators': 100}

```python
best_model = (
    RandomForestClassifier(random_state=0)
    .set_params(**best_params)
    .fit(X_train_model, y_train_model)
)
```

```python
f1_means_best, f1_stds_best, best_threshold_best = \
optmize_threshold_metric(best_model, X_train_threshold, y_train_threshold, f1_score, threshold_grid)

fig, ax = plt.subplots(figsize=(5, 2.5))
ax.plot(threshold_grid, f1_means_best)
ax.fill_between(threshold_grid, f1_means_best - 1.96 * f1_stds_best, f1_means_best + 1.96 * f1_stds_best, alpha=0.5)
ax.vlines(best_threshold_best, min(f1_means_best - 1.96 * f1_stds_best), max(f1_means_best + 1.96 * f1_stds_best), "k", label="Chosen threshold")
ax.set_xticks(np.linspace(0, 1, 11))
ax.set_xlabel("Threshold")
ax.set_ylabel("f1_score")
ax.legend()
plt.tight_layout()
```

    100%|██████████| 101/101 [01:13<00:00,  1.37it/s]

<p><center><img src="{{ site.baseurl }}/assets/img/threshold_dependent_opt/output_43_1.png"></center></p>

```python
f1_score(y_test, (best_model.predict_proba(X_test)[:, 1] > best_threshold_best).astype(int))
```

    0.24038461538461534

```python
f1_score(y_test, best_model.predict(X_test))
```

    0.0

<p><div align="justify">Notice that we got a much better <a href="https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html"><code>sklearn.metrics.f1_score</code></a> than the initial search was telling us we would get!</div></p>

___

## tl;dr

<p><div align="justify">When optimizing hyperparameters, threshold-dependent metrics make <a href="https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html"><code>sklearn.model_selection.GridSearchCV</code></a> like search methods use the estimator&#39;s <code>.predict</code> method instead of <code>.predict_proba</code>. This can be harmful as <code>0.5</code> might not be the best threshold, especially in imbalanced learning scenarios.</div></p>

<p><div align="justify">Always prioritize the threshold-independent metrics, but if you need to use a threshold-dependent metric, you can try to make it threshold-independent by getting the optimal value for it (<code>max</code> or <code>min</code> depending on if <code>greater_is_better=True</code> or <code>False</code>) for a threshold grid of options. As this is the same as optimizing it for the validation set, it can slightly overestimate your results.</div></p>

<p><div align="justify">A more honest way to do this is to explicitly optimize the threshold on a part of your training set for each cross-validation fold. This mimics reality better but is more time-consuming as this optimization takes time if you want it to be robust (for instance, using bootstrap to estimate better the performance value).</div></p>

## <a name="bibliography">Bibliography</a>

<p><div align="justify">[1] <a href="https://machinelearningmastery.com/threshold-moving-for-imbalanced-classification/">A Gentle Introduction to Threshold-Moving for Imbalanced Classification by Jason Brownlee.</a></div></p>

___

<p><div align="justify">You can find all files and environments for reproducing the experiments in the <a href="https://github.com/vitaliset/blog-notebooks/tree/main/Blog_Threshold_Dependent_Opt_2023_01_06">repository of this post</a>.</div></p>