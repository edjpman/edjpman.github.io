---
layout: page
title: NFL Outcome Modeling
permalink: /nflModel/
---

# NFL Outcome Modeling

<div class="project-links">
  <a href="/" class="pill-link">Home</a>
  <a href="https://github.com/edjpman/nfl-modeling" class="pill-link">GitHub Repo</a>
</div>

##### TLDR

Predicting NFL game outcomes using classical machine learning — feature engineering on
play-by-play and game-level data.

---

##### Motivation

NFL outcome prediction is a well-studied problem that provides a clean testbed for the full
classical ML pipeline: structured tabular data, meaningful feature engineering, class imbalance
handling, and model calibration. The goal here is not to build a betting model but to explore
which statistical signals in play-by-play and game-level data are most predictive of outcome,
and how well classical algorithms can separate them.

---

##### Data and Features

The dataset is sourced from [nflfastR](https://www.nflfastr.com/), a play-by-play dataset
covering multiple NFL seasons. Game-level aggregations are constructed from raw play data
including:

- Detailed list...

<!-- IMAGE SLOT: Feature importance or correlation heatmap -->
<!-- Replace the comment below with: <img src="/assets/img/nfl_feature_importance.png" class="centered-image" style="max-width: 100%; height: auto;"> -->
*[Feature importance chart — add image here]*

---

##### Modeling Approach

Modleing approach here...

<!-- IMAGE SLOT: ROC curve comparison across models -->
*[ROC curve comparison — add image here]*

---

##### Calibration Analysis

A key consideration for any probabilistic prediction task is whether the model's confidence
scores are well-calibrated. A model that says 70% win probability should be right about 70%
of the time. Calibration curves (reliability diagrams) are computed for each model, and
post-hoc calibration via Platt scaling and isotonic regression is applied where needed.

<!-- IMAGE SLOT: Calibration curves / reliability diagram -->
*[Calibration curves — add image here]*

---

##### Results

<!-- IMAGE SLOT: Summary results table or bar chart of model performance -->
*[Results summary — add image here]*

A brief summary of key findings...

---

##### Limitations and Next Steps

- Brief summary of where things could go wrong...

---

<br />
