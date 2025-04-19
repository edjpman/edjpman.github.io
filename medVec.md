---
layout: page
title: Projects
permalink: /medVec/
---

<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/katex.min.css">
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/katex.min.js"></script>
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/contrib/auto-render.min.js" onload="renderMathInElement(document.body);"></script>


<style>
  body {
    font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
    font-size: 18px;
    line-height: 1.6;
  }
</style>


## MedVec: ClinicalBERT-Powered Vector Diagnosis Recommender



#### TLDR

If you had two minutes to read: What it does and what problem it solves, what is the data supposed to simulate, what modeling approach I took, what did I find, and how could it be made better


#### Datasets

Includes the description, link to download, and considerations/caveats (how to improve performance, what may cause issues, queries to explore)

{% raw %}
```python

from datasets import load_dataset
import pandas as pd


gretel_ds = load_dataset("gretelai/symptom_to_diagnosis")

mayo_ds = load_dataset("celikmus/mayo_clinic_symptoms_and_diseases_v1")


df_gretel = pd.DataFrame(gretel_ds['train'])
df_mayo = pd.DataFrame(mayo_ds['train'])


```




#### Modeling Approach


Describes the high-level approach taken with BERT 
Go into some of the technical details behind BERT and how this text becomes a sentence embedding
Also discuss some of the math/theory behind the sentence embedding in particular


Test of latex in MD file

$$
y = \beta_0 + \beta_1 x + \varepsilon
$$


#### Vector Behaviors

Discuss the dimensionality reduction method used to assess the data
Give and example input, show the nearest neighbors, and describe the nuance of what that means from the disease data and modeling technique in general



#### Diagnotic Potential

Describes the top 1/3/5 most similar method to determining a potential accuracy 




#### Future Considerations

Explain where this may create value, how it can't replace but rather guide human diagnosis, where it could be realistically applied
Also discuss methods that may improve the results of this sort of comparison


