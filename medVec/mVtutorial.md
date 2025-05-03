---
layout: page
title: Projects
permalink: /medVec/
---


<head>
  <link href="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/themes/prism.min.css" rel="stylesheet" />
  <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/KaTeX/0.16.9/katex.min.css" integrity="sha512-0dz2VvleUq65lUnDeOKxuvcZVRipCrL+6fK7DJQzE2G/5uWc6tS9kbx0OFT3ld86KnKH3HQW25Ds3o7t0V/Qq6g==" crossorigin="anonymous" referrerpolicy="no-referrer" />
</head>


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


<style>
pre code {
  background-color: #f4f4f4;
  padding: 1em;
  display: block;
  border-left: 4px solid #007acc;
  overflow-x: auto;
}

pre code .string {
  color: #d14; 
}
</style>



```python

from datasets import load_dataset
import pandas as pd


gretel_ds = load_dataset("gretelai/symptom_to_diagnosis")

mayo_ds = load_dataset("celikmus/mayo_clinic_symptoms_and_diseases_v1")


df_gretel = pd.DataFrame(gretel_ds['train'])
df_mayo = pd.DataFrame(mayo_ds['train'])


```


# Testing Latex

### Math Block

```math
f(x) = \int_{-\infty}^\infty
    \hat f(\xi)\,e^{2 \pi i \xi x}
    \,d\xi
```

### Inline

$E = mc^2$

## Display

$$
\int_0^\infty e^{-x^2} \, dx = \frac{\sqrt{\pi}}{2}
$$

<script src="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/prism.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/components/prism-python.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/KaTeX/0.16.9/katex.min.js" integrity="sha512-LQNxIMR5rXvG2Q8N0NOirqT0p8XEmj0MagVjrOx0Vbg9TUpW+nvz2DmH16u1yJaoP4w1DHn+U4K2G+3CuY/HjKQ==" crossorigin="anonymous" referrerpolicy="no-referrer"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/KaTeX/0.16.9/contrib/auto-render.min.js" integrity="sha512-43Wxk2/wrJ75zDP4nQuZ4rula0i0Q2gMFX9Y2YmsDDPMnk6zV+DYx0MGoHuw==/Iu9r+XCQ+pg==" crossorigin="anonymous" referrerpolicy="no-referrer"></script>
<script>
  document.addEventListener("DOMContentLoaded", function() {
    renderMathInElement(document.body, {
      delimiters: [
        { left: "$$", right: "$$", display: true },
        { left: "$", right: "$", display: false }
      ]
    });
  });
</script>

