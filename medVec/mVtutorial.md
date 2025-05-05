---
layout: page
title: Projects
permalink: /medVec/
---


<head>
  <link href="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/themes/prism.min.css" rel="stylesheet" />
</head>


<style>
  body {
    font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
    font-size: 18px;
    line-height: 1.6;
  }
</style>


# MedVec: ClinicalBERT-Powered Vector Diagnosis Recommender



## TLDR - 2 minute summary

Since the existence of medicine, medical professional diagnostic tasks have been tasked with leveraging the professional’s formal education and clinical experience to make sense of what a patient is describing and comparing with their knowledge/experience in a mental model framework to determine the likely cause of illness. 

From a purely language based lense, encoder architectures coupled with classification algorithms act as a powerful tool in replicating the process currently performed by medical professionals. Encoders trained on a rich medical corpus can provide contextual nuance that make it possible for classifying diagnoses at the level of practitioners from linguistic features only. 

Results of the modeling run reached an accuracy score of XX%, recall of XX%, precision of XX%, and F1 score of XX%. Future considerations of this library will include unsupervised techniques for evaluation and classification, additional tools for prospective medical professionals to gain AI literacy in their field, and potentially a performance evaluation in classification between medVec and medical student subjects. 




## Power of Encoders

Despite seeming second nature to most, human language is infinitely complex in its use, rule sets, and presentation. In each domain language can have vastly different meanings and thus requires a way to differentiate. For example, in the medical field the word “discharge” may have two completely opposite meanings yet be used frequently; “The patient had a green nasal discharge” as in an anatomical descriptor and “The patient received the treatment and discharge paperwork and took it home to review” as is an organizational processing step.  


In 2018 BERT (Bidirectional Encoder Representations from Transformers) (cite) was developed as a method for quantitatively distinguishing these features. The general architecture consists of a series of steps to take text as an input and generate a vector of numerical features as an output that are in essence representing its context.


***bert image here***

![Alt text](assets/img/basic-bert-arch.png)



The following describes the process of generating BERT embeddings in the simplest manner possible. A full walkthrough with mathematical details will be provided in the future to a linked page:

Before diving into some of the details, the architecture of the encoder side of the transformer proposed in the famous paper “Attention is All You Need” (cite) will act as a guide to understand exactly how BERT works.


***bert image detailed here***

![Alt text](assets/img/detailed-bert-arch.png)


Additional description here....


## Are Medical Professionals the Original BERT?

Until recent breakthroughs in technology a key part of the role as a medical professional has been to provide a communicative service with a patient where the provider uses natural language as a medium for determining a diagnosis. This essentially acts as a classification task in which the professional utilizes the features identified in natural language to help make a decision on what a diagnosis may be and how to treat it. 

To make the diagnosis, a provider leverages the knowledge obtained through education and practice to compare the natural language through a mental model to come to a conclusion. This however is not a fool-proof strategy in that the role as a medical provider is much more holistic than simply being a translator to make a diagnosis. 

As a result we cannot realistically expect human medical professionals to be able to pick up on all details of a patient's language to form a highly accurate diagnosis.


## BERT Based Replication

To prove that medical professional efforts are more impactful from a holistic and relational standpoint, I implement a supervised classification scenario on patient symptom descriptions that are contextually sparse. This follows known uses for the BERT architecture proposed in (cite paper). The descriptions are transformed into embeddings utilizing a custom pre-trained BERT model named ClinicalBERT trained on over 3 million patient electronic health records. It's important to note that ClinicalBERT is technically a DistilBERT model which is a reduced form of BERT with half the amount of transformer encoding blocks but achieves nearly similar accuracy. For details please refer to the original DistilBERT paper (cite) as details will be provided in a different page in the future. After the embeddings are created, a simple feed-forward multi-layer perceptron neural network is trained on the embeddings for further fine tuning and classification testing.


### Dataset: 
The dataset of patient descriptions and their labels are loaded in from HuggingFace and processed removing any URLs, citations and associated marks, and any leftover HTML tagging. The data labels are then converted into numerical categories for proper data transformations later in the modeling process.

```python

dataproc = mvPreproc()
patdf, meddf = dataproc.ds_load()

patdf['text'] = patdf['text'].apply(dataproc.clean_text)

labels = {
    'allergy': 0,
    'arthritis': 1,
    'bronchial asthma': 2,
    'cervical spondylosis': 3,
    'chicken pox': 4,
    'common cold': 5,
    'dengue': 6,
    'diabetes': 7,
    'drug reaction': 8,
    'fungal infection': 9,
    'gastroesophageal reflux disease': 10,
    'hypertension': 11,
    'impetigo': 12,
    'jaundice': 13,
    'malaria': 14,
    'migraine': 15,
    'peptic ulcer disease': 16,
    'pneumonia': 17,
    'psoriasis': 18,
    'typhoid': 19,
    'urinary tract infection': 20,
    'varicose veins': 21
}


patdf = dataproc.feat_map(dataset=patdf,label_map=labels,col='label')

```


### Data Transformations: 



```python

bert_instance = cBERTbase(inputs=None)

X_train, X_dev, X_test, y_train, y_dev, y_test = bert_instance.ttd_splits(
    dataset=patdf,
    x_col='text',
    y_col='label'
)


embd_splits = bert_instance.mv_tokenizer(
    X_train, X_dev, X_test,
    y_train, y_dev, y_test,
    repr='pooled'
)


train_data = BERTdataset(embd_splits['train']['embeddings'], embd_splits['train']['labels'])
dev_data   = BERTdataset(embd_splits['dev']['embeddings'],   embd_splits['dev']['labels'])
test_data  = BERTdataset(embd_splits['test']['embeddings'],  embd_splits['test']['labels'])

```


### Classification Head Architecture: 



```python

train_loader = DataLoader(train_data, batch_size=4, shuffle=True)
dev_loader   = DataLoader(dev_data, batch_size=4, shuffle=False)
test_loader  = DataLoader(test_data, batch_size=4, shuffle=False)

classification_head = cBertMCChead(input_dim=768, hidden_dim=256, dropout=0.3, num_layers=1, num_classes=22)
tuner = sftTune(model=classification_head, learn_rate=1e-4)
optimizer = tuner.adamW()
loss_fn = tuner.xc_entropy()

```




### Train-Tune-Test: 


```python

trainer = cBERT_train()

for epoch in range(60):
    print(f"Epoch {epoch+1}")
    trainer.train_loop(train_loader, classification_head, loss_fn, optimizer, device='cpu')

tuning = Bert_test()
tuning.test_loop(dev_loader, classification_head, loss_fn)

tester = Bert_test()
tester.test_loop(test_loader, classification_head, loss_fn)


```


## Classification Head Tuning

After creating the patient embeddings and performing the training and tuning of the classification head, the following resulted in the best performance. The model was specifically evaluated on pure accuracy, precision, recall, and F1 to provide a more holistic lens to performance: 

Network Architecture: There were virtually no gains to increasing the number of layers within the network architecture, and many times decreased the classification performance of the model. This can potentially be attributed to the level of abstraction that is achieved during the embedding process. 

Training Epochs: A number of training epochs 50 and above were necessary as the loss despite taking longer to converge at a minimum consistently decreased.

Optimizer: Per the original transformers paper, the Adam optimizer is standard by which the encoder-decoder is trained with and has benefits given its weight decay handling. While testing various optimization techniques the Adam based optimizers produced the most promising results. The AdamW optimizer was identified as the best performing with most other options producing substantially worse performance. 

Batch Sizes: The batch size of one ended up providing the best forming results. 



## Future Considerations


Text here





## Test Code

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

### Inline

$E = mc^2$

$\int_0^\infty e^{-x^2} dx = \frac{\sqrt{\pi}}{2}$

### Display


\[
\int_0^\infty e^{-x^2} \, dx = \frac{\sqrt{\pi}}{2}
\]


<script src="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/prism.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/components/prism-python.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.9/MathJax.js?config=TeX-AMS-MML_HTMLorMML"></script>
<script>
  MathJax.Hub.Config({
    tex2jax: {
      inlineMath: [['$','$']],
      displayMath: [['\\[','\\]'],['$$','$$']],
      processEscapes: true
    }
  });
</script>
