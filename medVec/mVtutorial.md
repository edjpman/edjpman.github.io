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


In 2018 [BERT](https://arxiv.org/pdf/1810.04805) (Bidirectional Encoder Representations from Transformers) (cite) was developed as a method for quantitatively distinguishing these features. The general architecture consists of a series of steps to take text as an input and generate a vector of numerical features as an output that are in essence representing its context.


<img src="/assets/img/basic-bert-arch.png">


The following describes the process of generating BERT embeddings in the simplest manner possible. A full walkthrough with mathematical details will be provided in the future to multiple linked pages:

Before diving into some of the details, the architecture of the encoder side of the transformer proposed in the famous [paper](https://arxiv.org/pdf/1706.03762) “Attention is All You Need” (cite) will act as a guide to understand how BERT works.


<img src="/assets/img/detailed-bert-arch.png">


### Pretraining:


To train a BERT model on a corpus of text, the original [authors](https://arxiv.org/pdf/1810.04805) proposed two key methods. The first method called “Masked Language Modeling” describes a process of hiding a random 15% of the tokenized words to later be predicted, allowing BERT to learn word relationships. The second method, “Next Sentence Description”, highlights a process of pairing sentences together with 50% of pairs being actual consecutive sentences while the other 50% are randomly generated from the different parts of the corpus, allowing BERT to learn sentence relationships (cite book). 


#### Initial Embeddings: 

The text inputs are transformed into three separate embeddings that are finally summed together to be passed through the encoder. The first layer is of WordPiece tokens that break words into separate tokens, and even smaller chunks if the word does not exist. Finally a tag of [CLS] is added at the beginning of the body of text with a [SEP] tag between sentences to denote breaks.

To perform the tokenization words are compared with a vocabulary lookup table to note if a word or symbol exists. In BERT this comes to approximately 30,000 words (cite Devlin paper). As an example let's use the sentence “The apple is cold. It tasted good.” The would first be broken down into a list of [[CLS] , “The”, “apple”, “is”, “cold” ,“.”, [SEP], “It”, “tast”, “##ed”, “good”, “”. ]. Let assume that for the sake of the example the text chunk “##ed” did not exist in our vocabulary lookup table. The text would then be further split until a match is found; in our case [“##ed”] → [“##e”, “##d”].(cite hugging face) 

The other two layers that are part of the final input vector are the positional and segment embeddings. The positional embedding simply highlights the location of the token in the sequence of text (similar to an index), while the segment embedding denotes the location of the sentence within a sequence of text. All three of these embedding vectors are eventually made into one through an element wise sum. To illustrate please see the diagram below. (cite devlin paper)

***add devlin diagram here***


[Embedding image below here]


#### Multihead Attention: 

The inputs are then passed through the attention heads where the actual context is learned. The importance of this process can be exemplified by reusing the sentence in the beginning “The patient received the treatment and discharge paperwork and took it home to review.” To a human it may seem obvious that “it” in the sentence is referring to the “discharge paperwork” and not the actual treatment itself. However, to a computer this may not be so obvious. This brings forth the purpose of the attention heads, which is to place “attention” in various aspects of the sentence to which relationships and contextual nuance can be modeled.

The basic functionality of the attention heads are as follows. Let's assume a sentence of “The doctor prescribed the medication.” An attention head evaluates a sequence of tokens $\mathbf{X}_1, \mathbf{X}_2, \dots, \mathbf{X}_n$ where each $\mathbf{X}_i$ is the embedding vector for token $i$. For a given focus token at position $f$, the attention head will compute how much “attention” should be placed upon each previous token in the sequence $i \leq f$.

[attention head chart of sentence]
“The doctor prescribed the medication.”


During this process the model also has access to the current input along with all prior inputs. For instance when focusing on “medication” the attention head considers the focus token in relation to all previous tokens. That is “The”, “doctor”, “prescribed”, and “the”. The corresponding attention computation can be thought of as the weighted sum of all the token embeddings prior and up to the focus token: (cite textbook)

$$
\mathbf{A}_f = \sum_{i=1}^f \alpha_{fi} \mathbf{X}_i
$$

The notation $\alpha_{fi}$ is the scalar attention weight for token $i$ when the focus token is $f$, and all the weights sum to 1. 

This is completed through the dot product between the current focus vector $\mathbf{X}_f$ and each input vector $\mathbf{X}_i$. 

$$
\text{score}(\mathbf{X}_f, \mathbf{X}_i) = \mathbf{X}_f^\top \mathbf{X}_i
$$

The raw scores are then normalized through a softmax function:

$$
\alpha_{fi} = \text{softmax}(\text{score}(\mathbf{X}_f, \mathbf{X}_i)) = \frac{\exp(\mathbf{X}_f^\top \mathbf{X}_i)}{\sum_{j \leq f} \exp(\mathbf{X}_f^\top \mathbf{X}_j)}
$$


Using the word “medication” in our sentence as an example we would compute:

$$
\mathbf{X}_5^\top \mathbf{X}_1, \quad \mathbf{X}_5^\top \mathbf{X}_2, \quad \mathbf{X}_5^\top \mathbf{X}_3, \quad \mathbf{X}_5^\top \mathbf{X}_4, \quad \mathbf{X}_5^\top \mathbf{X}_5
$$


These scores are then passed through the softmax function to produce the attention weights part of the final attention adjusted embedding vector. This vector is of the same dimensions as the input vector.  

$$
\mathbf{A}_f = \sum_{i=1}^f \alpha_{fi} \mathbf{X}_i
$$

This describes just a single attention head in the process. BERT however includes multiple attention heads in the same layer to model different relational aspects of the input text. For additional mathematical details please reference this [paper](https://people.tamu.edu/~sji/classes/attn.pdf). (cite this paper) 


#### Layer Norm: 

After the inputs have passed through the multihead attention layer they are normalized through a z-score like method to improve their performance on deep learning based tasks as the values within a particular range support gradient based training. As with each layer of the encoder block, the normalization is applied to an embedding of a single token keeping the dimensionality the same, while adjusting for mean 0 and a standard deviation of 1. After the feedforward network portion, there is one final normalization layer before the final embedding is produced (cite devlin). 


#### Feed Forward: 

The last step of the encoder block (aside from a last normalization) is a 2-layer feedforward multi-layer perceptron neural network that maps the input embedding to a new vector that captures deeper and more abstract linguistic features. Across the different encoder blocks these networks learn different features such as parts of speech, grammatical roles, relationships of linguistic features, task specific functions, among others (cite devlin).


#### Inference BERT:

When using a pre-train version of BERT, the same architecture (tokenization, multi-head attention, normalization layers, feedforward networks) is used to produce deep contextual embeddings of the input text. The key difference with the inference step is that all input text is initialized with the token embeddings from the pre-trained model, allowing BERT to focus on optimizing the contextual representation of tokens based on the specific input, without needing to learn the embeddings again from scratch (cite devlin).



## Are Medical Professionals the Original BERT?

Until recent breakthroughs in technology a key part of the role as a medical professional has been to provide a communicative service with a patient where the provider uses natural language as a medium for determining a diagnosis. This essentially acts as a classification task in which the professional utilizes the features identified in natural language to help make a decision on what a diagnosis may be and how to treat it. 

To make the diagnosis, a provider leverages the knowledge obtained through education and practice to compare the natural language through a mental model to come to a conclusion. This however is not a fool-proof strategy in that the role as a medical provider is much more holistic than simply being a translator to make a diagnosis. 

As a result we cannot realistically expect human medical professionals to be able to pick up on all details of a patient's language to form a highly accurate diagnosis.


## BERT Based Replication

To help show that medical professional efforts are more impactful from a holistic and relational standpoint, I implement a supervised classification scenario on patient symptom descriptions that are contextually sparse. This follows known uses for the BERT architecture proposed in (cite devlin). The descriptions are transformed into embeddings utilizing a custom pre-trained BERT model named ClinicalBERT trained on over 3 million patient electronic health records. It's important to note that ClinicalBERT is technically a DistilBERT model which is a reduced form of BERT with half the amount of transformer encoding blocks but achieves nearly similar accuracy. For details please refer to the original DistilBERT paper (cite dbert paper) as details will be provided in a different page in the future. After the embeddings are created, a simple feed-forward multi-layer perceptron neural network is trained on the embeddings for further fine tuning and classification testing.


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

Optimizer: Per the original transformers [paper](https://arxiv.org/pdf/1706.03762) (cite), the Adam optimizer is standard by which the encoder-decoder is trained with and has benefits given its weight decay handling. While testing various optimization techniques the Adam based optimizers produced the most promising results. The AdamW optimizer was identified as the best performing with most other options producing substantially worse performance. 

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


## Citations

Paper citations, textbook, and hugging face data/models go here


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
