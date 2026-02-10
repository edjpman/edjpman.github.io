---
layout: page
title: Projects
permalink: /medVec/
---


<head>
  <link href="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/themes/prism.min.css" rel="stylesheet" />
</head>


<style>
  .wrapper {
    max-width: 900px !important; 
    margin: 0 auto !important;
    padding: 40px 24px !important;
  }
  
  body {
    font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
    font-size: 18px;
    line-height: 1.6;
  }

  .centered-image {
    display: block;
    margin: auto;
    max-width: 100%;
    height: auto
  }

  .project-links {
    margin: 12px 0 28px 0;
  }

  .pill-link {
    display: inline-block;
    margin-right: 12px;
    padding: 6px 14px;
    border-radius: 999px;
    text-decoration: none;
    font-size: 14px;
    font-weight: 500;
    color: #333;
    background-color: #f2f2f2;
    transition: background-color 0.2s ease, transform 0.1s ease;
  }

  .pill-link:hover {
    background-color: #e6e6e6;
    transform: translateY(-1px);
  }

  .callout {
    padding: 16px 20px;
    margin: 24px 0;
    border-left: 4px solid #0366d6; /* GitHub blue */
    background-color: #f6f8fa;
    border-radius: 6px;
    font-size: 16px;
    color: #24292e;
  }

  .callout-title {
    font-weight: 600;
    display: block;
    margin-bottom: 4px;
    color: #0366d6;
    text-transform: uppercase;
    font-size: 13px;
    letter-spacing: 0.5px;
  }
  
</style>


# MedVec: ClinicalBERT-Powered Vector Diagnosis Recommender

<div class="project-links">
  <a href="https://edjpman.github.io/" class="pill-link">Home</a>
  <a href="https://github.com/edjpman/medVec" class="pill-link">GitHub Repo</a>
</div>

## TLDR - 2 minute summary

Since the existence of medicine, medical professionals have been tasked with making sense of what a patient is describing by comparing it with their formal education and clinical experience in a mental model framework to determine the likely diagnosis. 

From a language based lens, encoder architectures coupled with classification algorithms act as a powerful tool in replicating the process currently performed by medical professionals. Encoders trained on a rich medical corpus can provide contextual nuance that make it possible for classifying diagnoses at the level of practitioners from linguistic features only. 

Results of the modeling run reached scores of aprroximately 80% for accuracy, 80% for precision, 80% for recall, 78% for F1. While these results are replicable between the tuning and testing set, there are concerns for overfitting given a dataset of only ~1,000. Therefore, a significantly larger dataset and additional evaluation methods are required to observe the true strength of BERT. Future considerations of this library will include unsupervised techniques for evaluation and classification, additional tools for prospective medical professionals to gain AI literacy in their field, and potentially a performance evaluation in classification between medVec and medical student subjects. 




## Power of Encoders

Despite seeming second nature to most, human language is infinitely complex in its use, rule sets, and presentation. In each domain, language can have vastly different meanings and thus requires a way to differentiate. For example, in the medical field the word “discharge” may have two completely opposite meanings yet both be used frequently; “The patient had a green nasal discharge” as in a bodily function descriptor and “The patient received the treatment and discharge paperwork and took it home to review” as is an organizational processing step.  


In 2018 [BERT](https://arxiv.org/pdf/1810.04805) (Bidirectional Encoder Representations from Transformers) (Devlin et al., 2019) was developed as a method for quantitatively distinguishing these features. The general architecture consists of a series of steps to take text as an input and generate a vector of numerical features as an output that are in essence representing its context.


<img src="/assets/img/basic-bert-arch.png" class="centered-image">


The following describes the process of generating BERT embeddings in the simplest manner possible. A full walkthrough of the architecture with mathematical details will be provided in the future in multiple linked pages:

Before diving into some of the details, the architecture of the encoder side of the transformer proposed in the famous [paper](https://arxiv.org/pdf/1706.03762) “Attention is All You Need” (Vaswani et al., 2023) will act as a guide to understand how BERT works.


<img src="/assets/img/detailed-bert-arch.png" class="centered-image">

<small>The Transformer – model architecture. Reprinted from Attention is all you need, by A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, Ł. Kaiser, & I. Polosukhin, 2023, arXiv, [Link](https://arxiv.org/abs/1706.03762)</small>

### Pretraining:

To train a BERT model on a corpus of text, the original authors proposed two key methods. The first method called “Masked Language Modeling” describes a process of hiding a random 15% of the tokenized words to later be predicted, allowing BERT to learn word relationships (Devlin et al., 2019). The second method, “Next Sentence Description”, highlights a process of pairing sentences together with 50% of pairs being actual consecutive sentences while the other 50% are randomly generated from the different parts of the corpus, allowing BERT to learn sentence relationships (Jurafsky & Martin, 2025). 


### Inference BERT:

When using a pre-trained version of BERT, the same architecture (tokenization, multi-head attention, normalization layers, feedforward networks) is used to produce deep contextual embeddings of the input text. The key difference with the inference step is that all input text is initialized with the token embeddings from the pre-trained model, allowing BERT to focus on optimizing the contextual representation of tokens based on the specific input, without needing to learn the embeddings again from scratch (Devlin et al., 2019).


#### Initial Embeddings: 

The text inputs to BERT are first transformed into three separate embeddings that are finally summed together to be passed through the encoder. The first layer is of WordPiece tokens that break words into separate tokens, and even smaller chunks if the word does not exist. Finally a tag of [CLS] is added at the beginning of the text body with a [SEP] tag between sentences to denote breaks.

To perform the tokenization, words are compared with a vocabulary lookup table to note if a word or symbol exists. In BERT this comes to approximately 30,000 words (Devlin et al., 2019). As an example let's use the sentence “The apple is cold. It tasted good.” The text would first be broken down into a list of [[CLS] , “the”, “apple”, “is”, “cold” ,“.”, [SEP], “it”, “taste”, “##d”, “good”, “.” ]. Lets assume that for the sake of the example the token “it” did not exist in our vocabulary lookup table. The text would then be further split until a match is found; in our case [“it”] → [“##i”, “##t”] (Hugging Face, n.d.).

The other two layers that are part of the final input vector are the positional and segment embeddings. The positional embedding simply highlights the location of the token in the sequence of text (similar to an index), while the segment embedding denotes the location of the sentence within a sequence of text. All three of these embedding vectors are eventually made into one through an element wise sum. To illustrate please see the diagram below. (Devlin et al., 2019)


<img src="/assets/img/input_embd.png" class="centered-image" style="max-width: 100%; height: auto;">

<small>Figure 3. BERT input representation. Reprinted from BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding by Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2019), arXiv, [Link](https://arxiv.org/abs/1810.04805)</small>


#### Multihead Attention: 

The inputs are then passed through the attention heads where the actual context is learned. The importance of this process can be exemplified by reusing the sentence in the beginning - “The patient received the treatment and discharge paperwork and took it home to review.” To a human it may seem obvious that “it” in the sentence is referring to the “discharge paperwork” and not the actual treatment itself. However, to a computer this may not be so obvious. This brings forth the purpose of the attention heads, which is to place “attention” in various aspects of the sentence to which relationships and contextual nuance can be modeled.

The basic functionality of the attention heads are as follows. Let's assume a sentence of “The doctor prescribed the medication.” An attention head evaluates a sequence of tokens $\mathbf{X}_1, \mathbf{X}_2, \dots, \mathbf{X}_n$ where each $\mathbf{X}_i$ is the embedding vector for token $i$. For a given focus token at position $f$, the attention head will compute how much “attention” should be placed upon each previous token in the sequence $i \leq f$.

<img src="/assets/img/attn_head.png" class="centered-image" style="max-width: 100%; height: auto;">

<small>Figure 4. Information flow in causal self-attention. Reprinted from Speech and language processing: An introduction to natural language processing, computational linguistics, and speech recognition with language models (3rd ed., Draft of January 12, 2025), by D. Jurafsky & J. H. Martin, 2025, [Link](https://web.stanford.edu/~jurafsky/slp3)</small>


During this process the model also has access to the current input along with all prior inputs. For instance when focusing on “medication” in the example the attention head considers the focus token in relation to all previous tokens. That is “The”, “doctor”, “prescribed”, and “the”. The corresponding attention computation can be thought of as the weighted sum of all the token embeddings prior and up to the focus token (Jurafsky & Martin, 2025): 

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


These scores are then passed through the softmax function to produce the attention weights for computing the final attention adjusted embedding vector. This vector is of the same dimensions as the input vector.  

$$
\mathbf{A}_f = \sum_{i=1}^5 \alpha_{5i} \mathbf{X}_i
$$

This describes just a single attention head in the process. BERT however includes multiple attention heads in the same layer to model different relational aspects of the input text. For additional mathematical details please reference this [paper](https://people.tamu.edu/~sji/classes/attn.pdf) (Ji, Xie, & Gao, 2019).


#### Layer Norm: 

After the inputs have passed through the multihead attention layer they are normalized through a z-score like method to improve their performance on deep learning based tasks. As with each layer of the encoder block, the normalization is applied to an embedding of a single token keeping the dimensionality the same, while adjusting for mean 0 and a standard deviation of 1. After the following feedforward network portion, there is one final normalization layer before the final embedding is produced (Devlin et al., 2019). 


#### Feed Forward: 

The last step of the encoder block (aside from a last normalization) is a 2-layer feedforward multi-layer perceptron neural network that maps the input embedding to a new vector that captures deeper and more abstract linguistic features. Across the different encoder blocks these networks learn different features such as parts of speech, grammatical roles, relationships of linguistic features, task specific functions, among others (Devlin et al., 2019). For more details on neural network architecture please reference this [overview](https://www.youtube.com/watch?v=A2RV4qOQQE0&t=0s).



## Are Medical Professionals the Original BERT?

Until recent breakthroughs in technology a key part of the role as a medical professional has been to provide a communicative service with a patient where the provider uses natural language as a medium for determining a diagnosis. This essentially acts as a classification task in which the professional utilizes the features identified in natural language to help make a decision on what a diagnosis may be and how to treat it. 

To make the diagnosis, a provider leverages the knowledge obtained through education and practice to compare the natural language through a mental model to come to a conclusion. This however is not a fool-proof strategy in that the role as a medical provider is much more holistic than simply being a translator to make a diagnosis. 

As a result we cannot realistically expect human medical professionals to be able to pick up on all details of a patient's language to form a highly accurate diagnosis.


## BERT Based Replication

To help show that medical professional efforts are more impactful from a holistic and relational standpoint, I implement a supervised classification scenario on patient symptom descriptions that are contextually sparse. This follows known uses for the BERT architecture proposed in the original BERT paper (Devlin et al., 2019). The patient descriptions are transformed into embeddings utilizing a custom pre-trained BERT model named "ClinicalBERT" trained on over 3 million patient electronic health records. It's important to note that ClinicalBERT is technically a DistilBERT model which is a reduced form of BERT with half the amount of transformer encoding blocks but achieves nearly similar accuracy. For details please refer to the original DistilBERT [paper](https://arxiv.org/pdf/1910.01108) (Sanh et al., 2020) as details will be provided on a different page in the future. After the embeddings are created, a simple feed-forward neural network is trained on the embeddings for further fine tuning and classification testing.


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

The custom BERT class is instantiated, splitting the text and labels into separate datasets for training, tuning, and testing (a default of 70%, 20%, and 10% respectively). The data is then split into tokens that meet the requirements for the specified model (i.e. ClinicalBERT) returning a pooled variant of the last hidden state as the sentence embedding. The final BERTdataset class wraps the embeddings into a PyTorch compatible dataset for efficient loading and batching during training. 

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

The dataset objects are subsequently iterated over using the PyTorch DataLoader class to create mini batches of size $n$ along with other functionalities such as shuffling to avoid learning order biases. The classifier head is then instantiated with the desired hyperparameters along with instantiating the sftTune class to specify the optimizer and loss function. It's important to note that the input dimensions must be 768 in this case to match the embedding vector, along with the number of classes being 22 corresponding to those identified in the initial labelling.

```python

train_loader = DataLoader(train_data, batch_size=1, shuffle=True)
dev_loader   = DataLoader(dev_data, batch_size=1, shuffle=False)
test_loader  = DataLoader(test_data, batch_size=1, shuffle=False)

classification_head = cBertMCChead(input_dim=768, hidden_dim=256, dropout=0.3, num_layers=1, num_classes=22)
tuner = sftTune(model=classification_head, learn_rate=1e-4)
optimizer = tuner.adamW()
loss_fn = tuner.xc_entropy()

```




### Train-Tune: 

The training and testing classes are instantiated to run the train loop over a set number of epochs. A breaking mechanism is included so that if the model accuracy did not improve over a set number of epochs the training loop would stop and the best model be kept. The accuracy metrics in this cycle are evaluated against the dev dataset to tune the hyperparameters that maximize performance. The resulting set of parameters are then modified on the classifier head with a new train-tune cycle executed. Top tuning scores achieved greater than 87% accuracy, 88% precision, 87% recall, and 86% F1. 

```python

trainer = cBERT_train()
testing = Bert_test()

top_model = None
lowest_tune_ls = float('inf')
no_improve = 0
epoch_limit = 5

for epoch in range(60):
    print(f"Epoch {epoch+1}")
    trainer.train_loop(train_loader, classification_head, loss_fn, optimizer, device='cpu')
    tune_loss, *_ = testing.test_loop(dev_loader, classification_head, loss_fn)

    if tune_loss < lowest_tune_ls:
        lowest_tune_ls = tune_loss
        top_model = copy.deepcopy(classification_head)
        no_improve = 0
        print('No improvement count reset!')
    else:
        no_improve += 1
        print('No improvement detected!')

        if no_improve >= epoch_limit:
            print('Epoch limit reached!')
            break


```

### Testing

A final test loop is performed on the testing dataset to evaluate the generalizability of the model. Utilizing the best tuned version of the model the testing run achieves scores of approximately 80% accuracy, 80% precision, 80% recall, and 78% F1. As mentioned in the introduction, the limited size of the dataset and the complexity of the model lend to a concern for overfitting. A larger dataset and deeper evaluation methods are required for a greater understanding of the model capabilities. 

```python

testing.test_loop(test_loader, top_model, loss_fn)

```



## Classification Head Tuning Considerations

After creating the patient embeddings and performing the training and initial evaluation of the classification head, the following settings of the hyperparameters resulted in the best performance. The model was specifically evaluated on accuracy, precision, recall, and F1 to provide a more holistic lens on performance: 

#### Network Architecture: 

There were virtually no gains to increasing the number of layers within the network architecture beyond 2, and many times decreased the classification performance of the model. This can potentially be attributed to the level of abstraction that is achieved during the embedding process. 

#### Training Epochs: 

A number of training epochs 50 and above were necessary as the loss consistently decreased. Beyond 60 epochs the accuracy of the model leveled off.

#### Optimizer: 

Per the original transformers [paper](https://arxiv.org/pdf/1706.03762) (Vaswani et al., 2023), the Adam optimizer is standard by which the encoder-decoder is trained with. While testing various optimization techniques the Adam based optimizers produced the most promising results. The AdamW optimizer was identified as the best performing with most other options producing substantially worse performance. 

#### Batch Sizes: 

In the case of tuning, batch sizes near 1 appear to produce the best results. The additional detail introduced through the smaller batch size appears to help the model reach a more optimum minima. A cause for concern is that given the training dataset is relatively small, the neural net may be too complex for the task and thus require a less complex model due to potential overfitting. 


## Citations


<small>Celik, M. (2023). Mayo Clinic Symptoms and Diseases v1 [Dataset]. Hugging Face. [Link](https://huggingface.co/datasets/celikmus/mayo_clinic_symptoms_and_diseases_v1)</small>

<small>Daniel Jurafsky and James H. Martin. 2025. Speech and Language Processing: An Introduction to Natural Language Processing, Computational Linguistics, and Speech Recognition with Language Models, 3rd edition. Online manuscript released January 12, 2025. [Link](https://web.stanford.edu/~jurafsky/slp3)</small>

<small>Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2019). BERT: Pre-training of deep bidirectional transformers for language understanding. arXiv. [Link](https://arxiv.org/abs/1810.04805)</small>

<small>Gretel.ai. (2023). Symptom to Diagnosis [Dataset]. Hugging Face. [Link](https://huggingface.co/datasets/gretelai/symptom_to_diagnosis)</small>

<small>Hugging Face. (n.d.). WordPiece tokenization. In Hugging Face Course. Retrieved May 7, 2025, from [Link](https://huggingface.co/learn/llm-course/en/chapter6/6)</small>

<small>Ji, S., Xie, Y., & Gao, H. (2019). A mathematical view of attention models in deep learning. Texas A&M University. [Link](https://people.tamu.edu/~sji/classes/attn.pdf)</small>

<small>Medical AI. (2025). ClinicalBERT [Model]. Hugging Face. [Link](https://huggingface.co/medicalai/ClinicalBERT)</small>

<small>Sanh, V., Debut, L., Chaumond, J., & Wolf, T. (2020). DistilBERT, a distilled version of BERT: Smaller, faster, cheaper and lighter. arXiv. [Link](https://arxiv.org/abs/1910.01108)</small>

<small>Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., & Polosukhin, I. (2023). Attention is all you need. arXiv. [Link](https://arxiv.org/abs/1706.03762)</small>



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
