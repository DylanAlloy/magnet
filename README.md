# 🧲 magnet

<small>small, efficient embedding (SL, 'small language') model toolkit</small>

<small>

_~ finetune SOTA models on knowledge bases, rapidly ~_

</small>

## 🎉 usage

[check out this notebook, it's really useful](./example.ipynb) `(./example.ipynb)`

<small>a snippet to get you started</small>

``` python
from magnet.utils import Utils
Utils().check_cuda()
raw_dir = "./raw"
cleaned_dir = './data'
source_data_file = 'your_kb_export.csv'
plaintext_column = 'clean'
```

<small>*(yes, this is all it takes to initialize a project!)*</small>

## 😥 compute requirements

_minimum_ requirements for ~6000 documents from a knowledge base:

 1. RAM
    - 32GB RAM
 2. CPU
    - `use_multiprocessing=True` on some `FinePrep()` functions will create a free space heater out of your computer
 3. GPU
    - multiprocessing is not necessary
    - 4x 16GB VRAM (*for finetuning with research efficiency*)
    - otherwise helpful with scoring/ranking

#### ⏱️ "Ready, Set, Go!"

| ⚡️ Processor 	| 𝍌 RAM 	| 📝 # Processed 	| 📊 Task 	| 🧠 Model Prep 	| 🏁 Time 	| ⚙️ Params 	|
|---	|---	|---	|---	|---	|---	|---	|
| M1 Max 10-core 	| 32GB 	| **5,725** 	| `Processor.export_with_sentences` 	| `bge-large-en-v1.5` 	| _06m 05s_ 	| - 	|
| M1 Max 10-core 	| 32GB 	| **178** 	| `FinePrep.generate_scored_data` 	| `bge-large-en-v1.5` 	| _03h 07m 05s_ 	| `use_multiprocessing`, `split=32` 	|
| M1 Max 10-core 	| 32GB 	| **75,864** 	| `FinePrep.generate_training_data` 	| `bge-large-en-v1.5` 	| _01m 08s_ 	| - 	|
| M1 Max 10-core 	| 32GB 	| **75,852** 	| `FinePrep.find_knn_neg` 	| `bge-large-en-v1.5` 	| _01h 00m 40s_ 	| `sample_range=[0,500]`, `num_hard_negatives=15` 	|

   - 💻 M1 Max 10-core - **_`04h 13m 58s`_ ⌛️ to synthesize a dataset for finetuned embeddings from a knowledge base containing 5725 articles worth of information**



## 👏 features

 - Apple silicon first class citizen
 - so long as your initial data has columns for article text and ids, `magnet` can do the rest
 - upload to S3
 - parallel inference on CPU
 - simple management principles - `raw` and `cleaned` dataset directories
 - ideal cyberpunk vision of LLM power users in vectorspace

## goals

- [x] finish `README.md`
- [ ] `deepspeed` integration for model parallelism on multiple GPU
