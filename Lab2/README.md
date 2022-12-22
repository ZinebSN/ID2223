# Scalable Machine Learning and Deep Learning
# Lab 2 : Swedish Transcriber
[[Blog]](https://openai.com/blog/whisper)
[[Paper]](https://cdn.openai.com/papers/whisper.pdf)
[[Model card]](model-card.md)
[[Colab example]](https://colab.research.google.com/github/openai/whisper/blob/master/notebooks/LibriSpeech.ipynb)

## Model: Whisper

Whisper is a general-purpose speech recognition model developped by OpenAi. It is trained on a large dataset of diverse audio and is also a multi-task model that can perform multilingual speech recognition as well as speech translation and language identification.

Its main idea is based on a trained transformer sequence-to-sequence modelon various speech processing tasks, including multilingual speech recognition, speech translation, spoken language identification, and voice activity detection. All of these tasks are jointly represented as a sequence of tokens to be predicted by the decoder, allowing for a single model to replace many different stages of a traditional speech processing pipeline. The multitask training format uses a set of special tokens that serve as task specifiers or classification targets.

## Available models and languages

There are five model sizes, four with English-only versions, offering speed and accuracy tradeoffs. Below are the names of the available models and their approximate memory requirements and relative speed. 


|  Size  | Parameters | Multilingual model | Required VRAM | Relative speed |
|:------:|:----------:|:------------------:|:-------------:|:--------------:|
|  tiny  |    39 M    |       `tiny`       |     ~1 GB     |      ~32x      |
|  base  |    74 M    |       `base`       |     ~1 GB     |      ~16x      |
| small  |   244 M    |      `small`       |     ~2 GB     |      ~6x       |
| medium |   769 M    |      `medium`      |     ~5 GB     |      ~2x       |
| large  |   1550 M   |      `large`       |    ~10 GB     |       1x       |

## Approach


We used the small Whisper model with 244M parameters and we followed the same steps as in the colab example.
First, we loaded the pre-trained FeatureExtractor and the pre-trained tokenizer and used it for fine-tuning without any further modifications. We have just set the target language to swedish and the task to transcription. These arguments inform the tokenizer to prefix the language and task tokens to the start of encoded label sequences.Then, we combined both into one processor to make it easier. 

Next, we created the DataCollector treating the data in an independent schema composed of inputs and labels, and we the WER metric to evaluate the performance of the fine-tuned model at the end.

Finaly, we loaded the pre-trained small whisper model from the openAi checkpoint.


## Model Fine-Tuning
In order to fine tune the pretrained model, we used a modal-centric approach. We tried different hyperparameters, and the ones that gave us the best performance are the following:

* per_device_train_batch_size=16,
* gradient_accumulation_steps=1,  # increase by 2x for every 2x decrease in batch size
* learning_rate=1e-5,
* warmup_steps=500,
* max_steps=3000,
* per_device_eval_batch_size=16,
* save_steps=1000,
* eval_steps=1000,
* ogging_steps=1
    


## Setup
we refactored the program into three pipelines:

### Feature Engineering Pipeline 
In the [feature pipeline](https://github.com/avatar46/ID2223_lab2/blob/main/Features_Engineering.ipynb), we process the Swedish language dataset downloaded from [Common Voice](https://huggingface.co/datasets/mozilla-foundation/common_voice_11_0) and store it in Hopsworks or Google Drive. Since, the dataset is quite large(about 16.7 GB), we compress it to reduce its size. This pipeline can be run on CPUs in an efficient way and fastly. The dataset can be then loaded in the training pipeline using gdown library and just in 1 minute from the compressed file stored on google drive.  

### Training pipeline 
The [training pipeline](https://github.com/avatar46/ID2223_lab2/blob/main/Swedish_fine_tune_whisper_Transcriber.ipynb) launches the fine tuning part of the pre-trained whisper model after loading the data. We save checkpoints in each 500 steps and evalute the performance of the model. The checkpoints are very large so that they cannot fit google drive capacity, so they are pushed directly to HuggingFace hub. (Check if .git ignore file on the local repo cloned from the hub contains chackpoints-* and if yes delete it). The checkpoints will help us to resume from when google colab stop avoid taking over the training from the beginning. 

### Inference UI 
After training, we deploy our fine tuned model and we provide stakeholders a [inference UI](https://huggingface.co/spaces/ZinebSN/Transcriber) on Huggingface to test our model. You can also access our trained model card on [Huggingface model](https://huggingface.co/ZinebSN/whisper-small-swedish-Test-3000).  
Our interactive UI, offers multiple facilities: Uploading an audio, real time recording or entering a Youtube url, and it outputs the transcription of the audio signal passed as input. In addition, users can also trim their recorded/uploaded audio.

## Results
We trained our initial model for 4000 steps, and in each 500 steps we evaluate model's performance. As the table below shows, the WER(word error rate) is decreasing almost after each 500 steps and reach the lowest at the 4000 step.

| **Step** | **WER** |
|----------|---------|
| 500      | 23.90   |
| 1000     | 22.42   |
| 2000     | 20.94   |
| 3000     | 20.38   |
| 4000     | 19.94   |

After fine tuning the model using model-centric approach, we got the following results in less training time:
| **Step** |  **WER**  |
|----------|-----------|
| 1000     | 21.4245   |
| 2000     | 20.0882   |
| 3000     | 19.6042   |

The link to our two models and their inference UIs:

Original model: [Huggingface model](https://huggingface.co/Yilin98/whisper-small-hi), [inference UI](https://huggingface.co/spaces/Yilin98/Whisper-Small-Swedish)

New model: [Huggingface model](https://huggingface.co/ZinebSN/whisper-small-swedish-Test-3000), [inference UI](https://huggingface.co/spaces/ZinebSN/Transcriber)


## Discussion: Further Improvements

### Model-centric Approach
* We can use a larger model with more layers to get better performance, e.g. whisper-medium or whisper-large. But this will require much more time to train the model.

* The selection of different hyperparameters will also affect the performance. We can select the parameters using random search and grid search.

### Data-centric Approach
* For this approach, we can add new data sources to train a better model. We can add other public Swedish audio dataset such as [NST Swedish ASR Database](https://www.nb.no/sprakbanken/en/resource-catalogue/oai-nb-no-sbr-56/) and [Spoken Corpora](https://www.clarin.eu/resource-families/spoken-corpora).
* We can also do some sort of data augmentation on the audios, by adding some noise for example.
