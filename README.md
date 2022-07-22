# DDPM - Denoising Diffusion Probabilistic Model

This repository provides a modular implementation for diffusion models. For the time being it includes a DDPM method.


## Example installation and use

### Install dependencies

```
pip install -r requirements.txt
```


### Things to know

- Training is only available on GPUs but generation is not. 
- There is default configuration that needs changed for certain parameters (path to dataset for instance)
- If you wish to overwrite a config parameter you can either create a new config file and provide the path in the CLI using `--config` 
  OR use the associated CLI param `--ParamName`

### Run program

There are two important params `-t`and `-l` for specifying wether you want to train and/or load a model.

```
$ # CUDA_VISIBLE_DEVICES="X"

# For training
$ python main.py -t

# If you wish to load an existing model
$ python main.py -l 

# to train an existing model again
$ python main.py -t -l
```

### Example

```
#Changing the number of steps in generation + different learning rate
$ python main.py -l --num_steps=300 --lr=0.0001
```

### List of variables that can written over

| Params      |  |       |   |
| :---        |    :----:   |    :----:     |  ---: |
| ```--train``` | ```--load``` | ```--config``` | ```--num_steps```|
| ```--num_workers``` | ```--dataprocessor_type``` | ```--sde``` | ```--sampler_name```|
| ```--generate_num_samples``` | ```--striding``` | ```--input_audio``` | ```--output_audio```|
| ```--generate_batch_size``` | ```--fs``` | ```--guidance``` | ```--diffusion_model_type```|
| ```--lr``` | ```--batch_size``` | ```--num_batches``` | ```--ema_rate```|
| ```--load_ema``` | ```--compute_fad``` | ```--input_files``` | ```--stats```|

# FAD

This repository provides supporting code used to compute the Fréchet Audio Distance (FAD), a reference-free evaluation metric for audio generation algorithms, in particular music enhancement.

For more details about Fréchet Audio Distance and how we verified it please check out our paper:

* K. Kilgour et. al.,
  [Fréchet Audio Distance: A Metric for Evaluating Music Enhancement Algorithms](https://arxiv.org/abs/1812.08466),

### Install dependencies

```shell
# The only new dependencies are the following
$ pip install apache-beam numpy scipy tensorflow

# Need to download this file to put in frechet_audio_distance/data
$ curl -o data/vggish_model.ckpt https://storage.googleapis.com/audioset/vggish_model.ckpt
```

### Create support file with your dataset

You need to create cvs and stat file yourself using the following : 

Before running the following command, see frechet_audio_distance/create_cvs.py file to add desired dataset in the __main__ part. CVS file is basically just a list of all audio files you want to compute. They need to be stored in the audio_cvs folder.

```shell
$ mkdir -p audio_cvs
$ python -m frechet_audio_distance.create_cvs
```

### Compute stats : embeddings and eastimate multivariate Gaussians
If not already created :
```shell
$ mkdir -p stats
```

Then compute the stats : with the following command for reference dataset (replace dataset by name of dataset in the following)

```shell
$ python -m frechet_audio_distance.create_embeddings_main --input_files audio_cvs/{dataset}.cvs --stats stats/{dataset}_stats
```

## Compute FAD manually
### A pipeline was included in the project to do so already but here goes: 
```shell
$ python -m frechet_audio_distance.compute_fad --background_stats stats/{dataset}_stats --test_stats stats/stat_file_of_audio_to_compute_fad_from

```

# Thanks

The whole code architecture stems from Dr. Hadjères' advises. It is extremely modular and allows to build new models on top of wrapper classes.
