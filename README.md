# OneshotCLIP
## Official Source code of "One-Shot Adaptation of GAN in Just One CLIP"

### Environment
Pytorch 1.7.1, Python 3.6

```
$ conda create -n oneshotCLIP python=3.6
$ conda install --yes -c pytorch pytorch=1.7.1 torchvision cudatoolkit=11.0
$ pip install ftfy regex tqdm
$ conda install -c anaconda git
$ pip install git+https://github.com/openai/CLIP.git
```

Before training, please download the pre-trained models on large datasets:

LINK: FFHQ

### Training 
To train the model, run

```
python train_oneshot.py --exp exp1 --data_path $DATA_PATH$ --ckpt $SRC_MODEL_PATH$
```
```$DATA_PATH$``` is a directory for single-shot target image

```$SRC_MODEL_PATH$``` is a path for source domain pre-trained model. 

Default: ```./checkpoint/stylegan2-ffhq-config-f.pt```



### Testing
To test the model with adapted generator,
```
python test_oneshot.py --exp exp1 --ckpt $TARGET_MODEL_PATH$
```

```$TARGET_MODEL_PATH$``` is path for adapted target domain model.

### Testing for real images
Will be updated layer
