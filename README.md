# OneshotCLIP
## Official Source code of "One-Shot Adaptation of GAN in Just One CLIP"

### Environment
Pytorch 1.7.1, Python 3.6

```
$ conda create -n oneshotCLIP python=3.6
$ conda install --yes -c pytorch pytorch=1.7.1 torchvision cudatoolkit=11.0
$ pip install ftfy regex tqdm
$ conda install -c anaconda git
$ conda install -c conda-forge packaging
$ pip install git+https://github.com/openai/CLIP.git
```

Before training, please download the pre-trained models on large datasets:

LINK: [FFHQ](https://drive.google.com/file/d/1nS5fqO5XLwB4-VjR0Ma059dgrKB9eyjb/view?usp=sharing)

### Training 
To train the model, run

```
python train_oneshot.py --exp exp1 --data_path $DATA_PATH$ --ckpt $SRC_MODEL_PATH$
```
```$DATA_PATH$``` is a directory for single-shot target image

```$SRC_MODEL_PATH$``` is a path for source domain pre-trained model. 

Default: ```./pretrained_model/stylegan2-ffhq-config-f.pt```

--exp is for checkpoint directory name

For human face dataset training, download portrait dataset in [LINK](https://github.com/mahmoudnafifi/HistoGAN)

### Testing
To test the model with adapted generator,
```
python test_oneshot.py --exp exp1 --ckpt $TARGET_MODEL_PATH$ --ckpt_source $SOURCE_MODEL_PATH$
```

```$TARGET_MODEL_PATH$``` is path for adapted target domain model.

```$SOURCE_MODEL_PATH$``` is path for source domain model. Default: ```./pretrained_model/stylegan2-ffhq-config-f.pt```

For testing, we provide several adapted models

[LINK](https://drive.google.com/drive/folders/1svLJjuuK-yCCJ7Xq9l4Dy4gSuzplK_7i?usp=sharing)


### Testing for real images
For testing on real images, we provide demo on Google Colab [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1Qqp3gRYArnY4pY6Am_aI9l_EOwXgxf9j?usp=sharing).
