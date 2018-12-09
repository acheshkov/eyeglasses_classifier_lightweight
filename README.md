# Eyeglasses detection with model less than 3Mb.

## Task definition

We need to create face image classifier to detect either we have eyeglasses or not. But we have a constraints: 

* we want to make our model less than 3Mb to use it on mobile devices.
* 100 ms limit to inference (GeForce GTX 1080 Ti)

## Solution and results

We decided to use SqueezeNet. It has 723 009 parameters for binary case. It will take about 2.75 Mb.
We going to train SqueezeNet usual way and with distillation. We compared ResNet-18, SqueezeNet trained usual way and distilled.


| Model | E[F1-score] | Var[F1-score] | 
| ------ | ------ | ------ |
| ResNet-18 | 0.997 | 0.004 |
| SqueezeNet (usual) | 0.956 | 0.014 |
| SqueezeNet (distillation) |  0.959 | 0.016 |
Table 1. Dataset of size 600. Cross validation over 5 folds.


| Model | Inference time, ms | 
| ------ | ------ | ------ |
| ResNet-18 | |
| SqueezeNet | |

Table 2. Inference time for single image

## Dataset

To train our model, we will use [MeGlass](https://github.com/cleardusk/MeGlass) dataset. We are going to use images of size 120х120 and resize it to 224x224. Both classes are balanced. 
To train model we put our samples to "0" and "1" folders. Both of these folders must be located at the same common folder.


## Inference

### Using docker 

```sh
$ docker run -it --rm  -v /images/to/classify:/workspace/images acheshkov/glasses
```
### Your computer

```sh
git clone ...
pip install -r requirements.txt
python inference.py --images-path='images/to/classify' --model-params-path='./dist/squeezenet_params'
```

### How to improve inference time

1. Quantization. Using fewer bits to store model weights. 
2. Pruning. Analyze weights and wipe out those whose values close to zero.
3. Looking for lighter architecture than SqueezeNet.
4. We could use smaller input image size.
5. We could exploit hardware specific optimization


## Training
 
```sh
git clone ...
pip install -r requirements.txt
python train.py --epochs=100 --batch-size=40 --images-path='/path/to/dataset'
```