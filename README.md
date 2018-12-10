# Eyeglasses detection with model less than 3Mb.

## Task definition

We need to create face image classifier to detect eyeglasses. We have following constraints: 

* we want to make our model less than 3Mb, to use it on mobile devices.
* 100 ms limit to inference (~ GeForce GTX 1080 Ti)

## Solution and results

We decided to use SqueezeNet. It has 723 009 parameters for binary case. It will take about 2.75 Mb.
We compared performance and inference time of ResNet-18 and SqueezeNet.


| Model | E[F1-score] | Var[F1-score] | 
| ------ | ------: | ------: |
| ResNet-18 | 0.997 | 0.004 |
| SqueezeNet | 0.984 | 0.007 |

Table 1. Dataset of size 1800. Cross validation over 5 folds.


| Model | Device | Inference time, ms | 
| ------ | ------ | ------ |
| ResNet-18 | K80 | 4.367 |
| ResNet-18 | CPU (Intel Xeon E5-2686 v4) | 101 |
| SqueezeNet | K80 | 4.256 |
| SqueezeNet | CPU (Intel Xeon E5-2686 v4) | 38 |

Table 2. Inference time for single image

## Dataset

To train our model, we used images of size 120x120 from [MeGlass](https://github.com/cleardusk/MeGlass) dataset. We took just 800 samples of each class. 
To train model we put our samples to "0" and "1" folders. Both of these folders must be located at the same common folder.


## Inference

### Using docker 

```sh
$ docker run -it --rm  -v /images/to/classify:/workspace/images acheshkov/glasses
```
### Your computer

```sh
git clone https://github.com/acheshkov/eyeglasses_classifier_ligth.git
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
git clone https://github.com/acheshkov/eyeglasses_classifier_ligth.git
pip install -r requirements.txt
python train.py --epochs=100 --batch-size=40 --images-path='/path/to/dataset'
```