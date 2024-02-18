# Local and Global Logit Adjustments (LGLA) Model


## 2. Requirements
* To install requirements: 
```
pip install -r requirements.txt
```

## 3. Datasets 
* Please download datasets: ImageNet-LT, iNaturalist 2018, and Places-LT and put them to the corresponding "data_dir" in the config file.
* CIFAR-100/CIFAR-10 will be downloaded automatically with the dataloader.


## 4. Pretrained models
* For the training on Places-LT, we follow previous methods and use the pre-trained ResNet-152 model.
* Please download the checkpoint. Unzip and move the checkpoint files to /model/pretrained_model_places/.


## Training
### (1) CIFAR10-LT 

* To train the LGLA model on CIFAR10-LT with IF=50, run this command:
```
python3 train.py -c configs/config_cifar10_ir50_lgla.json
```

* To train the LGLA model on CIFAR10-LT with IF=100, run this command:
```
python3 train.py -c configs/config_cifar10_ir100_lgla.json
```

### (2) CIFAR100-LT 

* To train the LGLA model on CIFAR100-LT with IF=50, run this command:
```
python3 train.py -c configs/config_cifar100_ir50_lgla.json
```

* To train the LGLA model on CIFAR100-LT with IF=100, run this command:
```
python3 train.py -c configs/config_cifar100_ir100_lgla.json
```

### (3) ImageNet-LT

* To train the LGLA model on ImageNet-LT with the backbone ResNet50, run this command:
```
python3 train.py -c configs/config_imagenet_lt_resnet50_lgla.json
```

* To train the LGLA model on ImageNet-LT with the backbone ResNeXt50, run this command:
```
python3 train.py -c configs/config_imagenet_lt_resnext50_lgla.json
```

### (4) iNaturalist 2018

* To train the LGLA model on iNaturalist 2018, run this command:
```
python3 train.py -c configs/config_iNaturalist_resnet50_lgla.json
```

### (5) Places-LT

* To train the LGLA model on Places-LT, run this command:
```
python3 train.py -c configs/config_places_lt_resnet152_lgla.json
```


## Evaluate
* To evaluate LGLA model on the above benchmarks, run:
``` 
python test.py -r checkpoint_path
``` 
