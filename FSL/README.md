We modify our code from [eTT](https://github.com/chmxu/eTT_TMLR2022). Please follow the instruction in [eTT](https://github.com/chmxu/eTT_TMLR2022) for data preparation and meta-training.

### Weight
 
You can download the parameter of DINO-pretrained ViT-small from [Link](https://drive.google.com/file/d/1EVKvdVFmwmb8mrdawMX9JKS_yOWrdA77/view?usp=sharing). 
    
### Inference

Run the meta-testing as follow:

```shell script
python test_extractor_pa_vit_prefix.py --data.test ilsvrc_2012 omniglot aircraft cu_birds dtd quickdraw fungi vgg_flower traffic_sign mscoco --model.ckpt {WEIGHT PATH}
```
