# Personalized Fashion Recommendation and Generation

This is our TensorFlow implementation for the paper:

*Wang-Cheng Kang, Chen Fang, Zhaowen Wang, Julian McAuley. Visually-Aware Fashion Recommendation and Design with Generative Image Models. In Proceedings of IEEE International Conference on Data Mining (ICDM'17) ([perm_link](http://ieeexplore.ieee.org/document/8215493/), [pdf](http://cseweb.ucsd.edu/~jmcauley/pdfs/icdm17.pdf))*

Please cite our paper if you use the code or datasets.

We provide the three modules in our framework: 

- **Deep Visually-Aware Bayesian Personalized Ranking (DVBPR):** Jointly learn user latent factors and extract task-guided visual features from implicit feedback for fashion recommendation.
- **GANs:** A conditional generative adversarial network for fashion generation.
- **Preference Maximization:** Adjust generated images that match a user's personal taste better (personalized fashion design).

## Environment
The code is tested under a Linux desktop with a single GTX-1080 Ti GPU.

- Tensorflow:  1.3
- Numpy
- PIL

## Datasets

The four fashion datasets can be downloaded via

```
bash download_dataset.sh 
```

All datasets are stored in *.npy* format, each item is associated with a JPG image. Please refer to DVBPR code for detail usage.

Amazon datasets are derived from [here](http://jmcauley.ucsd.edu/data/amazon/), tradesy dataset is introducde in [here](http://jmcauley.ucsd.edu/data/tradesy/). Please cite the corresponding papers if you use the datasets.

**Please note the raw images are for academic use only.**

## Model Training

**Step 1:** Train DVBPR:

```
cd DVBPR
python main.py
```

The default hyper-parameters are defined in *main.py*, you can change them accordingly. AUC (on validation and test set) is recorded in *DVBPR.log*.

**Step 2:** Train GANs:

```
cd GAN
python main.py --train True
```
The default hyper-parameters are defined in *main.py*, you can change them accordingly.

**Step 3:** Preference Maximization:

```
cd PM
python main.py
```

With a single GTX-1080 Ti, training DVBPR and GANs take around 7 hours respectively.

## Demo (with pretrained models)
A quick way to use our model is using pretrained models which can be acquired via: 

```
bash download_pretrained_models.sh 
```

## Misc

- Acknowledgments: GAN code borrows heavily from [DCGAN](https://github.com/carpedm20/DCGAN-tensorflow). GAN networks are modified from [LSGAN](https://arxiv.org/pdf/1611.04076.pdf).
- In principal, our framework can adapt with any GANs variant, we look forward to using advanced GANs to achieve better generation results with higher resolution.
