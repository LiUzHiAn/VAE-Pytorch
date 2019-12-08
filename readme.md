# 1.Overview
This is the Pytorch implementation of variational auto-encoder, applying on MNIST dataset.

Currently, the following models are supported:
- :heavy_check_mark:	VAE
- :heavy_check_mark:	Conv-VAE

# 2.Usage
```python
python train.py
```
The code is self-explanatory, you can specify some customized options in `train.py`.

# 3.Result
Here are some visualization results：

## 3.1 Reconstruction results
|Model | epoch 10| epoch 20| epoch 30 | epoch 40| epoch 50|
|:---: | :---:   | :---:   | :---:   | :---:   | :---:   |
|VAE   | ![]( ./VAEresult/reconstructed-10.png) |  ![](./VAEresult/reconstructed-20.png) | ![](./VAEresult/reconstructed-30.png) | \| \|
|Conv-VAE | ![](./convVAEresult/reconstructed-10.png)  |  ![](./convVAEresult/reconstructed-20.png) |  ![](./convVAEresult/reconstructed-30.png)  |  ![](./convVAEresult/reconstructed-40.png) |  ![](./convVAEresult/reconstructed-50.png) |

## 3.2 Randomly generated results

|Model | epoch 10| epoch 20| epoch 30 | epoch 40| epoch 50|
|:---: | :---:   | :---:   | :---:   | :---:   | :---:   |
|VAE  | ![](./VAEresult/random_sampled-10.png) | ![](./VAEresult/random_sampled-20.png) |  ![](./VAEresult/random_sampled-30.png)| \| \|
|Conv-VAE | ![](./convVAEresult/random_sampled-10.png)  |  ![](./convVAEresult/random_sampled-20.png) | ![](./convVAEresult/random_sampled-30.png ) | ![]( ./convVAEresult/random_sampled-40.png) |  ![](./convVAEresult/random_sampled-50.png) |

# 4. Pre-trained model
Donwload link:
