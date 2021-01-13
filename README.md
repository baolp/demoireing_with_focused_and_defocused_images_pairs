# Self-Adaptively Learning to Demoire from Focused and Defocused Image Pairs(NIPS'20)
Lin Liu, Shanxin Yuan, Jianzhuang Liu, Liping Bao, Gregory Slabaugh, Qi Tian

[\[Arxiv\]](https://arxiv.org/abs/2011.02055) 

## Abstract
Moiré artifacts are common in digital photography, resulting from the interference between high-frequency scene content and the color filter array of the camera. Existing deep 
learning-based demoiréing methods trained on large scale datasets are limited in handling various complex moiré patterns, and mainly focus on demoiréing of photos taken of 
digital displays. Moreover, obtaining moiré-free ground-truth in natural scenes is difficult but needed for training. In this paper, we propose a self-adaptive learning method 
for demoiréing a high-frequency image, with the help of an additional defocused moiré-free blur image. Given an image degraded with moiré artifacts and a moiré-free blur image, 
our network predicts a moiré-free clean image and a blur kernel with a self-adaptive strategy that does not require an explicit training stage, instead performing test-time 
adaptation. Our model has two sub-networks and works iteratively. During each iteration, one sub-network takes the moiré image as input, removing moiré patterns and restoring 
image details, and the other sub-network estimates the blur kernel from the blur image. The two sub-networks are jointly optimized. Extensive experiments demonstrate that our 
method outperforms state-of-the-art methods and can produce high-quality demoiréd results. It can generalize well to the task of removing moiré artifacts caused by display 
screens. In addition, we build a new moiré dataset, including images with screen and texture moiré artifacts. As far as we know, this is the first dataset with real texture moiré 
patterns.

## Citation
If you use this code, please cite:

```
@article{liu2020self,
  title={Self-Adaptively Learning to Demoir{\'e} from Focused and Defocused Image Pairs},
  author={Liu, Lin and Yuan, Shanxin and Liu, Jianzhuang and Bao, Liping and Slabaugh, Gregory and Tian, Qi},
  journal={arXiv preprint arXiv:2011.02055},
  year={2020}
}
```
