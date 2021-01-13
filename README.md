# Self-Adaptively Learning to Demoire from Focused and Defocused Image Pairs
[Tianyu Wang](https://stevewongv.github.io)\*, Xin Yang\*, Ke Xu, Shaozhe Chen, Qiang Zhang, [Rynson W.H. Lau](http://www.cs.cityu.edu.hk/~rynson/) † 
(\* Joint first author. † Rynson Lau is the corresponding author.)

[\[Project Page\]](https://stevewongv.github.io/derain-project.html) [\[Arxiv\]](https://arxiv.org/abs/1904.01538) 

## Abstract
Removing rain streaks from a single image has been drawing considerable attention as rain streaks can severely degrade the image quality and affect the performance of existing outdoor vision tasks. While recent CNN-based derainers have reported promising performances, deraining remains an open problem for two reasons. First, existing synthesized rain datasets have only limited realism, in terms of modeling real rain characteristics such as rain shape, direction and intensity. Second, there are no public benchmarks for quantitative comparisons on real rain images, which makes the current evaluation less objective. The core challenge is that real world rain/clean image pairs cannot be captured at the same time. In this paper, we address the single image rain removal problem in two ways. First, we propose a semi-automatic method that incorporates temporal priors and human supervision to generate a high-quality clean image from each input sequence of real rain images. Using this method, we construct a large-scale dataset of ∼29.5K rain/rain-free image pairs that cover a wide range of natural rain scenes. Second, to better cover the stochastic distributions of real rain streaks, we propose a novel SPatial Attentive Network (SPANet) to remove rain streaks in a local-to-global manner. Extensive experiments demonstrate that our network performs favorably against the state-of-the-art deraining methods.

## Citation
If you use this code or our dataset(including test set), please cite:

```
@InProceedings{Wang_2019_CVPR,
  author = {Wang, Tianyu and Yang, Xin and Xu, Ke and Chen, Shaozhe and Zhang, Qiang and Lau, Rynson W.H.},
  title = {Spatial Attentive Single-Image Deraining with a High Quality Real Rain Dataset},
  booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  month = {June},
  year = {2019}
}
```
