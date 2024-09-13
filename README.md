Code for paper "Faster ISNet for Background Bias Mitigation on Deep Neural Networks"

https://arxiv.org/abs/2401.08409

# Abstract

Bias or spurious correlations in image backgrounds can impact neural networks, causing shortcut learning (Clever Hans Effect) and hampering generalization to real-world data. ISNet, a recently introduced architecture, proposed the optimization of Layer-Wise Relevance Propagation (LRP, an explanation technique) heatmaps, to mitigate the influence of backgrounds on deep classifiers. However, ISNet's training time scales linearly with the number of classes in an application. Here, we propose reformulated architectures whose training time becomes independent from this number. Additionally, we introduce a concise and model-agnostic LRP implementation. We challenge the proposed architectures using synthetic background bias, and COVID-19 detection in chest X-rays, an application that commonly presents background bias. The networks hindered background attention and shortcut learning, surpassing multiple state-of-the-art models on out-of-distribution test datasets. Representing a potentially massive training speed improvement over ISNet, the proposed architectures introduce LRP optimization into a gamut of applications that the original model cannot feasibly handle.

# Faster ISNet Creation Examples

Dependencies: Python, PyTorch, PyTorch Lightning

### LRP-Flex-based ISNets: An easy and fast way to make classifiers ignore backgrounds
```
import LRPDenseNetZe
import ISNetFlexLightning

DenseNet=LRPDenseNetZe.densenet121(pretrained=False)#Example of DNN

#Stochastic ISNet
net=ISNetFlexLightning.ISNetFlexLgt(model=DenseNet,selective=False,multiple=False,
                                    HiddenLayerPenalization=False,
                                    randomLogit=True,heat=True)
                                
#Stochastic ISNet LRP Deep Supervision
net=ISNetFlexLightning.ISNetFlexLgt(model=DenseNet,selective=False,multiple=False,
                                    HiddenLayerPenalization=True,
                                    randomLogit=True,heat=True)
#Selective ISNet
net=ISNetFlexLightning.ISNetFlexLgt(model=DenseNet,selective=True,multiple=False,
                                    HiddenLayerPenalization=False,
                                    randomLogit=False,heat=True)

#Selective ISNet LRP Deep Supervision
net=ISNetFlexLightning.ISNetFlexLgt(model=DenseNet,selective=True,multiple=False,
                                    HiddenLayerPenalization=True,
                                    randomLogit=False,heat=True)
                                
#Original ISNet
net=ISNetFlexLightning.ISNetFlexLgt(model=DenseNet,selective=False,multiple=True,
                                    HiddenLayerPenalization=False,
                                    randomLogit=False,heat=True)
```

### LRP Block-based ISNets:
```
import ISNetLightningZe

#Dual ISNet
net=ISNetLightningZe.ISNetLgt(architecture='densenet121',classes=10,selective=False,multiple=False,
                              penalizeAll=False,highest=False,randomLogit=True,rule='z+e')

#Dual ISNet LRP Deep Supervision
net=ISNetLightningZe.ISNetLgt(architecture='densenet121',classes=10,selective=False,multiple=False,
                              penalizeAll=True,highest=False,randomLogit=True,rule='z+e')                           

```

# Use LRP-Flex to explain an arbitrary DNN decision
```
import ISNetFlexTorch
import LRPDenseNetZe

#Examples of network and image
DenseNet=LRPDenseNetZe.densenet121(pretrained=False)
image=torch.randn([1,3,224,224])

#LRP-Flex PyTorch Wrapper
net=ISNetFlexTorch.ISNetFlex(model=DenseNet,
                             architecture='densenet121',#write architecture name only for densenet, resnet and VGG
                             selective=True,Zb=True,multiple=False,HiddenLayerPenalization=False,
                             randomLogit=False,explainLabels=True)#set explainLabels=False when defining ISNet

#Explain class 3
out=net(image,runLRPFlex=True,labels=torch.tensor([3]))
logits=out['output']
heatmap=out['LRPFlex']['input']

#Plot heatmap
import matplotlib.pyplot as plt
import matplotlib.colors as colors
h=heatmap.squeeze().mean(0).detach().numpy()
norm=colors.TwoSlopeNorm(vmin=h.min(), vcenter=0, vmax=h.max())
plt.imshow(h,cmap='RdBu_r', norm=norm,interpolation='nearest')
plt.show()
```

# Content
### LRP-Flex-based ISNets:

ISNets based on the LRP-Flex model agnostic implementation from "Faster ISNet for Background Bias Mitigation on Deep Neural Networks".

ISNetFlexLightning.py: PyTorch Lightning implementation of Selective, Stochastic and Original ISNets, based on LRP-Flex.

ISNetFlexTorch.py: PyTorch implementation of Selective, Stochastic and Original ISNets, based on LRP-Flex.

### LRP Block-based ISNets:

ISNets based on the LRP Block implementation, from (1), with the modifications explained in Appendix B of the paper "Faster ISNet for Background Bias Mitigation on Deep Neural Networks". Implemented for DenseNet, ResNet, VGG and simple nn.Sequential backbones.

ISNetLightningZe.py: PyTorch Lightning implementation of all Faster and Original ISNets, based on LRP Block.

ISNetLayersZe.py: PyTorch implementation of all Faster and Original ISNets, based on LRP Block.

ISNetFunctionsZe.py: Functions for LRP Block, introduced in (1) and expanded in this work.

### ISNet Softmax Grad * Input Ablation:

ISNetLightningZeGradient.py: Implementation of ISNet Softmax Grad * Input ablation study.

### Extras:

globalsZe.py global variables shared across modules.

LRPDenseNetZe.py: DenseNet code, based on TorchVision. Removes in-place ReLU, and adds an extra ReLU in transition layers. From (1).

resnet.py: resnet code, based on TorchVision. Removes in-place ReLU, and adds an extra ReLU in transition layers.

### Training Script Examples:

RunISNetGrad.py: Train and test ISNet Softmax Grad* Input on MNIST.

RunISNet.py: Train and test LRP Block-based ISNets on MNIST.

RunISNetFlex.py: Train and test LRP-Flex-based ISNets on MNIST.

SingleLabelEval.py: Evaluation script.

compare_auc_delong_xu.py: Dependency of SingleLabelEval.py.

locations.py: Folder locations for training script.

# Citations
If you use this code, please cite the papers below:

Bassi, P. R. A. S., Decherchi, S., & Cavalli, A. (2024). Faster ISNet for Background Bias Mitigation on Deep Neural Networks. arXiv: http://arxiv.org/abs/2401.08409

Bassi, P.R.A.S., Dertkigil, S.S.J. & Cavalli, A. (2024). Improving deep neural network generalization and robustness to background bias via layer-wise relevance propagation optimization. Nature Communications 15, 291. https://doi.org/10.1038/s41467-023-44371-z

BibTeX:

```
@misc{bassi2024faster,
      title={Faster ISNet for Background Bias Mitigation on Deep Neural Networks}, 
      author={Pedro R. A. S. Bassi and Sergio Decherchi and Andrea Cavalli},
      year={2024},
      eprint={2401.08409},
      archivePrefix={arXiv},
      primaryClass={eess.IV}
}
```

```
@article{Bassi2024,
  title = {Improving deep neural network generalization and robustness to background bias via layer-wise relevance propagation optimization},
  volume = {15},
  ISSN = {2041-1723},
  url = {http://dx.doi.org/10.1038/s41467-023-44371-z},
  DOI = {10.1038/s41467-023-44371-z},
  number = {1},
  journal = {Nature Communications},
  publisher = {Springer Science and Business Media LLC},
  author = {Bassi,  Pedro R. A. S. and Dertkigil,  Sergio S. J. and Cavalli,  Andrea},
  year = {2024},
  month = jan 
}
```




# Benchmark models

For the benchmark models we followed the implementations in (1).


# Dependencies

Main dependencies:
PyTorch (1.11.0), PyTorch Lightning (1.6.3), Python (3.9).

Additional (training script dependencies):
torchvision (0.12.0), matplotlib (3.5.1), numpy (1.21.5), h5py (3.7.0), scikit-image (0.19.2), scikit-learn (0.23.2), scipy (1.7.3), pandas (1.4.2).


# Datasets

Our study is based on public datasets.

COVID-19 X-ray database: available from (1).

Stanford Dogs: http://vision.stanford.edu/aditya86/ImageNetDogs/

MNIST: http://yann.lecun.com/exdb/mnist/


# Reference List:
(1) Bassi, P.R.A.S., Dertkigil, S.S.J. & Cavalli, A. Improving deep neural network generalization and robustness to background bias via layer-wise relevance propagation optimization. Nature Communications 15, 291 (2024). https://doi.org/10.1038/s41467-023-44371-z

# Note
Additional code, with demo and dataset samples, along with trained DNNs will be released upon paper acceptance.


