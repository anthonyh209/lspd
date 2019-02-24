Learned Stochastic Primal-Dual Reconstruction
==================================

The LSPD algorithm is a reconstruction algorithm inspired from the Learned Primal-Dual Reconstruction algorithm (https://arxiv.org/abs/1707.06474). The neural network in this approach is trained against randomness of provided data which allows a subsampling of the operator. It outperforms state of the art reconstruction algorithms with less iterations. The written framework allows a plug-in approach and the framework can be used to try out different operators

### Prerequisites

* [Tensorflow] (https://nodejs.org/en/download/)
* [ODL] (https://github.com/odlgroup)


Pre-trained networks
--------------------
The training duration takes between 10 and 20 hours depending on the setup (number of positions and overall iterations). Please contact to request information about the pretrained weights 

Dependencies
------------
The code is lightly based on the latest version of [ODL](https://github.com/odlgroup/odl/pull/972). Subsampling is achieved through a self-written library package

```
$ pip install https://github.com/odlgroup/odl/archive/master.zip
```

The code also requires the utility library [adler](https://github.com/adler-j/adler) which can be installed via

Author
------------
* **Wontek Hong**
* supervised by **Dr Marta Betcke (UCL) & Dr Andreas Hauptmann (UCL)**
