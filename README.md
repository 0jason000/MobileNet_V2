# MobileNet_V2

The MobileNetV2 architecture is based on an inverted residual structure where the input and output of the residual block are thin bottleneck layers opposite to traditional residual models which use expanded representations in the input an MobileNetV2 uses lightweight depthwise convolutions to filter features in the intermediate expansion layer.

The architectural definition of each network refers to the following papers:

[1] Mark Sandler, Andrew Howard, Menglong Zhu, Andrey Zhmoginov, Liang-Chieh Chen. [MobileNetV2: Inverted Residuals and Linear Bottlenecks](https://arxiv.org/pdf/1801.04381.pdf). arXiv preprint arXiv: 1801.04381.

## Examples

***

### Train

- The following configuration uses 8 GPUs for training. The magnification factor is set to 1.0 and the image input size is set to 224.

  ```shell
  mpirun -n 8 python train.py --arch mobilenetv2 --alpha 1.0 --resize 224 --data_url ./dataset/imagenet
  ```

  output:

  ```text
  Epoch:[0/200], step:[2502/2502], loss:[4.676/4.676], time:872084.093, lr:0.10000
  Epoch time:883614.453, per step time:353.163, avg loss:4.676
  Epoch:[1/200], step:[2502/2502], loss:[4.452/4.452], time:693370.244, lr:0.09998
  Epoch time:693374.709, per step time:277.128, avg loss:4.452
  Epoch:[2/200], step:[2502/2502], loss:[3.885/3.885], time:685880.388, lr:0.09990
  Epoch time:685884.401, per step time:274.134, avg loss:3.885
  Epoch:[3/200], step:[2502/2502], loss:[3.550/3.550], time:689409.851, lr:0.09978
  Epoch time:689413.237, per step time:275.545, avg loss:3.550
  Epoch:[4/200], step:[2502/2502], loss:[3.371/3.371], time:692162.583, lr:0.09961
  Epoch time:692166.163, per step time:276.645, avg loss:3.371

  ...
  ```


### Eval

- The following configuration for eval. The magnification factor is set to 0.75 and the image input size is set to 192.

  ```shell
  python validate.py --arch mobilenetv2 --alpha 0.75 --resize 192 --pretrained True --data_url ./dataset/imagenet
  ```

  output:

  ```text
  {'Top_1_Accuracy': 0.6922876602564103, 'Top_5_Accuracy': 0.8871594551282052}
  ```
  