---

<div align="center">    
 
# V2Iformer

</div>

A new attention-based architecture for tasks such as video frame interpolation and prediction, multi-image summarization, and 3D image flattening. It is based on the uformer, but generalized to consume multiple images and spit out one synthesized image.


<div align="center">    

![My diagram of the V2Iformer Architecture](https://github.com/adam-mehdi/V2Iformer/blob/cc4fb79a7c6472ce338569f839595ab9aa23f8e9/V2Iformer-architecture.png)

</div>

## How to use   
```python
! pip install git+https://github.com/adam-mehdi/V2Iformer.git

import torch
from pytorch_lightning import Trainer
from v2iformer.gan import GAN

# dummy dataset
b, c, f, h, w = 4, 3, 5, 32, 32
dataloader = [(torch.randn(b, c, f, h, w), torch.randn(b, c, 1, h, w)) for i in range(100)]

model = GAN(c, f, h, w)

trainer = Trainer()
trainer.fit(model, dataloader)
```

## Citations

```
@misc{bertasius2021spacetime,
      title={Is Space-Time Attention All You Need for Video Understanding?}, 
      author={Gedas Bertasius and Heng Wang and Lorenzo Torresani},
      year={2021},
      eprint={2102.05095},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}

@misc{wang2021uformer,
      title={Uformer: A General U-Shaped Transformer for Image Restoration}, 
      author={Zhendong Wang and Xiaodong Cun and Jianmin Bao and Jianzhuang Liu},
      year={2021},
      eprint={2106.03106},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}

```   

Portions of the code code was built on the following repositories:
- [uformer-pytorch by lucidrains](https://github.com/lucidrains/uformer-pytorch)
- [my FastTimesFormer](https://github.com/adam-mehdi/FastTimeSformer)
