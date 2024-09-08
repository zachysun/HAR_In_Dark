## Models about Video Human Action Recognition in the Dark

This topic is from [UG2 Challenge 2021 Track 2](https://cvpr2023.ug2challenge.org/program21/track2.html) and it's also a course project of NTU EE6222 in 2023 Semester 2.

### Models

- ResNet50 + Late Fusion
- Slowfast Network
- Two-stream Network

<img src="./figs/pipeline.png" width=550></img>



### Usage

```shell
cd main
python main_resnet.py
python main_slowfast.py
python main_twostream.py
```

### Dataset

- [Dataset (ARID)](https://xuyu0010.github.io/arid.html)

- [Dataset (ARID_UG2_2.1)](https://github.com/xuyu0010/ARID_UG2_2.1)



### Related Repositories

- [ARID_v1](https://github.com/xuyu0010/ARID_v1)

- [SlowFast](https://github.com/facebookresearch/SlowFast)
- [pytorch-i3d](https://github.com/piergiaj/pytorch-i3d)



### Related Papers

- [ARID: A New Dataset for Recognizing Action in the Dark](https://arxiv.org/abs/2006.03876)
- [SlowFast Networks for Video Recognition](https://arxiv.org/abs/1812.03982)
- [Quo Vadis, Action Recognition? A New Model and the Kinetics Dataset](https://arxiv.org/abs/1705.07750)