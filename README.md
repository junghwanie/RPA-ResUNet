## RPA-ResUNet
### Overview
RPA-ResUNet: Refined Position of Attention ResUNet uses [ResUNet++](https://arxiv.org/abs/1911.07067) as its baseline and applies the motivation from the [CBAM](https://arxiv.org/abs/1807.06521) paper. It was trained and evaluated on Windows, it will also work for MacOS.
</br>
This code was written as part of the [DACON](https://dacon.io/) project with 2nd generation of 'DACrew' and developed by [`@junghwanie`](https://github.com/junghwanie).

### Dependencies
- Python 3.9.7
- Pytorch 1.13.1
- Numpy 1.23.0

### Datasets
- Data Science Bowl 2018 </br>
To download the DSB 2018 dataset, you must request access to that [URL](https://www.kaggle.com/c/data-science-bowl-2018/data) and provides simple exploratory data analysis of data used with the EDA.ipynb.

### Quick start
- environments:
`pip install -r requirements.txt`
- Training:
`python main.py`