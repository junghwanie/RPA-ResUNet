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

### Results
| Method | RPA-ResUNet | ResUNet++ | U-Net |
| --- | --- | --- | --- |
| mIoU | 0.6992 | 0.7164 | 0.6851 |

**Qualitative results** visualized on the DSB 2018 dataset with RPA-ResUNet.
From the left: Image, Ground truth, Prediction.</br>
<img src="https://github.com/user-attachments/assets/efbafe6d-39f0-46d6-9e58-901831d9cdfb" width="150" height="150">
<img src="https://github.com/user-attachments/assets/1f3de0b1-6027-4812-bde9-d448bb9d1eec" width="150" height="150">
<img src="https://github.com/user-attachments/assets/24c3058b-5423-497b-af12-7193919576d7" width="150" height="150"></br>
<img src="https://github.com/user-attachments/assets/af434c65-1a2c-4d4a-8072-8b40f5b66166" width="150" height="150">
<img src="https://github.com/user-attachments/assets/e74badc0-16df-4659-bb4a-b528b89b69aa" width="150" height="150">
<img src="https://github.com/user-attachments/assets/f9fb9ceb-33c8-4caa-aa6c-6641c2c35af3" width="150" height="150">

### Quick start
- environments:
`pip install -r requirements.txt`
- Training:
`python main.py`