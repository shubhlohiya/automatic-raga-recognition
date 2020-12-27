## Automatic Raga Recognition in Indian Classical Music

This repository contains the unofficial implementation of the [DeepSRGM](https://archives.ismir.net/ismir2019/paper/000064.pdf) paper by Sathwik Tejaswi Madhusudhan and Girish Chowdhary.  

This was done as a part of the course project of **IE 643 - Deep Learning: Theory and Practice**, under Prof. P. Balamurugan.  

The details about the project, previous work, our contributions, and our results can be found in the [report](./report.pdf).  

Repository Structure:
```
automatic-raga-recognition/
├── README.md
├── data
│   └── README.md
├── models
│   ├── gru_30_checkpoint.pth
│   └── lstm_25_checkpoint.pth
├── report.pdf
└── src
    ├── __init__.py
    ├── attention_layer.py
    ├── dataset_preprocessing.ipynb
    ├── deepSRGM.py
    ├── evaluate.ipynb
    ├── original-project-implementation
    │   ├── dataset_preprocessing-40.ipynb
    │   ├── deepSRGM-40rag.ipynb
    │   └── deepSRGM.ipynb
    ├── test_utils.py
    ├── train.ipynb
    └── train_utils.py

```

**Note:** The `original-project-implementation` directory contains the code for the original implementation during the IE643 course-project. This code has since been cleaned up and refactored. It has been included only to provide credibility to the claims in `report.pdf`. 

***
<p align='center'>Created with :heart: by <a href="https://www.linkedin.com/in/lohiya-shubham/">Shubham Lohiya</a> & <a href="https://www.linkedin.com/in/swarada-bharadwaj-5145a1174/">Swarada Bharadwaj</a></p>