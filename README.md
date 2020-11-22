## WARNING: mltS is still under development, so the code and documentation is very dynamic.

![Workflow](flowchart.png)

## Basic Description

This repo contains the implementation of mltS (mu**l**ti-label l**e**arning based on less-**t**rusted **S**ources) which is an ensemble based learning that  leverages the idea of estimating memebers reliability scores in an ensemble given a small reference collection dataset (step 2 in the figure above). mltS assumes that multi-source multi-label dataset is provided (top figure). This dataset can be generated using multiple multi-label (*U*) algorithms (e.g. PathoLogic, MinPath, mlLGPR, triUMPF, and leADS). Next, mltS performs an iterative procedure to: (a)- execute and train, concurrently, multiple instances of multi-label learning where each instance is designated to learn from a subset of dataset generated by a particular algorithm (step 1 in the figure above); (b) build a discrepancy table of learning instances using a trusted dataset, where each cell in that table indicates a particular learner’s ability to predict the corresponding label (step 2 in the figure above); (c)- estimate and update models specific weights (or reliability scores) using the discrepancy table to demonstrate which model is high likely to produce noisy labels (step 2 in the figure above); and (d)- optimize an overall global parameter vectors by aggregating local models parameters (step 3 in the figure above). Through this approach, mltS is capable to assess the reliability of each model. In addition, mltS can be used for inference in three ways: "meta-predict (mp)", "meta-weight (mw)", and "meta-adaptive (ma)". 

See tutorials on the [GitHub wiki](https://github.com/hallamlab/mltS/wiki) page for more informations and guidelines.

## Citing
<!-- If you find *mltS* useful in your research, please consider citing the following paper: -->
- [TO BE ADDED].

## Contact
For any inquiries, please contact: [arbasher@alumni.ubc.ca](mailto:arbasher@alumni.ubc.ca)
