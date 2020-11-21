![Workflow](flowchart.png)

## Basic Description
This repo contains an implementation of mltS (mu**l**ti-label l**e**arning based on less-**t**rusted **S**ources) that leverages the idea of estimating memebers reliability scores in an ensemble given a small reference collection dataset. Specifically, mltS performs an iterative procedure to: (a)- train multiple local learners where each memeber is assigned to learn outputs from an algorithm; (b) build a discrepancy table of learners describing algorithms performances; (c)- estimate the reliability of learners; and (d)- optimize an overall global parameter vectors. Through this approach, mltS is capable to assess the reliability of each model while providing a solution to voting during prediction in an adaptive manner. mltS was evaluated on the pathway prediction task using 10 multi-organism pathway datasets, where the experiments revealed that mltS achieved very compelling and competitive performances against the state-of-the-art pathway inference algorithms.

## Tutorials
All of our tutorials are available on the GitHub wiki page.

## Citing
If you find *mltS* useful in your research, please consider citing the following paper:
- [TO BE ADDED].

## Contact
For any inquiries, please contact: [arbasher@alumni.ubc.ca](mailto:arbasher@alumni.ubc.ca)
