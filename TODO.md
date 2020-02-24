## Current TODO

- * [ ] Search for some suitable non-image dataset
  - * [ ] heck some perturbations here

1. Wasserstein Distance
2. Sinkhole
3. Image deformation

- Do some literature search on image deformation for adversarial attacks 

Tex file:
- Loss function
- Actual model
- Perturbation model

Build IIC model on non-image dataset

## Some Papers
- Read paper thoroughly on [information clustering from Nico](http://openaccess.thecvf.com/content_ICCV_2019/papers/Ji_Invariant_Information_Clustering_for_Unsupervised_Image_Classification_and_Segmentation_ICCV_2019_paper.pdf)
  - Have a look at their Github repo
- Read paper from Guido on [unique information](https://arxiv.org/abs/1709.07487)

- Read papers from Pradeep:
  - [Deep Infomax](https://arxiv.org/pdf/1808.06670.pdf)
  - [Critics to it](https://arxiv.org/pdf/1907.13625.pdf)
  - A more recent theoretical [paper by Merkh and Guido](https://arxiv.org/pdf/1906.05460.pdf) is here on maximizers of the factorized mutual information

  - [Deep image prior](https://sites.skoltech.ru/app/data/uploads/sites/25/2018/04/deep_image_prior.pdf)

## Ideas for presentation:
- What to focus on?
  - summarize abstract of paper
  - thesis 
    - Aim & goal
    - what issues are solved?
  - methodology
  - what kind of data analysis?
  - briefly lsit results
  - short summary of discussion
  - see conclusion of paper
See [this link for more information](https://tipsforresearchpapersandessays.blogspot.com/2008/10/how-to-summarize-research-paper.html)

- introduce K-means as well 

1. What is clustering? Problem description? Toy example?
2. Explain k-means
3. Present idea of DeepInfomax
4. Critics to it
5. Methodology of Invariant paper

## Ideas for summary:
What papers to include:
  - Deep Infomax
  - Critics on MI maximization
  - Invariant Information Clustering (IIC)
  (- Deep Image Prior)
  - Relation to [co-clustering](https://www.cs.utexas.edu/users/inderjit/public_papers/kdd_cocluster.pdf)

- What is clustering and why to use it?
- Drawbacks of classical clustering algorithms
- How to use Mutual Information for clustering
- What is DeepInfomax -> issues and method
- How does IIC fix this issue
- Critics to plain maximization of MI


- Prior matching? (DIM paper)
- Discriminator function? for DIM paper
- Joint probability?
- IIC principle is equivalent to distilling their shared abstract content (co-clustering)
- IMSAT

- Read paper on Adam
- See image of blackboard
- Wasserstein kernel for diffusion -
- see whether model or dependent function is cause for smoothness and power of IIC
  - Utilize very simple model for encoding -> see what happens

- In order to to experiments:
  - get IIC to run on some non-image dataset -> Iris dataset?
  - For perturbation:
    - Check some odd perturbations -> How good are they?
    - Use normal Gaussian as noise
    - Check Wasserstein diffusion kernel 
    - Maybe then try to perturbate some image dataset
  - For model testing:
    - Get simple non-image dataset
    - get very simple neural classification algorithm (Guidos Idea) and check whether results come from model or MI

