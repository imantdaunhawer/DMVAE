# Disentangling Multimodal Variational Autoencoder

Official code to supplement the paper [Self-supervised Disentanglement of
Modality-specific and Shared Factors Improves Multimodal Generative
Models](https://mds.inf.ethz.ch/fileadmin/user_upload/gcpr_daunhawer_camera_ready.pdf)
published at [GCPR
2020](https://link.springer.com/chapter/10.1007/978-3-030-71278-5_33). This
repository contains a pytorch implementation of the disentangling multimodal
variational autoencoder (DMVAE) and the code to run the experiments from our
paper.

## Installation

```bash
# set up environment
$ conda env create -f environment.yml  # install dependencies
$ conda activate dmvae                 # activate environment
```

## Paired MNIST experiment
```bash
$ cd mmmnist
$ ./run_jobs                     # create dataset and run experiment
$ tensorboard --logdir runs/tmp  # monitor training
```

## MNIST/SVHN experiment
```bash
$ cd mnist_svhn
$ python make_mnist_svhn.py      # create dataset
$ ./run_jobs                     # run experiment
$ tensorboard --logdir runs/tmp  # monitor training
```

## Post-hoc analysis

The tensorboard logs contain a lot of metrics (likelihood values,
classification accuracies, etc.), but not the complete evaluation; for
instance, they do not include the coherence values nor the the unconditionally
generated samples and FID values with ex-post density estimation. To compute
these, run the post-hoc analysis using the script `post_hoc_analysis.py` or,
more conveniently, using the bash script `post_hoc_analysis_batch` as follows:
```
$ ./post_hoc_analysis_batch <path_to_experiment> <logdir>
```
where `path_to_experiment` is the directory of the experiment (e.g.,
`$PWD/mmmnist`) and `logdir` denotes directory with the logfiles for the
respective experiment (e.g., `$PWD/mmmnist/runs/tmp/version_x`). Results from
the post-hoc analysis are saved to the respective `logdir`.  There, you will
find quantitative results in `results.txt` and qualitative results in the form
of png images.

## BibTeX

If you find this project useful, please cite our paper:
```bibtex
@article{daunhawer2020dmvae,
  author    = {Imant Daunhawer and
               Thomas M. Sutter and
               Ricards Marcinkevics and
               Julia E. Vogt},
  title     = {Self-supervised Disentanglement of Modality-Specific and Shared Factors
               Improves Multimodal Generative Models},
  booktitle = {German Conference on Pattern Recognition},
  year      = {2020},
}
```
