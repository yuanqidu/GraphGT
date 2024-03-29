### GraphGT: Machine Learning Datasets for Graph Generation and Transformation (NeurIPS 2021)

[![website](https://img.shields.io/badge/website-live-brightgreen)](https://graphgt.github.io/)
[![GitHub Repo stars](https://img.shields.io/github/stars/yuanqidu/GraphGT)](https://github.com/yuanqidu/GraphGT/stargazers)

[**Dataset Website**](https://graphgt.github.io/) | [**Paper**](https://openreview.net/forum?id=NYgt9vcdyjm)

**Anouncement** As the data downloading through pip-package is problematic, especially for large files, we encourage you to download the dataset by the downloadable link (clicking the name of the dataset) provided [here](https://graphgt.github.io/).

## Installation

### Using `pip`

To install the core environment dependencies of GraphGT, use `pip`:

```bash
pip install GraphGT
```

**Note**: GraphGT is in the beta release. Please update your local copy regularly by

```bash
pip install GraphGT --upgrade
```

### DataLoader

```bash
import graphgt 
dataloader = graphgt.DataLoader(name=KEY, save_path='./', format='numpy')
```

KEY: 'qm9', 'zinc', 'moses', 'chembl', 'profold', 'kinetics', 'ntu', 'collab', 'n_body_charged', 'n_body_spring', 'random_geometry', 'waxman', 'traffic_bay', 'traffic_la', 'scale_free_{10|20|50|100}', 'ER_{20|40|60}', 'IoT_{20|40|60}', 'authen'.

**All the datasets could be downloaded by the link in [our website](https://graphgt.github.io/), while we are working hard to make sure all of them are downloadable through pip.**

## Cite Us

If you use our dataset in your work, please cite us:

```
@inproceedings{du2021graphgt,
  title={GraphGT: Machine Learning Datasets for Graph Generation and Transformation},
  author={Du, Yuanqi and Wang, Shiyu and Guo, Xiaojie and Cao, Hengning and Hu, Shujie and Jiang, Junji and Varala, Aishwarya and Angirekula, Abhinav and Zhao, Liang},
  booktitle={NeurIPS 2021},
  year={2021}
}
```

## Team
[Yuanqi Du](https://yuanqidu.github.io/) (Leader), Shiyu Wang, Xiaojie Guo, Hengning Cao, Shujie Hu, Junji Jiang, Aishwarya Varala, Abhinav Angirekula, [Liang Zhao](http://cs.emory.edu/~lzhao41/) (Advisor)

## Issue
Please raise a GitHub issue if you have any question.

## Contact
Send us an [email](mailto:ydu6@gmu.edu) or open an issue.
