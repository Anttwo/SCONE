# SCONE

<div align="center">
<h2>
SCONE: Surface Coverage Optimization in Unknown Environments<br> by Volumetric Integration
<p></p>

<a href="https://github.com/Anttwo">Antoine Guédon</a>&emsp;
<a href="https://imagine.enpc.fr/~monasse/">Pascal Monasse</a>&emsp;
<a href="https://vincentlepetit.github.io/">Vincent Lepetit</a>&emsp;

<img src="./docs/gifs/fushimi.gif" alt="fushimi.gif" width="400"/>
<img src="./docs/gifs/museum.gif" alt="museum.gif" width="400"/> <br>
<img src="./docs/gifs/pantheon.gif" alt="pantheon.gif" width="400"/>
<img src="./docs/gifs/colosseum.gif" alt="colosseum.gif" width="400"/>

</h2>
</div>

Official PyTorch implementation of [**SCONE: Surface Coverage Optimization in Unknown Environments by Volumetric Integration**](https://arxiv.org/abs/2208.10449) (NeurIPS 2022, Spotlight).

<div align="center">
:warning: :warning: :warning: :warning: :warning: :warning: :warning: :warning: :warning: :warning: :warning: :warning: 
</div>
<br>

**We released a new model called [MACARONS (CVPR 2023)](https://github.com/Anttwo/MACARONS) which is a direct improvement of SCONE.<br> 
MACARONS adapts the approach from SCONE to a fully self-supervised pipeline: It learns simultaneously to explore and reconstruct 3D scenes from RGB images only (there is no need for 3D ground truth data nor depth sensors). <br>
The codebase of MACARONS includes an entire, updated version of SCONE's code as well as detailed instructions for generating the ShapeNet training data we used for SCONE. <br>
Please refer to MACARONS' repo to find all the information you need.** <br>

<div align="center">
  Thank you! <br><br>
  :warning: :warning: :warning: :warning: :warning: :warning: :warning: :warning: :warning: :warning: :warning: :warning:
</div>

<br>
This repository currently contains:

- scripts to initialize and train models
- evaluation pipelines to reproduce quantitative results

**Note**: We will add **installation guidelines**, **training data generation scripts** and **test notebooks** as soon as possible to allow for reproducibility.

<details>
<summary>If you find this code useful, don't forget to <b>star the repo :star:</b> and <b>cite the paper :point_down:</b></summary>

```
@inproceedings{guedon2022scone,
  title={{SCONE: Surface Coverage Optimization in Unknown Environments by Volumetric Integration}},
  author={Gu\'edon, Antoine and Monasse, Pascal and Lepetit, Vincent},
  booktitle={{Advances in Neural Information Processing Systems}},
  year={2022},
}
```

</details>

<details>
<summary><b>Major code updates :clipboard:</b></summary>

- 11/22: first code release

</details>

## Installation :construction_worker:

We will add more details as soon as possible.

## Download Data

### 1. ShapeNetCore

We generate training data for both occupancy probability prediction and coverage gain estimation from [ShapeNetCore v1](https://shapenet.org/). <br>
We will add the data generation scripts and corresponding instructions as soon as possible.

### 2. Custom Dataset of large 3D scenes

We conducted inference experiments in large environments using 3D meshes downloaded on the website Sketchfab under the CC license. <br>
3D models courtesy of [Brian Trepanier](https://sketchfab.com/CMBC), [Andrea Spognetta](https://sketchfab.com/spogna), and [Vr Interiors](https://sketchfab.com/vrInteriors). <br>
We will add more details as soon as possible.

## How to use :rocket:

We will add more details as soon as possible.

## Further information :books:

We adapted the code from [Phil Wang](https://github.com/lucidrains/se3-transformer-pytorch/blob/main/se3_transformer_pytorch/spherical_harmonics.py) to generate spherical harmonic features. <br>
We thank him for this very useful harmonics computation script! <br>

We also thank [Tom Monnier](https://www.tmonnier.com/) for his Markdown template, which we took inspiration from.
