# ARES Framework

This repository contains the implementation and data used in the paper [Adaptive Optimization Framework for Control and Verification of Cyber Physical Systems](https://link.springer.com/chapter/10.1007/978-3-662-54580-5_17 ), published at [TACAS 2017](https://etaps.org/index.php/2017/tacas). To cite the work, you can use:

```
@inproceedings{DBLP:conf/tacas/LukinaEHBYTSG17,
  author    = {Anna Lukina and
               Lukas Esterle and
               Christian Hirsch and
               Ezio Bartocci and
               Junxing Yang and
               Ashish Tiwari and
               Scott A. Smolka and
               Radu Grosu},
  title     = {{ARES:} Adaptive Receding-Horizon Synthesis of Optimal Plans},
  booktitle = {{TACAS} {(2)}},
  series    = {Lecture Notes in Computer Science},
  volume    = {10206},
  pages     = {286--302},
  year      = {2017}
}

```
## Abstract

We present an optimization-based framework for control and verification of cyber-physical systems that synthesizes actions steering a given system towards a specified state. We demonstrate that our algorithm can be applied to any system modeled as a controllable Markov decision process equipped with a cost (reward) function. A key feature of the procedure we propose is its automatic adaptation to the performance of optimization towards a given global objective. Combining model-predictive control and ideas from sequential Monte-Carlo methods, we introduce a performance-based adaptive horizon and implicitly build a Lyapunov function that guarantees convergence. Experiments on two model problems, birds reaching a V-formation from an arbitrary configuration and covering an arbitrary area by a team of drones, illustrate successful application of our framework. Statistical model-checking is used to verify our algorithm and assess its reliability.

## To reproduce

Run ``main.p`` to obtain results in the paper.

Folder "ARES" contains work-in-progress re-implementation in C++.
