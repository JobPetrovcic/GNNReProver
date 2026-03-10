# Combining Textual and Structural Information for Premise Selection in Lean

This repository contains the official implementation of the paper **"Combining Textual and Structural Information for Premise Selection in Lean"**, accepted at the **NeurIPS 2025 Workshop on MATH-AI**.

**Authors:** Job Petrovčič, David Narváez, Ljupčo Todorovski  
*(University of Ljubljana, Jožef Stefan Institute)*

[**Read the Paper on arXiv**](https://arxiv.org/abs/2510.23637)

**Note:** This codebase is a modification of [LeanDojo's ReProver](https://github.com/lean-dojo/ReProver). To extract the data used for the experiments please check out the modified [LeanDojo](https://github.com/JobPetrovcic/LeanDojoModification).

## Overview

Premise selection is a bottleneck for scaling theorem proving in large formal libraries. Existing language-based methods often treat premises in isolation, ignoring the web of dependencies connecting them.

We present a graph-augmented approach that combines:
1.  **Dense text embeddings** of Lean formalizations.
2.  **Graph Neural Networks (GNNs)** over a heterogeneous dependency graph (capturing state–premise and premise–premise relations).

On the **LeanDojo Benchmark**, our method outperforms the ReProver baseline by over **25%** across standard retrieval metrics, demonstrating the value of relational information in theorem proving.

## Citation

If you use this code or our results in your research, please cite:

```bibtex
@inproceedings{petrovcic2025combining,
  title={Combining Textual and Structural Information for Premise Selection in Lean},
  author={Petrov\v{c}i\v{c}, Job and Narv\'{a}ez, David and Todorovski, Ljup\v{c}o},
  booktitle={NeurIPS 2025 Workshop on MATH-AI},
  year={2025},
  url={https://arxiv.org/abs/2510.23637}
}
