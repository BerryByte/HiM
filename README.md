# HiM: Hierarchical Mamba with Hyperbolic Embeddings

This repository contains the official implementation of the HiM (Hierarchical Mamba) architecture, which combines state-of-the-art Mamba models with hyperbolic geometry to better represent hierarchical structures.

## About

HiM implements hierarchical representation learning using Lorentz and Poincaré hyperbolic manifolds with Mamba architecture. Our approach leverages the benefits of both Mamba's efficient sequence modeling and hyperbolic geometry's natural representation of hierarchical data.

## Datasets

Datasets Link: https://zenodo.org/records/14036213
Place the downloaded dataset in a directory and update the data_path in example_config.yaml.

```bash
pip install -r requirements.txt
```

If you need CUDA support, uncomment the PyTorch line in requirements.txt or run:
```bash
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu118
```

## Running the Code

### Train the Sentence Mamba Model
The Sentence Mamba model is trained on the SNLI dataset to generate sentence embeddings:
```bash
python train_sentencemamba.py
```
This will save the trained model in the output-1/model-distillation-{timestamp}/final directory.

### Train HiM Models
You can train the HiM model in hyperbolic space either in the Lorentz or Poincaré manifold (load trained Sentence Mamba Model in these two models):
Lorentz Version
```bash
python HIM_l.py
```

Poincaré Version
```bash
python HIM_p.py
```

You may view the .sh files on how to run the codes (either with DDP [HIM_l.sh or HIM_p.sh] or directly using python [run_experiments.sh])

## Attribution and License

For building upon hyperbolic losses and downstream tasks, this project references HiT (Hierarchy Transformers) library by Yuan He: 
- Repository: https://github.com/KRR-Oxford/HierarchyTransformers

For hyperbolic losses, HiM modifies existing HiT's hyperbolic losses by: 
- **Adaptive Curvature**: Dynamic curvature parameter learning for hyperbolic manifolds
- **Loss Functions**: Enhanced numerical stability in hyperbolic distance computations

All original copyright notices from HiT are preserved in modified files.

---

**Note**: This repository contains code submitted for anonymous peer review. Author identities are withheld in accordance with conference/journal requirements. All third-party attributions are maintained as required by their respective licenses.
