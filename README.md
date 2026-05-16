# TreeAttMSA: Decoupling Cross-Modal Contexts with Tree-Structured Graph Attention

**Paper:** "TreeAttMSA: Decoupling Cross-Modal Contexts with Tree-Structured Graph Attention for Multimodal Sentiment Analysis"  
**Journal:** The Visual Computer  
**DOI:** [To be added after publication]  
**GitHub:** https://github.com/xiaohanyue1/TREEATTMSA

## Requirements

- Python >= 3.8
- PyTorch >= 2.10
- NumPy

Install required packages:
```bash
pip install -r requirements.txt

## Datasets
The CMU-MOSI and CMU-MOSEI datasets were created and released by the Multicomp Lab at Carnegie Mellon University.
They support the findings of this study and are publicly available at https://github.com/thuiar/Self-MM.
Both datasets are released under the MIT License, which permits use and publication.
The CH-SIMS dataset was created and released by Wenmeng Yu et al, is also publicly accessible at https://thuiar.github.io/sims.github.io/chsims, and is recommended for academic use with proper citation.
The subject images in Figures 1, 2, 7 and 8 are from the CMU-MOSI, CMU-MOSEI, and CH-SIMS datasets, all of which are publicly downloadable.

## Run
python run.py
The main body of the paper presents the hyperparameter settings used to achieve optimal results on each dataset.
