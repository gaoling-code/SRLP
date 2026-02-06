# SRLP: a semantic representation learning perspective-based approach for multimodal science question answering.

This repository contains the code for the paper "A Semantic Representation Learning Perspective-based Approach for Multimodal Science Question Answering". 

## Abstract

Multimodal science question answering performs multimodal learning and logical reasoning based on given visual and text input to answer science questions. For a particular input pair, visual and text express the common semantic meaning through different modal carriers. The semantic meanings expressed by the visual and text modal carriers sourcing different pairs are independent. However, the high entanglement between modal and semantic representation and the weak semantic differences between different pairs hinder the learning of visual and text semantic representation. In this work, we introduce a semantic representation learning perspective-based approach (SRLP) for multimodal science question answering. First, we maximize the matching degree between visual and text semantic representations within the particular pair by disentanglement and agreement constraints. At the same time, we combine contrastive learning with disentangled representation learning to enhance the cross-modal semantic representation learning of the particular pair. The learned cross-modal semantic representation is finally fed into the decoder to generate the answer text. Experiments on two multimodal science question answering datasets demonstrate the excellent performance of SRLP, proving the effectiveness of cross-modal agreement and contrastive learning.

## Requirements

Install all required python dependencies:

```
pip install -r requirements.txt
```

## Datasets

The models are trained and evaluated on two open-source datasets:
- ScienceQA: Available at:
  - [Hugging Face Repository](https://huggingface.co/cooelf/vision_features/tree/main)
  - [Google Drive Link 1](https://drive.google.com/file/d/13B0hc_F_45-UlqPLKSgRz-ALtFQ8kIJr/view?pli=1)
  - [Google Drive Link 2](https://drive.google.com/drive/folders/1w8imCXWYn2LxajmGeGH_g5DaL2rabHev)
- CK12-QA dataset: Accessible at [Textbook Question Answering](https://prior.allenai.org/projects/tqa).


The processed vision features for ScienceQA are available at [huggingfcae vision features](https://huggingface.co/cooelf/vision_features/tree/main). `all-MiniLM-L6-v2` and `unifiedqa-t5-base` can be downloaded at [huggingface sentence-transformers](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) and [huggingface unifiedqa-t5-base](https://huggingface.co/allenai/unifiedqa-t5-base).


The folder with all related files looks like:

```
SRLP
├── assets
├── results
│   ├── base_pretrained_scienceqa
│   │   ├── answer
│   │   │   ├── ...
│   │   ├── rationale
│   │   │   ├── ...
├── models
│   ├── all-MiniLM-L6-v2
│   ├── unifiedqa-t5-base
├── data
│   ├── vision_features
│   ├── scienceqa
```

## Usage

To inference with pretrained weights (`results/base_pretrained_scienceqa/`), run `run_eval_scienceqa.sh`.

To train the model by yourself, please run `run_train_scienceqa.sh`.

## Acknowledgements 

We highly thank "Multimodal Chain-of-Thought Reasoning in Language Models". [paper](https://arxiv.org/abs/2302.00923), [code](https://github.com/amazon-science/mm-cot) and "Boosting the Power of Small Multimodal Reasoning Models to Match Larger Models with Self-Consistency Training". [paper](https://arxiv.org/abs/2311.14109), [code](https://github.com/chengtan9907/mc-cot)

## Reference
```
@article{Ling_2026_SRLP,
  title={A Semantic Representation Learning Perspective-based Approach for Multimodal Science Question Answering},
  author={[Ling Gao](https://scholar.google.com.hk/citations?user=tl-cCNUAAAAJ&hl=zh-CN), Nan Sheng, Rui Song and Hao Xu},
  journal={},
  year={2026}
}
```

