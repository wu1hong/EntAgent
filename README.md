# EntAgent
This repository contains the dataset and code for evaluating our proposed Entity Linking Agent for Question Answering (QA), introduced in the paper ["An Entity Linking Agent for Question Answering"](https://arxiv.org/abs/2508.03865) by Yajie Luo, Yihong Wu.
## Install
Run command ```pip install -r requirements.txt``` to install necessary libraries.
Use the following command to install [ScalableVectorSearch](https://github.com/intel/ScalableVectorSearch), a vector search library and a replacement for Faiss.

```
# Install SVS for indexing (feel free to install on your customized dir) 
git clone https://github.com/intel/ScalableVectorSearch
cd ScalableVectorSearch
# Install svs using pip (don't use uv pip)
pip install bindings/python
```

## Data Preprocess
Use the following command to download datasets:

```bash data/2wiki_download.sh```
```bash data/triviaqa_download.sh```

Use the following command to preprocess datasets under the ```data``` folder

```python preprocess.py```


## Inference
1. Modify the ```config.toml``` file in ```ent_agent``` folder
2. Launch vLLM with command ```bash serve.sh```
3. Run command ```python main.py``` for entity linking task
4. Run command```python main_qa.py``` for QA evaluation

## Citation
If you use our code in your research, please cite our work:
```
@article{luo2025entity,
  title={An Entity Linking Agent for Question Answering},
  author={Luo, Yajie and Wu, Yihong and Li, Muzhi and Mo, Fengran and Sun, Jia Ao and Wang, Xinyu and Ma, Liheng and Zhang, Yingxue and Nie, Jian-Yun},
  journal={arXiv preprint arXiv:2508.03865},
  year={2025}
}
```
