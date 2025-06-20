# EntAgent

## Data Preprocess
Use the following command to download datasets:

```bash 2wiki_download.sh```

Use the following command to preprocess datasets under the ```data``` folder

```python preprocess.py```

## Inference
1. Modify the ```config.toml``` file in ```ent_agent``` folder
2. Launch Vllm with command ```bash serve.sh```
3. Run command ```python main.py```
