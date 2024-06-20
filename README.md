## Dependencies

See `requirements.txt`.

## Usage

Main program:
```shell
python main.py --dataset [DATASET] --ratio [RATIO] --method [METHOD]
```

Available datasets:
- MUTAG
- NCI1
- ENZYMES
- NCI109
- PROTEINS
- PTC_MR
- IMDB-BINARY
- IMDB-MULTI
- REDDIT-BINARY
- REDDIT-MULTI

Avaiable methods(See `main.py` for details.):
- SDSumm
- sc (stands for 'spectral clustering')
- mgc
- sgc
- graphzoom
- LV_nei
- LV_edge

