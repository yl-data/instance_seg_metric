# Object dice metric index for instance segmentation

## Features
- Support object-level dice metric, which reference to [Gland Segmentation in Colon Histology Images: The GlaS Challenge Contest](https://arxiv.org/abs/1603.00275)
    - Support semantic dice too
- Inherited from `pycocotools.cocoeval.COCOeval`, support coco style dataset, and easy to integration into common detection frameworks
- Easy and straightforward run the example directly

## Run example

```bash
pip3 install boxx pycocotools
python3 dice_metric.py
```