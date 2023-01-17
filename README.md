# Foobar

Foobar is a Python library for dealing with word pluralization.

## Installation

Create respective environments for each repo using environment.yml given in their directories

```bash
conda create -f environment.yml
```

## Attention-Target-Detection Usage

Use ```python Inference Class```   in ```demo.py```  to test gaze estimation.
Initializer requires:
- Weights path

- Main function i-e ```runOverBatch ()```  requires frame and head bounding box list.

## Multimodal-across-domains-gaze-target-detection

Use ```python Inference Class```   in ```elm_infer.py```  to test gaze estimation.
Initializer requires:
- Weights path

- Main function i-e ```runOverBatch ()```  requires frame, depth image and head bounding box list.


