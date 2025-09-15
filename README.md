## Research Paper Assets

This folder contains a self-contained snapshot of code and artifacts relevant to the paper.
There are no cloud-provider (AWS/SageMaker) dependencies, and no personal paths.

### Contents
- `train_joint_model.py`: Standalone training script defining the model and a normal local training loop.
- `heuristic_candidates.py`: Tools to compute features on FITS frames and rank heuristic candidates.
- `review_labels.py`: Matplotlib-based labeling review UI (grid) for FITS files.
- `joint_frame_seq.pt`: Trained model weights (PyTorch state_dict).
- `requirements.txt`: Minimal Python dependencies for this folder.

### Installation
Use a fresh virtual environment, then install requirements:
```
python -m venv .venv
. .venv/Scripts/activate  # Windows PowerShell: .venv\\Scripts\\Activate.ps1
pip install -r requirements.txt
```

### Training the Model (Local)
The script matches the optimizer and loss configuration used in training.
Example:
```
python train_joint_model.py \
  --mixed-dir PATH/TO/mixed_frames \
  --seq-dirs PATH/TO/toi5443,PATH/TO/toi5595,PATH/TO/toi5944 \
  --label-csv-primary PATH/TO/mixed_labels.csv \
  --extra-label-dirs PATH/TO/labels5443,PATH/TO/labels5595,PATH/TO/labels5944 \
  --epochs 20 --frame-batch 32 --seq-batch 8 --img-size 128 --encoder spatial \
  --model-out joint_frame_seq.pt
```

Notes:
- Optimizer: Adam (lr=1e-3, weight_decay=1e-4)
- Loss: BCEWithLogitsLoss with positive-class weighting (computed from data)
- GPU is used automatically if available; CPU otherwise

### Using the Pretrained Weights
You can load `joint_frame_seq.pt` in PyTorch via:
```python
import torch
from train_joint_model import JointFrameSeqModel

model = JointFrameSeqModel(encoder_type='spatial')
state = torch.load('joint_frame_seq.pt', map_location='cpu')
model.load_state_dict(state, strict=False)
model.eval()
```

### Heuristic Candidates
Compute features and rank candidates from a folder of FITS images:
```
python heuristic_candidates.py FOLDER --out-features features.csv --out-candidates candidates.csv
```

### Labeling Review UI
Launch the grid-based reviewer (provide your own paths):
```
python review_labels.py --folder /path/to/fits --csv labels.csv --filter bad --rows 3 --cols 3
```

### Privacy
This folder contains no personal information or hard-coded personal paths.


