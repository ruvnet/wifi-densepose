#!/bin/bash
set -ex
cd ~/wiflow-std-bench

# 1. clone upstream at the pinned commit
if [ ! -d upstream ]; then
  git clone https://github.com/DY2434/WiFlow-WiFi-Pose-Estimation-with-Spatio-Temporal-Decoupling upstream
fi
cd upstream && git checkout 06899d294a0f44709d601a53e91dbf24759daefb && cd ..

# 2. documented deviation: fix upstream import bug (TemporalConvNet does not exist)
sed -i 's/from .tcn import TemporalConvNet/from .tcn import TemporalBlock/; s/'"'"'TemporalConvNet'"'"'/'"'"'TemporalBlock'"'"'/' upstream/models/__init__.py

# 3. venv: torch cu128 (RTX 5080 = sm_120 needs >=2.7; their pin 2.3.1 predates Blackwell)
if [ ! -d venv ]; then
  python3 -m venv venv
  ./venv/bin/pip install -q --upgrade pip
  ./venv/bin/pip install -q torch --index-url https://download.pytorch.org/whl/cu128
  ./venv/bin/pip install -q numpy pandas matplotlib seaborn scikit-learn opencv-python-headless scipy tqdm psutil kagglehub
fi
./venv/bin/python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"

# 4. dataset via kagglehub (anonymous, public dataset)
DS=$(./venv/bin/python -c "import kagglehub; print(kagglehub.dataset_download('kaka2434/wiflow-dataset'))")
echo "dataset at: $DS"

# 5. run.py hardcodes ../preprocessed_csi_data relative to upstream/
ln -sfn "$DS/preprocessed_csi_data" ~/wiflow-std-bench/preprocessed_csi_data

# 6. train with upstream defaults (seed 42 set inside run.py)
../venv/bin/python ../clean_nan.py 2>/dev/null || venv/bin/python clean_nan.py
cd upstream
../venv/bin/python run.py --gpu 0 --batch_size 64 --epochs 50 --output_dir ../train_output
