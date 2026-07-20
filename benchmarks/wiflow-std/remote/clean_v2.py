import numpy as np, os
d = os.path.expanduser('~/wiflow-std-bench/preprocessed_csi_data')
csi = np.load(os.path.join(d, 'csi_windows.npy'), mmap_mode='r+')
zeroed = 0
chunk = 4000
for i in range(0, len(csi), chunk):
    block = csi[i:i+chunk]
    finite = np.isfinite(block)
    bad = (~finite).any(axis=(1, 2)) | (np.abs(np.where(finite, block, 0)).max(axis=(1, 2)) > 1.5)
    if bad.any():
        block[bad] = 0.0
        zeroed += int(bad.sum())
csi.flush()
print(f'zeroed {zeroed} corrupted windows entirely')
