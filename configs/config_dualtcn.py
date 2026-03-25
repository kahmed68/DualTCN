"""config_p9d.py — P9d: LateTimePCRN + standard input + d_sf auxiliary head."""
from config_v4 import *

# Weight for the auxiliary seafloor-depth (d_sf = d₁+d₂) Huber loss.
# Added on top of the standard WeightedPCRNLoss in train_p9d.py.
DSF_AUX_WEIGHT = 0.5
