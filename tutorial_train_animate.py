from share import *

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from tutorial_dataset import MyDataset
from aldm.logger import ImageLogger
from aldm.model import create_model, load_state_dict


# Configs
resume_path = './models/reference_sd15_ini.ckpt'
batch_size = 2
logger_freq = 300
learning_rate = 1e-5


# First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
model = create_model('./models/aldm_v15.yaml').cpu()
model.load_state_dict(load_state_dict(resume_path, location='cpu'))
model.learning_rate = learning_rate


# Misc
dataset = MyDataset()
dataloader = DataLoader(dataset, num_workers=0, batch_size=batch_size, shuffle=True)
logger = ImageLogger(batch_frequency=logger_freq)
trainer = pl.Trainer(gpus=1, precision=32, callbacks=[logger])


# Train!
trainer.fit(model, dataloader)
