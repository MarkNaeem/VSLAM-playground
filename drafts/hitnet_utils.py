import torch
import pytorch_lightning as pl
from TinyHITNet.models import build_model

class PredictModel(pl.LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.model = build_model(self.hparams)
#         self.model = self.model.cuda()
        

    def forward(self, left, right):
        left = left * 2 - 1
        right = right * 2 - 1
        return self.model(left, right)


def np2torch(x, t=True, bgr=False):
    if len(x.shape) == 2:
        x = x[..., None]
    if bgr:
        x = x[..., [2, 1, 0]]
    if t:
        x = np.transpose(x, (2, 0, 1))
    if x.dtype == np.uint8:
        x = x.astype(np.float32) / 255
    x = torch.from_numpy(x.copy())
    return x


@torch.no_grad()
def predict(model, left, right):
    # left = np2torch(left, bgr=True).cuda().unsqueeze(0)
    # right = np2torch(right, bgr=True).cuda().unsqueeze(0)
    left = np2torch(left, bgr=True).unsqueeze(0)
    right = np2torch(right, bgr=True).unsqueeze(0)
    return model(left, right)
