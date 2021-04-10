import torch
import torchvision as tv
import pytorch_lightning as pl

from .oscar.modeling_bert import BertForImageCaptioning
from .efficientnet.net import EfficientNet


class Monocle(pl.LightningModule):

    def __init__(self, *args, **kwargs):
        super().__init__()
        
        self.backbone = EfficientNet.from_pretrained(
            'efficientnet-b3',
            in_channels=3,
            include_top=False)
        self.bert = BertForImageCaptioning

    def forward(self, imgs, boxes):
        tv.ops.roi_align()
    
    def training_step(self, batch, batch_idx):
        x, _ = batch

        # encode
        x = x.view(x.size(0), -1)
        z = self.encoder(x)

        # decode
        recons = self.decoder(z)

        # reconstruction
        reconstruction_loss = nn.functional.mse_loss(recons, x)
        return reconstruction_loss

     def validation_step(self, batch, batch_idx):
        x, _ = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        recons = self.decoder(z)
        reconstruction_loss = nn.functional.mse_loss(recons, x)
        self.log('val_reconstruction', reconstruction_loss)

     def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.0002)