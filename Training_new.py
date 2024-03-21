import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.optim.lr_scheduler import ReduceLROnPlateau
from snntorch import functional as SF

import config
import torch
from SegFormer import SegFormer
from snntorch import utils
from Dataloader_copy import GetData

class accuracy_calc:
    '''
    calculate accuracy for the batch  
    '''
    def __call__(self, spk_rec, targets):
        return SF.accuracy_rate(spk_rec, targets)

class SegFormerModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = SegFormer(
            in_channels = config.in_channels,
            widths=config.widths,
            depths=config.depths,
            all_num_heads=config.all_num_heads,
            patch_sizes=config.patch_sizes,
            overlap_sizes=config.overlap_sizes,
            reduction_ratios=config.reduction_ratios,
            mlp_expansions=config.mlp_expansions,
            decoder_channels=config.decoder_channels,
            scale_factors=config.scale_factors,
            num_classes=config.num_classes,
        )
        self.criterion = SF.ce_rate_loss()
        self.metrics = accuracy_calc()

    def custom_forward_pass(self, data, num_steps= config.num_steps):
        '''
        i think this will take care of the num steps
        '''
        spk_rec = []
        utils.reset(self.model)  # resets hidden states for all LIF neurons in net
        for step in range(num_steps):
            spk_out = self.model(data[step])
            spk_rec.append(spk_out)

        return torch.stack(spk_rec)
    
    def forward(self,x):
        #i guess yaha modify maarna hai
        return self.custom_forward_pass(x)
    
    def process(self, image, segment):
        out=self(image)
        loss= self.criterion(out, segment)
        accuracy = self.metrics(out, segment)
        return loss, accuracy
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=config.learning_rate)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=config.lr_factor, patience=config.lr_patience, verbose=True)
        return {'optimizer': optimizer, 'lr_scheduler': scheduler, 'monitor' : 'val_loss'}
    
    def training_step(self, batch, batch_idx):
        image, segment = batch #YAHA SE CHANGES KAR SAKTE HO YE ANHI PATA HAI 
        loss, accuracy = self.process(image, segment)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_iou', accuracy, on_step=True, on_epoch=True, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        image, segment = batch #YAHA SE BHI
        loss, accuracy = self.process(image, segment)
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('val_iou', accuracy, on_step=True, on_epoch=True, prog_bar=True)
        return loss
    

model = SegFormerModel()
datamodule = GetData(batch_size=config.batch_size)

checkpoint_callback = ModelCheckpoint(monitor='val_loss', dirpath='checkpoints', filename='file', save_last = True)

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger

tb_logger = TensorBoardLogger("logs/", name = "SegFormer_v2_epoch")


trainer = Trainer(max_epochs=config.max_epochs,
                  accelerator="cuda" if torch.cuda.is_available() else "cpu",
                  callbacks=[checkpoint_callback],
                  num_sanity_val_steps=0,
                  logger = tb_logger,
                  )

trainer.fit(model, datamodule=datamodule)
# Loading the best model from checkpoint
best_model = SegFormerModel.load_from_checkpoint(checkpoint_callback.best_model_path)

# Define the file path where you want to save the model weights
weights_path = "hopes.pth"

# Save the model weights
torch.save(best_model.state_dict(), weights_path)
