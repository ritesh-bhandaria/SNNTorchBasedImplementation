from snntorch import functional as SF
from snntorch import utils
import torch
from SegFormer import SegFormer
import config
import torch.utils.tensorboard as tb
import numpy as np 

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = SegFormer(
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

optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
num_epochs = config.max_epochs
loss_hist = []
val_acc_hist = []
counter = 0
loss_fn = SF.ce_rate_loss()
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
writer = tb.SummaryWriter()

best_val_loss = np.inf

scheduler.step()

def batch_accuracy(train_loader, net, num_steps):
  with torch.no_grad():
    total = 0
    acc = 0
    net.eval()

    train_loader = iter(train_loader)
    for data, targets in train_loader:
      data = data.to(device)
      targets = targets.to(device)
      spk_rec, _ = forward_pass(net, num_steps, data)

      acc += SF.accuracy_rate(spk_rec, targets) * spk_rec.size(1)
      total += spk_rec.size(1)

  return acc/total

def forward_pass(net, num_steps, data):
  spk_rec = []
  utils.reset(net)  # resets hidden states for all LIF neurons in net

  for step in range(num_steps):
      spk_out = model(data)
      spk_rec.append(spk_out)

  return torch.stack(spk_rec)


for epoch in range(num_epochs):

    for data, target in iter(train_loader):
        data = data.to(device)
        target = target.to(device)

        model.train()
        spk_rec = forward_pass(model, config.num_steps, data)

        loss_val  = loss_fn(spk_rec, target)

        optimizer.zero_grad()
        loss_val.backard()
        optimizer.step()

        loss_hist.append(loss_val.item())

        # validation set
        with torch.no_grad():
            model.eval()

            # Test set forward pass
            val_acc = batch_accuracy(val_loader, model, config.num_steps)
            print(f"Iteration {counter}, validation Acc: {val_acc * 100:.2f}%\n")
            val_acc_hist.append(val_acc.item())
            writer.add_scalar('Acc/val', val_acc.item(), counter)
        
        writer.add_scalar('Loss/train', loss_val.item(), counter)
        writer.add_scalar('Learning_rate', scheduler.get_last_lr()[0], counter)
        counter+=1
    
torch.save(model.state_dict(), 'SNN_segformer_weights_cityscapes.pth')
writer.close()

#CHECKPOINTS NOT IMPLEMENTED    

     

# class SegFormerDataModule(pl.LightningDataModule):
    
#     def __init__(self, batch_size):
#         super().__init__()
#         self.batch_size = batch_size

#     def setup(self, stage=None):
#         self.train_dataset = 
#         self.val_dataset = 
#         # print(self.train_dataset.shape)

#     def train_dataloader(self):
#         return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

#     def val_dataloader(self):
#         return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)
    


# class SegFormerModel(pl.LightningModule):
#     def __init__(self):
#         super().__init__()
#         self.model = SegFormer(
#             in_channels=config.in_channels,
#             widths=config.widths,
#             depths=config.depths,
#             all_num_heads=config.all_num_heads,
#             patch_sizes=config.patch_sizes,
#             overlap_sizes=config.overlap_sizes,
#             reduction_ratios=config.reduction_ratios,
#             mlp_expansions=config.mlp_expansions,
#             decoder_channels=config.decoder_channels,
#             scale_factors=config.scale_factors,
#             num_classes=config.num_classes,
#         )
        
    
#     def forward(self,x):
#         return self.model(x)
    
#     def process(self, image, segment): #image and segment needs to be checked 
#         out=self(image)
#         segment = encode_segmap(segment)
#         loss= SF.ce_rate_loss(out, segment)
#         acc = SF.accuracy_rate(out, segment)
#         return loss, acc
    
#     def configure_optimizers(self):
#         optimizer = torch.optim.AdamW(self.parameters(), lr=config.learning_rate)
#         scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=config.lr_factor, patience=config.lr_patience, verbose=True)
#         return {'optimizer': optimizer, 'lr_scheduler': scheduler, 'monitor' : 'val_loss'}
    
#     def training_step(self, batch, batch_idx):
#         image, segment = batch
#         loss, acc = self.process(image, segment)
#         self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
#         self.log('batch accuracy', acc, on_step=False, on_epoch=True, prog_bar=True)
        
#         return loss
    
#     def validation_step(self, batch, batch_idx):
#         image, segment = batch
#         loss, acc = self.process(image, segment)
#         self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
#         self.log('batch accuracy', acc, on_step=True, on_epoch=True, prog_bar=True)
#         return loss
    


# model = SegFormerModel()
# datamodule = SegFormerDataModule(batch_size=config.batch_size)

# checkpoint_callback = ModelCheckpoint(monitor='val_loss', dirpath='checkpoints', filename='file', save_last = True)
# early_stop_callback = EarlyStopping(monitor='val_loss', patience=config.early_stopping_patience, verbose=True, mode='min')

# tb_logger = TensorBoardLogger("logs/", name = "Spiking_Seg_Former")

# trainer = Trainer(max_epochs=config.max_epochs,
#                   accelerator="cuda" if torch.cuda.is_available() else "cpu",
#                   callbacks=[checkpoint_callback, early_stop_callback],
#                   num_sanity_val_steps=0,
#                   logger = tb_logger,
#                   )

# trainer.fit(model, datamodule=datamodule)
# # Loading the best model from checkpoint
# best_model = SegFormerModel.load_from_checkpoint(checkpoint_callback.best_model_path)
# # Define the file path where you want to save the model weights
# weights_path = config.path
# torch.save(best_model.state_dict(), weights_path)
