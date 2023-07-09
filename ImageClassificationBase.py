import torch
import torch.nn as nn
import torch.nn.functional as F

class ImageClassificationBase(nn.Module):
  def training_step(self, batch):
    images, labels = batch
    out = self(images)
    loss = F.cross_entropy(out, labels)
    return loss

  def validation_step(self, batch):
    images, labels = batch
    out = self(images)
    loss = F.cross_entropy(out, labels) # calculate loss
    acc = accuracy(out, labels) # calculate accuracy
    return {'val_loss': loss.detach(), 'val_acc': acc}

  def validation_epoch_end(self, outputs):
    batch_loss_vals = [x['val_loss'] for x in outputs]
    batch_losses = torch.stack(batch_loss_vals).mean()
    batch_acc_vals = [x['val_acc'] for x in outputs]
    batch_accs = torch.stack(batch_acc_vals).mean()
    return {'val_loss': batch_losses.item(), 'val_acc': batch_accs.item()}

  def epoch_end(self, epoch, result):
    print("Epoch {}: train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
            epoch+1, result['train_loss'], result['val_loss'], result['val_acc']))
    

def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))