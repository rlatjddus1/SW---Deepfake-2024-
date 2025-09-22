import gc
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from datautils import MyTrainDataset
import os
from models import SSDNet2D
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import torch
from torch.utils.tensorboard import SummaryWriter
from pytorchtools import EarlyStopping
import numpy as np

def ddp_setup():
    init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    
class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_data: DataLoader,
        valid_data: DataLoader,
        snapshot_path: str,
        num_classes: int 
    ) -> None:
        self.gpu_id = int(os.environ["LOCAL_RANK"])
        self.model = model.to(self.gpu_id)
        self.train_data = train_data
        self.valid_data = valid_data
        self.epochs_run = 1
        self.learning_rate = 9e-6
        self.best_loss = np.Inf
        self.snapshot_path = snapshot_path
        self.num_classes = num_classes  

        self.optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, T_0=10, T_mult=2)
        if os.path.exists(snapshot_path):
            print("Loading snapshot")
            self._load_snapshot(snapshot_path)
        self.model = DDP(self.model, device_ids=[self.gpu_id],output_device=self.gpu_id,find_unused_parameters=True)

        if self.gpu_id == 0:
            self.writer = SummaryWriter('./runs1-weight_final10')

    def _load_snapshot(self, snapshot_path):
        loc = f"cuda:{self.gpu_id}"
        snapshot = torch.load(snapshot_path, map_location=loc)
        state_dict = snapshot["MODEL_STATE"]
        
        model_state_dict = self.model.state_dict()

        filtered_state_dict = {k: v for k, v in state_dict.items() if 'fc3' not in k}
        model_state_dict.update(filtered_state_dict)
        self.model.load_state_dict(model_state_dict, strict=False)
        self.optimizer.load_state_dict(snapshot["OPTIMIZER_STATE"])
        self.epochs_run = snapshot["EPOCHS_RUN"] + 1
        self.scheduler.load_state_dict(snapshot["SCHEDULER_STATE"])
        self.learning_rate = self.scheduler.get_last_lr()
        self.best_loss = snapshot["BEST_LOSS"]
        print(f"Resuming training from snapshot at Epoch {self.epochs_run}")
    

    def _run_batch(self, source, targets):
        self.optimizer.zero_grad()
        output = self.model(source)
        
        assert output.shape[1] == self.num_classes, \
            f"Model output shape {output.shape} does not match num_classes {self.num_classes}"
        
        if targets.ndim == 1 or targets.shape[1] == 1:
            targets_one_hot = F.one_hot(targets.long(), num_classes=self.num_classes).float()
        else:
            targets_one_hot = targets.float()
        
        targets_one_hot = targets_one_hot.squeeze(1) 
        targets_one_hot = targets_one_hot.to(self.gpu_id) 

        loss_fn = nn.BCELoss()
        loss = loss_fn(output, targets_one_hot)
        loss.backward()
        self.optimizer.step()

        
        return loss.item() * len(targets), len(targets)  

    def _run_valid_batch(self, source, targets):
        output = self.model(source)
        
        assert output.shape[1] == self.num_classes, \
            f"Model output shape {output.shape} does not match num_classes {self.num_classes}"

        if targets.ndim == 1 or targets.shape[1] == 1:
            targets_one_hot = F.one_hot(targets.long(), num_classes=self.num_classes).float()
        else:
            targets_one_hot = targets.float()
        
        targets_one_hot = targets_one_hot.squeeze(1) 
        targets_one_hot = targets_one_hot.to(self.gpu_id) 

        loss_fn = nn.BCELoss()
        loss = loss_fn(output, targets_one_hot)        
        return loss.item() * len(targets), len(targets)  



    def _train_valid_process(self):
        train_loss = 0.0
        train_len = 0
        valid_loss = 0.0
        valid_len = 0

        self.model.train()
        for source, targets in self.train_data:
            source = source.to(self.gpu_id)
            targets = targets.to(self.gpu_id)
            batch_loss, batch_len = self._run_batch(source, targets)
            train_loss += batch_loss
            train_len += batch_len

        dist.barrier()
        self.model.eval()
        for source, targets in self.valid_data:
            source = source.to(self.gpu_id)
            targets = targets.to(self.gpu_id)
            batch_loss, batch_len = self._run_valid_batch(source, targets)
            valid_loss += batch_loss
            valid_len += batch_len

        dist.barrier()

        return (
            train_loss / train_len, 
            valid_loss / valid_len, 

        )



    def _run_epoch(self, epoch):
        if self.gpu_id == 0:
            train_loss, valid_loss= self._train_valid_process()
            self.writer.add_scalar("Loss/train", train_loss, epoch)
            self.writer.add_scalar("Loss/valid", valid_loss, epoch)
            print(f"[GPU{self.gpu_id}] Epoch {epoch} | LR {self.learning_rate} | "
                f"Train Loss {train_loss:.6e} | Valid Loss {valid_loss:.6e}")
        else:
            train_loss, valid_loss, _, _ = self._train_valid_process()
            print(f"[GPU{self.gpu_id}] Epoch {epoch} | LR {self.learning_rate} | "
                f"Train Loss {train_loss:.6e} | Valid Loss {valid_loss:.6e}")

        return valid_loss

    
    def _save_snapshot(self, epoch, is_best, best_loss, learning_rate):
        snapshot = {
            "MODEL_STATE": self.model.module.state_dict(),
            'OPTIMIZER_STATE': self.optimizer.state_dict(),
            "EPOCHS_RUN": epoch,
            'SCHEDULER_STATE': self.scheduler.state_dict(),
            "BEST_LOSS": best_loss,
        }
        if is_best:
            # 가장 좋은 스냅샷
            torch.save(snapshot, "./best_weight_final10.pt")
        else:
            torch.save(snapshot, self.snapshot_path)
            print(f"Epoch {epoch} | Training snapshot saved at {self.snapshot_path}")

    def train(self, max_epochs: int):
        is_early_stop = torch.zeros(1, device=self.gpu_id)
        if self.gpu_id == 0:
            # patience: 몇번까지 조기중단 할지
            early_stopping = EarlyStopping(patience=25, verbose=True, val_loss_best=self.best_loss)
        for epoch in range(self.epochs_run, max_epochs+1):
            valid_loss = self._run_epoch(epoch)
            dist.barrier()
            if self.gpu_id == 0:
                early_stopping(valid_loss)
                if self.best_loss > early_stopping.get_loss():
                    self.best_loss = early_stopping.get_loss()
                self._save_snapshot(epoch,False,self.best_loss,self.learning_rate)
                if early_stopping.early_stop:
                    is_early_stop[0] = 1
                    print("Early stopping")
                elif not early_stopping.is_small:
                    self._save_snapshot(epoch,True,self.best_loss,self.learning_rate)
            dist.all_reduce(is_early_stop, op=dist.ReduceOp.SUM)
            if is_early_stop != 0:
                break
        if self.gpu_id == 0:
            self.writer.close()
            
def init_weights(m):
    if isinstance(m, nn.Conv3d):
        nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.ConvTranspose3d):
        nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm3d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight)
        nn.init.constant_(m.bias, 0)

def load_train_objs(num_classes: int = 2): 
    model = SSDNet2D(num_classes=num_classes) 
    return model


def prepare_dataloader(dataset: Dataset, batch_size: int):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=True,
        sampler=DistributedSampler(dataset, shuffle=True)
    )


def main(total_epochs: int, batch_size: int, num_classes: int = 2, snapshot_path: str = "./weight_final10.pt"):
    ddp_setup()
    train_dataset = MyTrainDataset('./model_train/mixtrain80_file2.csv','./CQT_mixed_train')
    valid_dataset = MyTrainDataset('./model_train/mixtrain20_file2.csv','./CQT_mixed_train')
    train_data = prepare_dataloader(train_dataset, batch_size)
    valid_data = prepare_dataloader(valid_dataset, batch_size)
    model = load_train_objs(num_classes)
    dist.barrier()
    trainer = Trainer(model, train_data, valid_data, snapshot_path, num_classes=num_classes)  # Pass num_classes
    trainer.train(total_epochs)
    destroy_process_group()

if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser(description='simple distributed training job')
    parser.add_argument('total_epochs', type=int, help='Total epochs to train the model')
    parser.add_argument('--batch_size', default=16, type=int, help='Input batch size on each device (default: 16)')
    parser.add_argument('--num_classes', default=2, type=int, help='Number of output classes (default: 2)')  # add num_classes argument
    
    args = parser.parse_args()
    
    main(args.total_epochs, args.batch_size,args.num_classes)
