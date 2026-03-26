import os
import math
import torch
import torch.nn as nn
import pandas as pd
from sklearn.metrics import accuracy_score
from scipy.stats import pearsonr
from loguru import logger

import matplotlib.pyplot as plt



class Trainer:
    def __init__(self, model, train_dat, val_dat, device, optimizer, model_dir, 
                 model_name="model", batch_size=128, patience=3, min_delta=0.001,
                 save_half=False, save_one=False):
        self.model = model.to(device)
        # We no longer call .to(device) here; data stays on disk/memmap
        self.train_dat = train_dat
        self.val_dat = val_dat
        self.device, self.optimizer, self.batch_size = device, optimizer, batch_size
        self.model_dir = model_dir
        self.model_name = model_name
    
        # Early Stopping parameters
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.early_stop = False
        
        self.reg_criterion = nn.MSELoss()
        self.clf_criterion = nn.BCEWithLogitsLoss()
        
        self.save_half = save_half
        self.save_one = save_one
        
        self.best_pearson = -1.0
        self.history = []
        os.makedirs(model_dir, exist_ok=True)

    def save_checkpoint(self, label):
        os.makedirs(self.model_dir, exist_ok=True) 
        model_path = os.path.join(self.model_dir, f"{self.model_name}_{label}.pt")
        torch.save(self.model.state_dict(), model_path)
        
        if self.history:
            csv_path = os.path.join(self.model_dir, "training_log.csv")
            pd.DataFrame(self.history).to_csv(csv_path, index=False)


    def _train_one_epoch(self, epoch):
        self.model.train()
        total_running_loss = 0.0  # Accumulate absolute sum
        num_samples = len(self.train_dat)
        num_batches = math.ceil(num_samples / self.batch_size)
        shuffled_indices = torch.randperm(num_samples).numpy()
        
        for i in range(num_batches):
            start = i * self.batch_size
            end = min(start + self.batch_size, num_samples)
            # Get actual batch size for this iteration
            current_batch_size = end - start 
            batch_indices = shuffled_indices[start:end]
            actual_indices = self.train_dat.indices[batch_indices]
            bx = torch.from_numpy(self.train_dat.X[actual_indices]).to(self.device)
            byr = torch.from_numpy(self.train_dat.Yr[actual_indices]).to(self.device)
            byc = torch.from_numpy(self.train_dat.Yc[actual_indices]).to(self.device)
            self.optimizer.zero_grad()
            preds = self.model(bx)
            loss_reg = self.reg_criterion(preds[:, 0], byr)
            loss_clf = self.clf_criterion(preds[:, 1], byc)
            loss = loss_reg + loss_clf
            loss.backward()
            self.optimizer.step()
            total_running_loss += loss.item() * current_batch_size
            # TODO:Do I need to save at 1/3?
            if self.save_half and epoch == 0 and i == num_batches // 2:
                self.save_checkpoint("half")
        
        if self.save_one and epoch == 0:
            self.save_checkpoint("one")   

        return total_running_loss / num_samples



    def _validate(self, epoch, train_loss): # Fixed signature
        self.model.eval()
        all_preds_reg, all_gts_reg = [], []
        all_preds_clf, all_gts_clf = [], []
        total_val_loss = 0
        
        num_samples = len(self.val_dat)
        num_batches = math.ceil(num_samples / self.batch_size)
        
        with torch.no_grad():
            for i in range(num_batches):
                start = i * self.batch_size
                end = min(start + self.batch_size, num_samples)
                batch_indices = self.val_dat.indices[start:end]
                
                bx = torch.from_numpy(self.val_dat.X[batch_indices]).to(self.device)
                byr = torch.from_numpy(self.val_dat.Yr[batch_indices]).to(self.device)
                byc = torch.from_numpy(self.val_dat.Yc[batch_indices]).to(self.device)
                
                out = self.model(bx)
                
                # Calculate batch loss (mean)
                loss_val = self.reg_criterion(out[:, 0], byr) + self.clf_criterion(out[:, 1], byc)
                # Weight by batch size for accurate epoch mean
                total_val_loss += loss_val.item() * (end - start)

                all_preds_reg.extend(out[:, 0].cpu().numpy())
                all_gts_reg.extend(byr.cpu().numpy())
                
                preds_class = (torch.sigmoid(out[:, 1]) > 0.5).float()
                all_preds_clf.extend(preds_class.cpu().numpy())
                all_gts_clf.extend(byc.cpu().numpy())

        # Compute Final Metrics
        val_loss = total_val_loss / num_samples # Accurate weighted average
        val_pearson = pearsonr(all_preds_reg, all_gts_reg)[0]
        val_acc = accuracy_score(all_gts_clf, all_preds_clf)

        # Create the metrics dictionary here
        epoch_metrics = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "val_pearson": val_pearson
        }

        # --- Log Reporting Table ---
        self._print_epoch_summary(epoch_metrics)

        # Early Stopping Logic
        if val_pearson > self.best_pearson:
            self.best_pearson = val_pearson
            self.save_checkpoint("best")
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        
        self.history.append(epoch_metrics)
        return val_pearson


    def _print_epoch_summary(self, m):
        """Logs a clean matrix-style table of results to the log file"""
        # We use a single string to ensure the table isn't broken up by other logs
        table = (
            f"\nEpoch {m['epoch']} Summary:\n"
            f"{'Set':<10} | {'Loss':<10} | {'Accuracy':<10} | {'Pearson R':<10}\n"
            f"{'-' * 52}\n"
            f"{'Train':<10} | {m['train_loss']:<10.4f} | {'-':<10} | {'-':<10}\n"
            f"{'Val':<10} | {m['val_loss']:<10.4f} | {m['val_acc']:<10.4f} | {m['val_pearson']:<10.4f}\n"
        )
        logger.info(table)
    
    
    def _plot_learning_curve(self, save_path="learning_curve.pdf"):
        if not self.history:
            logger.info("No history to plot.")
            return
        df = pd.DataFrame(self.history)
        plt.figure(figsize=(4,3))
        plt.plot(df['epoch'], df['train_loss'], label='Train Loss', marker='o')
        plt.plot(df['epoch'], df['val_loss'], label='Val Loss', marker='x')
        plt.title('Training vs. Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Save the plot
        plt.savefig(save_path)
        logger.info(f"Learning curve saved to {save_path}")
        # Display to screen
        plt.show()
        plt.close()



    def train(self, epochs=10):
        for epoch in range(epochs):
            train_loss = self._train_one_epoch(epoch)
            # Note: _validate now handles the dictionary creation and appending to history
            self._validate(epoch, train_loss)
            if self.early_stop: 
                break
        # Final CSV Save
        csv_path = os.path.join(self.model_dir, "training_log.csv")
        pd.DataFrame(self.history).to_csv(csv_path, index=False)
        # Generate Plot
        self._plot_learning_curve(os.path.join(self.model_dir, "learning_curve.pdf"))
        return self.best_pearson