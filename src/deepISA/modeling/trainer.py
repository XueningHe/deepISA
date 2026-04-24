import os
import math
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from loguru import logger

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator




class Trainer:
    def __init__(self, model, mode, train_dat, val_dat, test_dat, 
                 device, model_dir, trainer_config):
        self.model = model.to(device)
        self.mode = mode
        self.train_dat = train_dat
        self.val_dat = val_dat
        self.test_dat = test_dat
        self.device = device
        self.model_dir = model_dir
        self.optimizer = torch.optim.Adam(model.parameters(), lr=trainer_config.get("learning_rate", 1e-3))
        # unpack trainer_config with defaults
        self.epochs = trainer_config.get("epochs", 10)
        self.batch_size = trainer_config.get("batch_size", 128)
        self.patience = trainer_config.get("patience",3)
        self.min_delta = trainer_config.get("min_delta", 0.001)
        self.save_one_fourth = trainer_config.get("save_one_fourth", False)
        self.save_one = trainer_config.get("save_one", False)
        self.counter = 0
        # Loss functions
        self.reg_criterion = nn.MSELoss()
        self.clf_criterion = nn.BCEWithLogitsLoss()
        self.best_score = -np.inf
        self.metrics_log = []
        os.makedirs(model_dir, exist_ok=True)
        
        
    def _compute_loss(self, preds, byr, byc):
        """Unified loss calculation based on mode."""
        if self.mode == 'dual':
            return self.reg_criterion(preds[:, 0], byr) + self.clf_criterion(preds[:, 1], byc)
        elif self.mode == 'regression':
            val = preds[:, 0] if preds.ndim > 1 and preds.shape[1] > 1 else preds.squeeze()
            return self.reg_criterion(val, byr)
        else: 
            raise ValueError(f"Unsupported mode: {self.mode}")
        

    def _save_checkpoint(self, label):
        os.makedirs(self.model_dir, exist_ok=True) 
        model_path = os.path.join(self.model_dir, f"model_{label}.pt")
        torch.save(self.model.state_dict(), model_path)
        
    def _flush_metrics(self):
        if not self.metrics_log:
            return
        csv_path = os.path.join(self.model_dir, "metrics.csv")
        pd.DataFrame(self.metrics_log).to_csv(csv_path, index=False)
        logger.info(f"Metrics saved → {csv_path}")
            
    def _fetch_batch(self, data_container, batch_indices):
        """Vectorized read from memmap handles."""
        bx = torch.from_numpy(data_container.X[batch_indices]).to(self.device)
        byr = torch.from_numpy(data_container.Yr[batch_indices]).to(self.device)
        byc = torch.from_numpy(data_container.Yc[batch_indices]).to(self.device)
        return bx, byr, byc
    
    
    def _train_one_epoch(self, epoch):
        self.model.train()
        total_loss, num_samples = 0, len(self.train_dat)
        num_batches = math.ceil(num_samples / self.batch_size)
        indices = torch.randperm(num_samples).numpy()
        
        for batch_num, i in enumerate(range(0, num_samples, self.batch_size)):
            batch_idx = indices[i : i + self.batch_size]
            bx, byr, byc = self._fetch_batch(self.train_dat, batch_idx)
            self.optimizer.zero_grad()
            loss = self._compute_loss(self.model(bx), byr, byc)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item() * len(batch_idx)
            # Fixed checkpoint logic using batch_num
            if self.save_one_fourth and epoch == 0 and batch_num == num_batches // 4:
                self._save_checkpoint("_1_4")
                
        if self.save_one and epoch == 0:
            self._save_checkpoint("_1")
            
        return total_loss / num_samples



    def _calculate_metrics(self, total_loss, preds, yr, yc, n_samples):
        # Keep everything as tensors on the current device
        preds = torch.cat(preds)
        yr = torch.cat(yr)
        yc = torch.cat(yc)
        m = {"loss": total_loss / n_samples}

        # --- Regression (Pearson) ---
        if self.mode in ['regression', 'dual']:
            p_val = preds[:, 0] if self.mode == 'dual' else preds.view(-1)
            # Calculate Pearson R using Torch
            # Stack p_val and yr to get a 2xN matrix for corrcoef
            stacked = torch.stack([p_val, yr])
            corr_matrix = torch.corrcoef(stacked)
            m["pearson"] = corr_matrix[0, 1].item() 
        else:
            m["pearson"] = 0.0

        # --- Dual (Pearson, accuracy)---
        if self.mode=='dual':
            c_val = preds[:, 1] if self.mode == 'dual' else preds.view(-1)
            # Pure Torch accuracy logic
            binary_preds = (c_val > 0).float()
            acc = (binary_preds == yc).float().mean()
            m["accuracy"] = acc.item()
        else:
            m["accuracy"] = 0.0

        return m

    def _report_metrics(self, stage, metrics, epoch=None):
        # One concise log line
        parts = [f"[{stage}]"]
        if epoch is not None:
            parts.append(f"epoch={epoch}")
        parts += [f"{k}={v:.4f}" for k, v in metrics.items()]
        logger.info("  ".join(parts))

        # Append to unified log
        row = {"stage": stage, "epoch": epoch, **metrics}
        self.metrics_log.append(row)



    def _evaluate(self, data_container):
        """Performs a full pass over a dataset to calculate loss and metrics."""
        self.model.eval()
        total_loss = 0
        all_preds, all_yr, all_yc = [], [], []
        indices = np.arange(len(data_container))
        
        with torch.no_grad():
            for i in range(0, len(indices), self.batch_size):
                batch_idx = indices[i : i + self.batch_size]
                bx, byr, byc = self._fetch_batch(data_container, batch_idx)
                preds = self.model(bx)
                loss = self._compute_loss(preds, byr, byc)
                total_loss += loss.item() * len(batch_idx)
                all_preds.append(preds)
                all_yr.append(byr)
                all_yc.append(byc)

        return self._calculate_metrics(total_loss, all_preds, all_yr, all_yc, len(data_container))


    def _validate(self):
        return self._evaluate(self.val_dat)


    def _test(self):
        logger.info("Running final evaluation...")
        best_path = os.path.join(self.model_dir, "model_best.pt")
        self.model.load_state_dict(torch.load(best_path, weights_only=True))
        test_metrics = self._evaluate(self.test_dat)
        self._report_metrics("test", test_metrics)
        return test_metrics


    def _plot_learning_curve(self, save_path="learning_curve.pdf"):
        df = pd.DataFrame(self.metrics_log)
        train_df = df[df['stage'] == 'train']
        val_df   = df[df['stage'] == 'val']
        plt.plot(train_df['epoch'], train_df['loss'], label='Train Loss', marker='o')
        plt.plot(val_df['epoch'],   val_df['loss'],   label='Val Loss',   marker='x')
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.title('Training vs. Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        # Save the plot
        plt.savefig(save_path)
        logger.info(f"Learning curve saved to {save_path}")
        # Display to screen
        plt.show()
        plt.close()


    def train(self):
        """Main execution loop."""
        for epoch in range(self.epochs):
            train_loss = self._train_one_epoch(epoch)
            # Validation pass (No longer takes train_loss)
            val_m = self._validate()
            # TODO： report in same line
            self._report_metrics("val", val_m, epoch=epoch)
            self._report_metrics("train", {"loss": train_loss}, epoch=epoch)

            # Early Stopping Logic
            current_score = val_m["pearson"]
            if current_score > (self.best_score + self.min_delta):
                self.best_score = current_score
                self._save_checkpoint("best")
                self.counter = 0  # Reset the counter because we found a new best
            else:
                self.counter += 1  # Increment because there was no improvement
                if self.counter >= self.patience:
                    logger.info(f"Early stopping triggered after {self.patience} epochs without improvement.")
                    break

        # Final logic
        self._test()
        self._flush_metrics() 
        self._plot_learning_curve(os.path.join(self.model_dir, "learning_curve.pdf"))
        return None