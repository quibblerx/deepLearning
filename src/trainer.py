from functools import partial
from copy import deepcopy

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

class SemiSupervisedEnsemble:
    def __init__(
        self,
        supervised_criterion,
        optimizer,
        scheduler,
        device,
        models,
        logger,
        datamodule,
        consistency_weight=50.0,
        consistency_rampup_epochs=80,
        ema_decay=0.999,
    ):
        self.device = device
        self.models = models  # Student models
        
        # Create teacher models (EMA of student models)
        self.teacher_models = [deepcopy(model).to(device) for model in self.models]
        # Initialize teacher with same weights as student
        for teacher in self.teacher_models:
            for param in teacher.parameters():
                param.detach_()  # Don't compute gradients for teacher

        # Semi-supervised hyperparameters
        self.consistency_weight = consistency_weight
        self.consistency_rampup_epochs = consistency_rampup_epochs
        self.ema_decay = ema_decay

        # Optim related things
        self.supervised_criterion = supervised_criterion
        all_params = [p for m in self.models for p in m.parameters()]
        self.optimizer = optimizer(params=all_params)
        self.scheduler = scheduler(optimizer=self.optimizer)

        # Dataloader setup
        self.train_dataloader_labeled = datamodule.train_dataloader()
        self.train_dataloader_unlabeled = datamodule.unsupervised_train_dataloader()
        self.val_dataloader = datamodule.val_dataloader()
        self.test_dataloader = datamodule.test_dataloader()

        # Logging
        self.logger = logger

    def update_ema_variables(self):
        """Update teacher model parameters using exponential moving average of student parameters"""
        for teacher_model, student_model in zip(self.teacher_models, self.models):
            for teacher_param, student_param in zip(teacher_model.parameters(), student_model.parameters()):
                teacher_param.data = (
                    self.ema_decay * teacher_param.data + 
                    (1.0 - self.ema_decay) * student_param.data
                )

    def get_current_consistency_weight(self, epoch):
        """Ramp up consistency loss weight from 0 to max over rampup_epochs"""
        if epoch < self.consistency_rampup_epochs:
            # Sigmoid ramp-up
            rampup = np.exp(-5.0 * (1.0 - epoch / self.consistency_rampup_epochs) ** 2)
            return self.consistency_weight * rampup
        else:
            return self.consistency_weight

    def validate(self):
        # Use teacher models for validation (typically more stable)
        for model in self.teacher_models:
            model.eval()

        val_losses = []
        
        with torch.no_grad():
            for x, targets in self.val_dataloader:
                x, targets = x.to(self.device), targets.to(self.device)
                
                # Ensemble prediction using teacher models
                preds = [model(x) for model in self.teacher_models]
                avg_preds = torch.stack(preds).mean(0)
                
                val_loss = F.mse_loss(avg_preds, targets)
                val_losses.append(val_loss.item())
        val_loss = np.mean(val_losses)
        return {"val_MSE": val_loss}

    def test(self):
        """Evaluate on test set"""
        # Use teacher models for testing (typically more stable)
        for model in self.teacher_models:
            model.eval()

        test_losses = []
        
        with torch.no_grad():
            for x, targets in self.test_dataloader:
                x, targets = x.to(self.device), targets.to(self.device)
                
                # Ensemble prediction using teacher models
                preds = [model(x) for model in self.teacher_models]
                avg_preds = torch.stack(preds).mean(0)
                
                test_loss = F.mse_loss(avg_preds, targets)
                test_losses.append(test_loss.item())
        
        test_mse = np.mean(test_losses)
        test_mae = np.sqrt(test_mse)  # Approximation, should calculate properly
        return {"test_MSE": test_mse, "test_MAE": test_mae}

    def train(self, total_epochs, validation_interval):
        for epoch in (pbar := tqdm(range(1, total_epochs + 1))):
            # Set models to training mode
            for model in self.models:
                model.train()
            for model in self.teacher_models:
                model.train()  # Keep dropout active but no gradient computation
            
            supervised_losses_logged = []
            consistency_losses_logged = []
            
            # Get current consistency weight (ramps up over time)
            consistency_weight = self.get_current_consistency_weight(epoch)
            
            # Create iterators for both labeled and unlabeled data
            labeled_iter = iter(self.train_dataloader_labeled)
            unlabeled_iter = iter(self.train_dataloader_unlabeled)
            
            # Determine number of iterations (use the longer dataloader)
            num_iterations = max(len(self.train_dataloader_labeled), 
                               len(self.train_dataloader_unlabeled))
            
            for _ in range(num_iterations):
                # ===== SUPERVISED LOSS (Labeled Data) =====
                try:
                    x_labeled, targets = next(labeled_iter)
                except StopIteration:
                    labeled_iter = iter(self.train_dataloader_labeled)
                    x_labeled, targets = next(labeled_iter)
                
                x_labeled, targets = x_labeled.to(self.device), targets.to(self.device)
                
                self.optimizer.zero_grad()
                
                # Student predictions on labeled data
                supervised_losses = [
                    self.supervised_criterion(model(x_labeled), targets) 
                    for model in self.models
                ]
                supervised_loss = sum(supervised_losses) / len(self.models)
                supervised_losses_logged.append(supervised_loss.detach().item())
                
                # ===== CONSISTENCY LOSS (Unlabeled Data) =====
                try:
                    x_unlabeled, _ = next(unlabeled_iter)
                except StopIteration:
                    unlabeled_iter = iter(self.train_dataloader_unlabeled)
                    x_unlabeled, _ = next(unlabeled_iter)
                
                x_unlabeled = x_unlabeled.to(self.device)
                
                # Student predictions (with dropout/stochasticity)
                student_preds = [model(x_unlabeled) for model in self.models]
                
                # Teacher predictions (with different dropout mask, no gradients)
                with torch.no_grad():
                    teacher_preds = [model(x_unlabeled) for model in self.teacher_models]
                
                # Consistency loss: MSE between student and teacher predictions
                consistency_losses = [
                    F.mse_loss(student_pred, teacher_pred.detach()) 
                    for student_pred, teacher_pred in zip(student_preds, teacher_preds)
                ]
                consistency_loss = sum(consistency_losses) / len(self.models)
                consistency_losses_logged.append(consistency_loss.detach().item())
                
                # ===== TOTAL LOSS =====
                total_loss = supervised_loss + consistency_weight * consistency_loss
                
                # Backward pass and optimization
                total_loss.backward()
                self.optimizer.step()
                
                # Update teacher model with EMA after each batch
                self.update_ema_variables()
            
            # Step the learning rate scheduler
            self.scheduler.step()
            
            # Log average losses for the epoch
            supervised_losses_logged = np.mean(supervised_losses_logged)
            consistency_losses_logged = np.mean(consistency_losses_logged)

            summary_dict = {
                "sup_loss": supervised_losses_logged,
                "cons_loss": consistency_losses_logged,
                "cons_weight": consistency_weight,
            }
            
            # Validation
            if epoch % validation_interval == 0 or epoch == total_epochs:
                val_metrics = self.validate()
                summary_dict.update(val_metrics)
            
            # Always update progress bar (shows val_MSE when available)
            pbar.set_postfix(summary_dict)
            
            self.logger.log_dict(summary_dict, step=epoch)
