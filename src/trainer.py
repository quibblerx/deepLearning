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
        grad_clip_norm: float = 0.0,
        use_target_normalization: bool = True,
        epoch_mode: str = "labeled",
        mt_augment_mode: str = "none",
        mt_augment_strength: float = 0.05,
    ):
        self.device = device
        self.models = models  # Student models
        
        # Create teacher models (EMA of student models)
        self.teacher_models = [deepcopy(model).to(device) for model in self.models]
        # Initialize teacher with same weights as student and freeze it.
        for teacher in self.teacher_models:
            teacher.eval()
            for param in teacher.parameters():
                param.requires_grad_(False)

        # Semi-supervised hyperparameters
        self.consistency_weight = consistency_weight
        self.consistency_rampup_epochs = consistency_rampup_epochs
        self.ema_decay = ema_decay
        self.grad_clip_norm = grad_clip_norm
        self.use_target_normalization = use_target_normalization
        self.epoch_mode = epoch_mode
        self.mt_augment_mode = mt_augment_mode
        self.mt_augment_strength = mt_augment_strength

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

        # Target normalization computed from the labeled training set.
        self.target_mean = 0.0
        self.target_std = 1.0
        if self.use_target_normalization:
            all_targets = []
            for _, targets in self.train_dataloader_labeled:
                all_targets.append(targets.detach().cpu().view(-1))
            if all_targets:
                all_targets = torch.cat(all_targets, dim=0)
                self.target_mean = float(all_targets.mean())
                self.target_std = float(all_targets.std())
                if self.target_std == 0:
                    self.target_std = 1.0

        # Logging
        self.logger = logger
        self._best_state = None
        self._best_epoch = None

    @torch.no_grad()
    def update_ema_variables(self):
        """Update teacher model parameters using exponential moving average of student parameters"""
        for teacher_model, student_model in zip(self.teacher_models, self.models):
            for teacher_param, student_param in zip(teacher_model.parameters(), student_model.parameters()):
                teacher_param.mul_(self.ema_decay).add_(student_param, alpha=1.0 - self.ema_decay)

            # Keep batch-norm state aligned with the student.
            for teacher_buffer, student_buffer in zip(teacher_model.buffers(), student_model.buffers()):
                teacher_buffer.copy_(student_buffer)

    def get_current_consistency_weight(self, epoch):
        """Ramp up consistency loss weight from 0 to max over rampup_epochs"""
        if epoch < self.consistency_rampup_epochs:
            # Sigmoid ramp-up
            rampup = np.exp(-5.0 * (1.0 - epoch / self.consistency_rampup_epochs) ** 2)
            return self.consistency_weight * rampup
        else:
            return self.consistency_weight

    def augment_unlabeled(self, data):
        """Create the student view for unlabeled data."""
        if self.mt_augment_mode == "none" or self.mt_augment_strength <= 0:
            return data

        data_aug = deepcopy(data)
        x = data_aug.x
        if not torch.is_floating_point(x):
            x = x.float()

        if self.mt_augment_mode == "noise":
            x = x + self.mt_augment_strength * torch.randn_like(x)
        elif self.mt_augment_mode == "mask":
            mask = torch.rand_like(x) < self.mt_augment_strength
            x = x.masked_fill(mask, 0.0)
        else:
            raise ValueError(f"Unknown mt_augment_mode: {self.mt_augment_mode}")

        data_aug.x = x
        return data_aug

    def _evaluate(self, dataloader, models):
        total_squared_error = 0.0
        total_absolute_error = 0.0
        total_examples = 0

        for model in models:
            model.eval()

        with torch.no_grad():
            for x, targets in dataloader:
                x, targets = x.to(self.device), targets.to(self.device)

                # Ensemble prediction using the provided model set
                preds = [model(x) for model in models]
                if self.use_target_normalization:
                    preds = [(p * self.target_std + self.target_mean) for p in preds]
                avg_preds = torch.stack(preds).mean(0)

                diff = avg_preds - targets
                total_squared_error += diff.pow(2).sum().item()
                total_absolute_error += diff.abs().sum().item()
                total_examples += diff.numel()

        mse = total_squared_error / total_examples
        mae = total_absolute_error / total_examples
        return mse, mae

    def validate(self):
        val_mse, val_mae = self._evaluate(self.val_dataloader, self.models)
        return {"val_MSE": val_mse, "val_MAE": val_mae}

    def test(self):
        """Evaluate on test set"""
        test_mse, test_mae = self._evaluate(self.test_dataloader, self.models)
        return {"test_MSE": test_mse, "test_MAE": test_mae}

    def train(self, total_epochs, validation_interval, early_stopping_patience=None):
        patience = 10 if early_stopping_patience is None else int(early_stopping_patience)
        best_val_loss = float('inf')
        patience_counter = 0
        stop_training = False

        for epoch in (pbar := tqdm(range(1, total_epochs + 1))):
            # Set models to training mode
            for model in self.models:
                model.train()

            supervised_losses_logged = []
            consistency_losses_logged = []

            # Get current consistency weight (ramps up over time)
            consistency_weight = self.get_current_consistency_weight(epoch)

            # Epoch schedule can either be labeled-driven (faster) or use the
            # longer labeled/unlabeled schedule for more unlabeled updates.
            if self.epoch_mode == "labeled":
                unlabeled_iter = iter(self.train_dataloader_unlabeled)
                epoch_batches = self.train_dataloader_labeled
            elif self.epoch_mode == "max":
                labeled_iter = iter(self.train_dataloader_labeled)
                unlabeled_iter = iter(self.train_dataloader_unlabeled)
                num_iterations = max(len(self.train_dataloader_labeled), len(self.train_dataloader_unlabeled))
                epoch_batches = range(num_iterations)
            else:
                raise ValueError(f"Unknown epoch_mode: {self.epoch_mode}")

            for batch in epoch_batches:
                if self.epoch_mode == "labeled":
                    x_labeled, targets = batch
                else:
                    try:
                        x_labeled, targets = next(labeled_iter)
                    except StopIteration:
                        labeled_iter = iter(self.train_dataloader_labeled)
                        x_labeled, targets = next(labeled_iter)

                x_labeled, targets = x_labeled.to(self.device), targets.to(self.device)

                self.optimizer.zero_grad()

                # Student predictions on labeled data
                if self.use_target_normalization:
                    targets_for_loss = (targets - self.target_mean) / self.target_std
                else:
                    targets_for_loss = targets

                supervised_losses = [
                    self.supervised_criterion(model(x_labeled), targets_for_loss)
                    for model in self.models
                ]
                supervised_loss = sum(supervised_losses) / len(self.models)
                supervised_losses_logged.append(supervised_loss.detach().item())

                # Log supervised error back in raw QM9 units for readability.
                with torch.no_grad():
                    preds_raw = [p * self.target_std + self.target_mean for p in [model(x_labeled) for model in self.models]] if self.use_target_normalization else [model(x_labeled) for model in self.models]
                    ensemble_raw = torch.stack(preds_raw).mean(0)
                    raw_sup_mse = F.mse_loss(ensemble_raw, targets).item()
                    # Replace the normalized training-loss scale with the raw-scale metric.
                    supervised_losses_logged[-1] = raw_sup_mse

                # ===== CONSISTENCY LOSS (Unlabeled Data) =====
                try:
                    x_unlabeled, _ = next(unlabeled_iter)
                except StopIteration:
                    unlabeled_iter = iter(self.train_dataloader_unlabeled)
                    x_unlabeled, _ = next(unlabeled_iter)

                x_unlabeled_student = self.augment_unlabeled(x_unlabeled)

                x_unlabeled = x_unlabeled.to(self.device)
                x_unlabeled_student = x_unlabeled_student.to(self.device)

                # Student predictions (with dropout/stochasticity)
                student_preds = [model(x_unlabeled_student) for model in self.models]

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

                if self.grad_clip_norm and self.grad_clip_norm > 0.0:
                    params = [p for m in self.models for p in m.parameters() if p.grad is not None]
                    torch.nn.utils.clip_grad_norm_(params, self.grad_clip_norm)

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

                cur_val = val_metrics["val_MSE"]
                if cur_val < best_val_loss:
                    best_val_loss = cur_val
                    patience_counter = 0
                    try:
                        self._best_state = [{k: v.cpu().clone() for k, v in m.state_dict().items()} for m in self.models]
                        self._best_epoch = epoch
                    except Exception:
                        self._best_state = None
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print(f"Early stopping at epoch {epoch}")
                        pbar.set_postfix(summary_dict)
                        self.logger.log_dict(summary_dict, step=epoch)
                        stop_training = True
                        break

            # Always update progress bar (shows val_MSE when available)
            pbar.set_postfix(summary_dict)
            
            self.logger.log_dict(summary_dict, step=epoch)

            if stop_training:
                break

        if self._best_state is not None:
            try:
                for model, state in zip(self.models, self._best_state):
                    model.load_state_dict(state)
                print(f"Restored best model from epoch {self._best_epoch} for testing/evaluation.")
            except Exception:
                print("Failed to restore best model state; using final weights.")

        return summary_dict
