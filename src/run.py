
from itertools import chain
import hydra
import torch
from omegaconf import OmegaConf

from utils import seed_everything


@hydra.main(
    config_path="../configs/",
    config_name="run.yaml",
    version_base=None,
)
def main(cfg):
    # print out the full config
    print(OmegaConf.to_yaml(cfg))

    if cfg.device in ["unset", "auto"]:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(cfg.device)
    
    print(f"🖥️  Device: {device}")
    if device.type == "cuda":
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
    else:
        print(f"   ⚠️  CUDA available: {torch.cuda.is_available()}")

    seed_everything(cfg.seed, cfg.force_deterministic)

    print("🚀 Initializing logger...")
    logger = hydra.utils.instantiate(cfg.logger)
    hparams = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    logger.init_run(hparams)

    print("📊 Loading QM9 dataset (this may take a few minutes)...")
    dm = hydra.utils.instantiate(cfg.dataset.init)

    print("🧠 Creating model...")
    model = hydra.utils.instantiate(cfg.model.init).to(device)

    if cfg.compile_model:
        print("⚡ Compiling model...")
        model = torch.compile(model)
    models = [model]
    
    print("👨‍🏫 Creating Mean Teacher trainer (copying model for teacher)...")
    trainer = hydra.utils.instantiate(cfg.trainer.init, models=models, logger=logger, datamodule=dm, device=device)

    print("🏋️ Starting training...\n")
    trainer.train(**cfg.trainer.train)
    
    print("\n" + "="*60)
    print("🎯 TRAINING COMPLETE - FINAL EVALUATION")
    print("="*60)
    
    # Final validation score
    print("\n📊 Final Validation Results:")
    val_results = trainer.validate()
    for key, value in val_results.items():
        print(f"  {key}: {value:.6f}")
    
    # Test set evaluation
    print("\n🧪 Test Set Results:")
    test_results = trainer.test()
    for key, value in test_results.items():
        print(f"  {key}: {value:.6f}")
    
    print("\n" + "="*60)
    print("✅ All done!")
    print("="*60 + "\n")



if __name__ == "__main__":
    main()
