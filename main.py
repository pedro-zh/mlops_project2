import argparse
import lightning as L
from pytorch_lightning.loggers import WandbLogger
from data_module import GLUEDataModule
from model import GLUETransformer
import wandb

def train(args):
    # Log into Wandb
    wandb.login()
    
    # Set up WandbLogger
    logger = WandbLogger(project="mlops_project_2", name=f"experiment_lr{args.lr}_wd{args.weight_decay}_ws{args.warmup_steps}")
    
    # Define hyperparameters
    hyperparams = {
        "learning_rate": args.lr,
        "weight_decay": args.weight_decay,
        "warmup_steps": args.warmup_steps,
    }
    
    # Log hyperparameters
    logger.log_hyperparams(hyperparams)
    
    # Set seed for reproducibility
    L.seed_everything(42)
    
    # Initialize Data Module
    dm = GLUEDataModule(
        model_name_or_path="distilbert-base-uncased",
        task_name="mrpc",
    )
    
    dm.setup("fit")
    
    # Initialize Model
    model = GLUETransformer(
        model_name_or_path="distilbert-base-uncased",
        num_labels=dm.num_labels,
        eval_splits=dm.eval_splits,
        task_name=dm.task_name,
        learning_rate=hyperparams["learning_rate"],
        weight_decay=hyperparams["weight_decay"],
        warmup_steps=hyperparams["warmup_steps"],
    )
    
    # Initialize Trainer
    trainer = L.Trainer(
        max_epochs=3,
        accelerator="auto",
        devices=1,
        logger=logger,
        default_root_dir=args.checkpoint_dir
    )
    
    # Train Model
    trainer.fit(model, datamodule=dm)
    
    # Finish Wandb session
    wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", type=str, default="models", help="Directory to save the checkpoints")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--warmup_steps", type=int, default=0, help="Warmup steps")
    args = parser.parse_args()
    
    train(args)
