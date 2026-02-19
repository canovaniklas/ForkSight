"""
W&B Sweep worker — runs inside a SLURM job on a compute node.

Resumes the wandb sweep run (created by sweep_wandb_train.py on the
login node) and executes the full training + evaluation pipeline.
Environment variables have already been set by sweep_wandb_run.sh.
"""

import argparse
import os

import wandb


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--entity", required=True)
    parser.add_argument("--project", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--sweep-id", required=True)
    args = parser.parse_args()

    # resume the sweep run that was created on the login node
    run = wandb.init(
        entity=args.entity,
        project=args.project,
        id=args.run_id,
        resume="must",
    )

    # import training module AFTER env vars are set (module-level reads)
    from Segmentation.SAM import sam_lora_train as T

    # monkey-patch init_wandb_run to reuse the resumed sweep run
    # (sweep config is already populated by wandb.init in sweep_wandb_train.py)
    def _sweep_init_wandb_run(*_args, **_kwargs):
        run_out_dir = T.get_init_run_out_dir(run)
        with open(str(run_out_dir / "wandb_run_id.txt"), "w") as f:
            f.write(run.id)
        return run

    T.init_wandb_run = _sweep_init_wandb_run

    # run training + evaluation
    T.seed_everything(T.SEED)

    device = T.torch.device("cuda" if T.torch.cuda.is_available() else "cpu")
    sam_lora = T.init_model(device)
    trainloader, validationloader, train_size, val_size = T.init_data_loaders()

    wandb_run = T.init_wandb_run(train_size, val_size, sum(
        p.numel() for _, p in T.get_trainable_params(sam_lora)))

    T.train(sam_lora, wandb_run, trainloader, validationloader, device)
    T.evaluate_checkpoints(wandb_run, device)

    wandb.finish()


if __name__ == "__main__":
    main()
