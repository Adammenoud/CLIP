import os
import re
import shutil
import wandb


def copy_best_checkpoint_from_run(
    run,
    run_dir,
    monitor="val_loss",
    save_every=2,
    output_name="best_model.pt",
    format_type="custom"
):
    checkpoints_dir = os.path.join(run_dir, "checkpoints")
    if not os.path.isdir(checkpoints_dir):
        print(f"⚠️  No checkpoints dir in {run_dir}")
        return

    history = run.history(keys=[monitor, "epoch"])
    history = history.dropna(subset=[monitor, "epoch"])

    if history.empty:
        print(f"⚠️  No history for run {run.name}")
        return

    best_epoch = int(history.loc[history[monitor].idxmin()]["epoch"])
    if format_type=="custom":
        ckpt_epoch = best_epoch//save_every #divide (corresponds to file labels)
        if   ckpt_epoch==0:
            ckpt_epoch=1 #0 was not saved
    else:
        ckpt_epoch=best_epoch

    #format path
    if format_type == "custom":
        ckpt_path = os.path.join(checkpoints_dir, f"checkpoint_{ckpt_epoch}", "model.pt")
    elif format_type == "lightning":
        ckpt_file = None
        for fname in os.listdir(checkpoints_dir):
            if f"_checkpoint_epoch={ckpt_epoch}" in fname:
                ckpt_file = fname
                break
        if ckpt_file is None:
            print(f"❌ No checkpoint file for epoch {ckpt_epoch} in {checkpoints_dir}")
            return
        ckpt_path = os.path.join(checkpoints_dir, ckpt_file)
    else:
        raise ValueError("format_type must be 'folder' or 'file'")

    dst = os.path.join(run_dir, f"best_model.pt")
    shutil.copy2(ckpt_path, dst)

    print(
        f"✅ {run.name}: best_epoch={best_epoch} → "
        f"checkpoint_{ckpt_epoch}/model.pt"
    )


def process_sweep(
    sweep_id,        # "entity/project/sweep_id"
    sweep_root,      # local folder corresponding to that sweep
    monitor="val_loss",
    save_every=2,
    format_type = "custom"
):
    api = wandb.Api()
    sweep = api.sweep(sweep_id)

    # Run name
    wandb_runs = {run.name: run for run in sweep.runs}


    for run_dir_name in os.listdir(sweep_root):
        run_dir = os.path.join(sweep_root, run_dir_name)
        if not os.path.isdir(run_dir):
            continue

        # Try to match local folder to W&B run
        run = wandb_runs.get(run_dir_name)

        if run is None:
            print(f"⚠️  No W&B run found for folder {run_dir_name}")
            continue

        copy_best_checkpoint_from_run(
            run=run,
            run_dir=run_dir,
            monitor=monitor,
            save_every=save_every,
            format_type=format_type
        )


if __name__ == "__main__":
    '''
    Select the best checkpoint for each run in a sweep, and copy it as "best_model.pt" in the corresponding folder.
    '''
    process_sweep( #contrastive images
    "adammenoud/sweep/1a25twyl",        # "entity/project/sweep_id"
    "Model_saves/sweep_contrastive_images",      # local folder corresponding to that sweep
    monitor="Cross-entropy validation",
    )
    process_sweep( #classifier emb
    "adammenoud/sweep/17gfx76m",        
    "Model_saves/sweep_classifier_emb",      
    monitor="MSE on validation set",
    format_type = "lightning"
    )
    process_sweep( #contrastive specie
    "adammenoud/sweep/aj8aunmz",       
    "Model_saves/sweep_contrastive_species",      
    monitor="Cross-entropy validation",
    )
    process_sweep( #classifier specie name
    "adammenoud/sweep/1wv36j0v",       
    "Model_saves/sweep_classifier",     
    monitor="cross_entropy validation",
    format_type = "lightning"
    )