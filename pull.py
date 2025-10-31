from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="icampara/spirkappdocker",   # your Space name
    repo_type="space",                   # type = space (important!)
    local_dir=r"C:\Users\Irfan\Documents\GitHub\smirk6166Retrained",
    local_dir_use_symlinks=False,        # real files, not links
    resume_download=True                 # can safely re-run
)