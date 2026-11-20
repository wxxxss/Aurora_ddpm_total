# First revision archive

The 27 first-round experiment scripts previously under `paper_modif/R1-*` and `paper_modif/R2-*` were moved into this directory, retaining their original comment-group structure. An additional 62 files from the same round (previously unpushed figures, CSVs, TeX tables and SSUSI preparation script) have now also been archived here. The previous folder paths were removed. Imports and paths tied to the original directory depth were adjusted where identified; the experiment algorithms and model parameters were not changed, apart from path references needed for the moved files. Because the repository does not contain the private datasets and checkpoints, full end-to-end execution has not been re-verified.

Run archived scripts from the repository root. For example:

```bash
python paper_modif/First_revision/R2-1_R2-3/test_heldout_eval_utils.py
```

New tests and reviewer responses belong in `../Second_revision/`.
