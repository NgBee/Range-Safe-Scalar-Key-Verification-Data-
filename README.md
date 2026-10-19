# Range-Safe Scalar-Key Verification — Data and Supplements

Public data and supplementary materials for the paper
"Scalar-Key Verification with Selection-Domain Enforcement for Private Language Model Inference"
(Yujun Gan). The manuscript itself is not in this repository.

## Contents

- `交互式安全推理_研究资料/投稿整理_20261004/upload/` — original submission files (Oct 2026).
- `交互式安全推理_研究资料/160M补强_20261017/revision/supplement/` — revised supplements:
  - `SupplementaryServer160M.zip` — server data and offline analysis entry points for the earlier eight-input server batch.
  - `SupplementaryConfirmAndNetwork.zip` — the confirmatory server study (16 held-out inputs × 3 sessions × 3 models), the three-party network-condition matrix (14M/31M × 2 inputs × 3 tiers), the full server re-run of component, guard, cache-selection, numerical-diagnostic, boundary and fault experiments (including the 160M cache mux and 160M aligned-history diagnostics), the recheck records, and the fifteen-value membership-check archive. Per-request records, compiled graphs, configurations and source snapshots are included.
  - `SupplementaryConfirmAndNetwork.MANIFEST.json` — SHA-256 per file for the above package.
  - `original_server_scripts/` — the 160M-capable launcher snapshots.
- `交互式安全推理_研究资料/160M补强_20261017/revision/SHA256SUMS` — SHA-256 of the revised delivery files.

## Notes

- Zip archives are stored with Git LFS.
- The earlier `final_v3` code and numerical packages are delivered with the original submission materials.
- Verification: download the packages, check each file against its manifest SHA-256, and follow the supplement READMEs for the offline analysis entry points.
