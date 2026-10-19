# Server Supplements

- `SupplementaryServer160M.zip` — server data and offline analysis entry points for the earlier eight-input server batch (requests, compiled graphs, setup records).
- `SupplementaryConfirmAndNetwork.zip` — the confirmatory server study (16 held-out inputs x 3 sessions x 3 models), the three-party network-condition matrix (14M/31M x 2 inputs x 3 shaped tiers with measured RTT), the full server re-run of the component, guard, cache-selection, numerical-diagnostic, boundary and fault experiments (including 160M cache mux and 160M aligned-history diagnostics), the recheck records, and the fifteen-value membership-check archive.
- `SupplementaryConfirmAndNetwork.MANIFEST.json` — per-file SHA-256 for the above package.
- `original_server_scripts/` — the 160M-capable launcher snapshots used on the server.

Verification: download the packages, check every file against its manifest SHA-256, then use the recorded entry points for offline re-analysis. The earlier `final_v3` code and numerical packages remain part of the original submission materials.
