## 2026-10-20 — A
- Did: Set up WSL Ubuntu 22.04, conda env (hls4ml-tutorial), started Vitis 2024.1 install (Zynq-7000 only), created repo
- Broke: First paste into WSL added ^[[200~ junk — fixed by opening Ubuntu directly
- Broke: Vitis install crashed at 96% — C: filled up because the WSL disk file grows but never shrinks (hit 120 GB). Fixed by freeing space, reinstalling from cached downloads, then fstrim + diskpart compact (120 → 79 GB).
- Decided / rejected: Ubuntu 22.04 over 26.04 (Vitis needs libtinfo5)
