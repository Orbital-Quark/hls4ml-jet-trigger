# Setup manual: from a fresh Windows laptop to running the sweep

This gets you from nothing to running the project's code and seeing its graphs. Follow the parts in order. Total time is about half a day, most of it waiting for downloads.

Commands in **PowerShell** boxes go in Windows PowerShell. Everything else goes in the **Ubuntu** terminal.

---

## Part 0: Before you start

| Check | Why |
|---|---|
| **About 80 GB free on C:** | Vitis takes about 50 GB, and the chip builds from a sweep take 10–20 GB more |
| **An AMD account** (free, at amd.com) | You need it to download Vitis |
| **Laptop plugged in, sleep turned off** (Settings → System → Power) | The installer and the sweeps run for hours; sleep pauses them |

---

## Part 1: Linux on Windows (WSL), about 1 hour

### 1.1 Install Ubuntu 22.04
Open **PowerShell as Administrator**:
```powershell
wsl --install -d Ubuntu-22.04
```
Restart when asked. Ubuntu opens and asks you to pick a Linux username and password.

- Use **22.04**, not a newer Ubuntu. Vitis needs an older library (`libtinfo5`) that newer versions don't have.
- If you already have WSL, check the version with `lsb_release -a`. If it's not 22.04, run the command above to install 22.04 next to it.
- **Open Ubuntu from the Start menu**, not through another terminal. Otherwise pasting can add junk characters like `^[[200~` to the start of commands.

### 1.2 Install the basic tools
```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y build-essential git wget curl unzip \
  libtinfo5 libncurses5 locales libxrender1 libxtst6 libxi6 libgtk2.0-0
sudo locale-gen en_US.UTF-8
```
`build-essential` gives the C++ compiler that hls4ml needs. The `lib...` packages are needed by the Vitis installer.

### 1.3 Give Linux enough memory
In Windows, create the file `C:\Users\<your-windows-username>\.wslconfig` with:
```ini
[wsl2]
memory=12GB
processors=6
swap=8GB
```
Set **memory** to your total RAM minus 4 GB, and **processors** to your number of cores minus 2. Then, in PowerShell:
```powershell
wsl --shutdown
```
and open Ubuntu again.

### 1.4 Install conda (Miniforge)
```bash
cd ~
wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
bash Miniforge3-Linux-x86_64.sh -b -p ~/miniforge3
~/miniforge3/bin/conda init bash
exec bash
```

### 1.5 VS Code (optional, recommended)
Install **VS Code** on Windows and add the **WSL** extension. Then typing `code .` inside any Linux folder opens it in VS Code. Once inside, install the **Python** extension *in WSL* and pick the `hls4ml-tutorial` interpreter (bottom-right corner).

**Never use VS Code's "Run" button** for this project. It can pick Windows Python, which fails. Always run commands in the Ubuntu terminal.

---

## Part 2: Get the code and the Python environment, about 30 minutes

```bash
cd ~
git clone https://github.com/Orbital-Quark/hls4ml-jet-trigger.git
cd hls4ml-jet-trigger
git checkout v2-improved   # the v2-v4 code; main only has v1
conda env create -f environment.yml
conda activate hls4ml-tutorial
```
If `git checkout` says the branch doesn't exist, ask A to push it, then run `git pull` and try again.

`conda env create` takes 10–30 minutes. It installs Python 3.10, TensorFlow/Keras 3, QKeras and hls4ml at versions known to work together.

Check it worked. This should print three version numbers and no error:
```bash
python -c "import hls4ml, qkeras, keras, tensorflow as tf; print('hls4ml', hls4ml.__version__, '| keras', keras.__version__, '| tf', tf.__version__)"
```

---

## Part 3: Install Vitis 2024.1, 2–4 hours (mostly download)

Vitis HLS is AMD's tool that turns the network into a chip design. We use **2024.1**. Newer releases changed how the tool is launched, and hls4ml can lag behind.

### 3.1 Download the installer
1. Search "Vivado Vitis download" and open AMD's **Downloads** page. Pick version **2024.1**.
2. Download the **Linux Self Extracting Web Installer** (`FPGAs_AdaptiveSoCs_Unified_2024.1_..._Lin64.bin`). It's small; the rest downloads during installation.
3. Log in with your AMD account and fill in the export form.

### 3.2 Run it
Copy it from Windows into Linux and start it:
```bash
cp /mnt/c/Users/<your-windows-username>/Downloads/FPGAs_AdaptiveSoCs_Unified_2024.1_*_Lin64.bin ~/
cd ~
chmod +x FPGAs_AdaptiveSoCs_Unified_2024.1_*_Lin64.bin
./FPGAs_AdaptiveSoCs_Unified_2024.1_*_Lin64.bin
```
A graphical installer window opens. In it:
1. Log in with your AMD account and choose **Download and Install Now**.
2. **Product:** pick **Vitis**. It includes Vitis HLS and Vivado.
3. **Devices:** untick **everything** except **7 Series → Zynq-7000**. This is what keeps the install around 50 GB instead of 200+.
4. **Install location:** change it to `/home/<your-linux-username>/Xilinx`.
5. Accept the licences and install.

If it hangs at **"Generating installed device list"**, close it and run:
```bash
sudo apt install -y libtinfo5 libncurses5
sudo $(find ~/Xilinx -name installLibs.sh | head -1)
```
Then start the installer again.

### 3.3 Make Vitis available in every terminal
```bash
echo 'source ~/Xilinx/Vitis/2024.1/settings64.sh' >> ~/.bashrc
source ~/.bashrc
which vitis-run vitis_hls
```
Both should print a path. If they don't, look for `settings64.sh` under `~/Xilinx/Vitis_HLS/2024.1/` instead and use that path in the `echo` line.

### 3.4 Watch your disk space
WSL keeps Linux in one big file on C: that **grows but never shrinks by itself**. Deleting files inside Linux doesn't give the space back to Windows. If C: gets full mid-install, the install crashes. To shrink it after deleting things:
```bash
sudo fstrim -av
```
Then, in PowerShell as Administrator:
```powershell
wsl --shutdown
diskpart
```
Inside diskpart, type these one at a time (the path is where your Ubuntu disk file lives; search for `ext4.vhdx` under `C:\Users\<you>\AppData\Local\Packages\` if it's not here):
```
select vdisk file="C:\Users\<you>\AppData\Local\wsl\<id>\ext4.vhdx"
compact vdisk
exit
```

---

## Part 4: Get the data, about 2 minutes

```bash
cd ~/hls4ml-jet-trigger
conda activate hls4ml-tutorial
python prepare_data.py
```
This downloads the hls4ml jet dataset (830,000 jets, 16 features each, 5 jet types: quark, gluon, W, Z, top) and saves it in `data/`, split 80% for training and 20% for testing. It should end by printing the shape `(830000, 16)` and the five classes.

---

## Part 5: Check everything works, about 5 minutes

Run two tiny models through the whole chain: training, chip synthesis, circuit emulation and graphs.
```bash
GRID=test VERSIONS=v4 REUSE=1 python sweep.py
```
You should see two lines starting with `ok`, a summary table, and a browser window with graphs. If you see `FAIL`, check the error in `results/failed/` and the troubleshooting table at the end.

The first run also trains a "teacher" model (about 2 minutes), which is reused afterwards.

---

## Part 6: Running the project

### The versions
| Version | What it is |
|---|---|
| **v1** | The hls4ml paper's setup, reproduced |
| **v2** | Robust: drops a duplicate input, log-scales wide inputs, trains on drifted jets, learns from a teacher model |
| **v3** | Three data-driven experiments, tested only in the ablation (only one helped) |
| **v4** | v2 + real-world rate weights + a trigger that corrects its own threshold as the detector drifts |

### Run a full sweep of one version
```bash
VERSIONS=v4 N_JOBS=4 python sweep.py 2>&1 | tee sweep_v4.log
```
- It builds every design in the grid: 4 bit widths × 3 sizes × 3 pruning levels × 2 reuse settings, plus a float model and two mixed-bit models. That's 82 chip designs for v4, roughly **1.5–2.5 hours** with 4 jobs.
- The terminal shows **one line per finished design**: accuracy, trigger efficiency, drift loss, chip resources, latency, and whether it fits the chip.
- All the noisy output (training epochs, Vitis messages) goes to `logs/`.
- The **graph page opens in your browser** after the first few designs and refreshes itself every minute.
- At the end it prints a summary and draws the final graphs.
- If it stops halfway, run the same command again. Finished designs are skipped.

`N_JOBS` is how many designs are built at once. Use 4 if you have 16 GB+ RAM and 8+ cores, otherwise 2. It works without an NVIDIA GPU too, just slower.

### Look at the results again later
```bash
python analyze.py              # the summary tables
python plots.py v4 --open      # one version's graphs in the browser
python plots.py v1 v2 v4 --open   # versions compared on the same graphs
```
Graphs are saved in `figures/<version>/`. Each folder has an `index.html` page showing every graph with a short caption. Every design's numbers are in `results/results.csv`.

### The ablations (which change does what; Keras only, no chip builds)
```bash
ABLATE=v2 python ablation.py   # v1 vs each v2 change
ABLATE=v3 python ablation.py   # v2 vs each v3 change
```

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `Vitis installation not found. Make sure "vitis-run" is on PATH` | The `source ~/Xilinx/Vitis/2024.1/settings64.sh` line is missing from `~/.bashrc`, or you're in a terminal opened before you added it. Open a new terminal (Part 3.3) |
| Installer stuck on "Generating installed device list" | Part 3.2: install `libtinfo5 libncurses5` and run `installLibs.sh` with sudo |
| Dataset download fails with a checksum error | You ran it with Windows Python (e.g. VS Code's Run button). Run it from the Ubuntu terminal with `conda activate hls4ml-tutorial` |
| `code .` says "Exec format error" | That terminal is stale. Close it and open a new Ubuntu terminal |
| C: drive full / install crashed partway | Free space on Windows, then shrink the WSL disk (Part 3.4) and restart the installer, which reuses what it downloaded |
| A design shows `FAIL` | The error is in `results/failed/<design>.txt`; the full log is in `logs/points/` |
| Graphs page doesn't open | Open `figures/<version>/index.html` yourself, or run `python plots.py <version> --open` |
| Out of memory, or the laptop freezes during a sweep | Use `N_JOBS=2` |
