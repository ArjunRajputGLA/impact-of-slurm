# ⚡ Impact of Slurm & Intel VTune Optimization — Streamlit Lab

An interactive, high-performance computing (HPC) dashboard and distributed Deep Learning lab deployed using **Streamlit**. Based on `HPC_Workshop_Day4_Slurm_VTune_Lab.ipynb`.

---

## 🌟 Overview & Key Results

This application demonstrates the performance impact of hardware-aware **Slurm thread pinning**, **NUMA-affinity**, **Intel oneDNN SIMD vectorization (AVX-512)**, and **asynchronous I/O pipelines** on distributed deep learning workloads:

| Stage | Configuration | Cluster Throughput | Speedup | Primary Bottleneck Solved |
|---|---|---|---|---|
| **1. Baseline** | Oversubscribed (8 OMP threads, unbuffered) | 310 samples/sec | **1.00x** | Cache thrashing & lock contention |
| **2. Pinned** | Compact Affinity (`OMP_NUM_THREADS=1`) | 640 samples/sec | **2.06x** | Eliminates thread oversubscription |
| **3. Optimized** | Pinned + oneDNN (AVX-512) + Prefetch | 1,120 samples/sec | **3.61x** | Hardware SIMD & zero I/O wait |
| **4. Cluster** | Slurm Multi-Worker Emulation (2 Ranks) | 9,329 samples/sec | **Multi-Node** | Ring-AllReduce linear scaling |

---

## 🚀 Quickstart: Run Locally

### 1. Install Dependencies
```bash
git clone https://github.com/ArjunRajputGLA/impact-of-slurm.git
cd impact-of-slurm
pip install -r requirements.txt
```

### 2. Launch the Streamlit App
```bash
streamlit run app.py
```
The app will automatically open in your default browser at `http://localhost:8501`.

---

## 🌐 Deploy to Streamlit Community Cloud (Free)

1. Push this repository to GitHub:
   ```bash
   git init
   git add .
   git commit -m "Deploy HPC Slurm & VTune Lab on Streamlit"
   git branch -M main
   git remote add origin https://github.com/<your-username>/impact-of-slurm.git
   git push -u origin main
   ```
2. Go to [share.streamlit.io](https://share.streamlit.io).
3. Connect your GitHub repository.
4. Set the Main file path to: `app.py`.
5. Click **Deploy!**

---

## 🐳 Deploy with Docker

Build and run the container:
```bash
# Build Docker image
docker build -t hpc-slurm-vtune-lab .

# Run container on port 8501
docker run -p 8501:8501 hpc-slurm-vtune-lab
```

---

## 📂 Project Structure

```
├── HPC_Workshop_Day4_Slurm_VTune_Lab.ipynb   # Original Day 4 Jupyter Notebook
├── app.py                                   # Main interactive Streamlit application
├── requirements.txt                         # Python dependencies
├── submit_distributed_tf.sbatch             # Slurm cluster multi-node submission script
├── train_slurm_tf.py                        # Distributed TensorFlow training script
├── Dockerfile                               # Production Docker container setup
├── .dockerignore                            # Docker build exclusions
├── .streamlit/
│   └── config.toml                          # Custom dark theme and server settings
├── assets/
│   └── cell_10_output_0.png                 # Extracted notebook benchmark plot
└── README.md                                # Project documentation & deployment guide
```

---

## 🛠️ Features in the Streamlit App

1. **📊 Executive Dashboard**: High-level KPI metrics, before/after throughput charts, and architectural diagrams.
2. **🖥️ Intel oneAPI & Xeon Topology**: Hardware probe, CPU topology breakdown (AVX-512, AMX, NUMA), and `setvars.sh` setup guide.
3. **📝 Slurm & Distributed TF Studio**: Interactive SBATCH builder and distributed training script customizer with one-click downloads.
4. **🚀 Multi-Worker Slurm Emulation**: Live simulation of Slurm multi-worker training with dynamic port discovery, Ring-AllReduce synchronization, and console output.
5. **⚡ Intel oneDNN Kernel Profiling**: Hardware SIMD dispatch inspection under `ONEDNN_VERBOSE=1` and primitive breakdown tables.
6. **⏱️ cProfile Pipeline Hotspots**: Analysis of I/O wait times (`time.sleep`) vs GPU/CPU compute, with pipeline timeline comparisons.
7. **🎯 Intel VTune Profiler Analysis**: Microarchitecture analysis (Top-Down Analysis, memory bound vs core bound), hotspots (`gemm_pack_b`, `kmp_wait`), and cloud diagnostic notes.
8. **📈 Benchmark & Scaling Studio**: Replicated and interactive Day 4 benchmark charts and Amdahl's Law cluster scaling calculator across 1 to 64 nodes.
9. **📓 Original Notebook Explorer**: Cell-by-cell interactive viewer for all 11 cells and recorded terminal/graph outputs.

