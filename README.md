# GNN-for-Space-Debris: Real-Time Collision Risk Assessment

Official repository for the paper:  
**Graph Neural Networks for Real-Time Collision Risk Assessment in Large Satellite Constellations**  
[arXiv link will be added soon] (November 2025)

## Overview
This project demonstrates a scalable GNN architecture for predicting high-risk satellite conjunctions in mega-constellations (up to 100k objects). Key results:
- >90% recall on high-risk events (15-30 min horizon)
- Sub-second inference time
- Linear O(n) scaling via spatial hashing + message passing
- Validated on synthetic LEO data and real 2025 Space-Track CDMs

## Quick Start
```bash
conda create -n gnn-space python=3.10 pytorch pytorch-geometric -c pytorch -c pyg
pip install -r requirements.txt
python src/inference.py --size 100000  # ~920 ms on GPU
