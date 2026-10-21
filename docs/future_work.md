# Future Research Directions

1. **Physics-Informed Deep Learning (PINNs)**: Embedding thermodynamic conservation laws (energy balance, mass conservation across compressor and turbine stages) directly into the loss function to constrain predictions within physically feasible bounds.
2. **Transformer & State-Space Models**: Investigating linear attention mechanisms and modern State-Space Models (Mamba / S4) to model exceptionally long operational trajectories with lower memory overhead than LSTMs.
3. **Causal Representation Learning**: Moving beyond post-hoc gradient attribution toward invariant causal prediction and structural causal modeling to disentangle true damage mechanics from operating condition shifts.
4. **Active Learning & Edge Quantization**: Implementing 8-bit integer post-training quantization (INT8 / CoreML) for ultra-low-power microcontrollers directly mounted on aviation sensor clusters.
