## Step-by-Step Project Roadmap

**1. Data Loading and Exploration (Days 1-2)**

- Write a script in `src/dataset.py` to efficiently open and read the compressed `.b2` files using `blosc2`.
    
- Use `EDA.ipynb` to plot the CSI matrices and ATI timing intervals.
    
- Analyze the class distribution in `samples.csv` to see if you have an equal number of examples for 0, 1, 2, 3, and 4 people. If the data is imbalanced, you will need to account for this later in your loss function.
    

**2. The End-to-End Baseline (Days 3-4)**

- Do not build a massive neural network yet. Build a simple, naive model (like a basic CNN or even a Random Forest) in `src/models/baselines.py`.
    
- Write your PyTorch `Dataset` and `DataLoader` classes to feed batches of data into your model.
    
- Write a basic training loop in `train.py`.
    
- Use `export.py` to convert this basic model to ONNX format.
    
- Upload this baseline model to the Agorize platform to ensure your pipeline works from start to finish. This guarantees you won't be scrambling with ONNX conversion errors on the final day.
    

**3. Deep Learning Architecture (Days 5-10)**

- Start building your real model in `src/models/neural_nets.py`.
    
- Implement a CNN to extract features from the spatial/frequency dimensions (antennas and subcarriers).
    
- Attach a temporal layer (LSTM, GRU, or Transformer) to process the 600 time frames.
    
- Modify your forward pass to ingest both the `frames` tensor and the `intervals` tensor, as the timing gaps contain valuable information.
    
- Train this model on the labelled dataset (`S`) and track your validation accuracy.
    

**4. Tackling Cross-Room Generalization (Days 11-16)**

- This is where you win the competition. If your model just memorizes the training rooms, it will fail on the test set.
    
- **Data Augmentation:** Artificially add noise, mask out certain subcarriers, or randomly drop frames during training to force the model to learn robust features rather than memorizing exact signal values.
    
- **Leverage Unlabelled Data:** You have an unlabelled dataset (`U`). Train an Autoencoder on this data to learn the general physics of wireless propagation, then use the encoder weights as the starting point for your headcount classifier.
    
- **Domain Adversarial Training:** Add a secondary classification head to your network that tries to guess which room the data came from. Reverse the gradients from this head so the main network is actively penalized for learning room-specific features.
    

**5. Profiling and Optimization (Final Days)**

- Check your model size. The ONNX file must be under 100 MB. If it is too large, use techniques like quantization (converting float32 weights to float16 or int8) to shrink it.
    
- Test your inference speed. The CPU prediction time must average less than 0.7 seconds per sample. If your model is a massive Transformer, it might be too slow. Swap heavy layers for lighter ones (like MobileNet blocks or smaller GRUs) until you hit the target.
    
- Submit your final, optimized `.onnx` file to the platform.