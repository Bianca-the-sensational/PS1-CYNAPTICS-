🧠 Overview

The Frequency Quest is an audio classification project that converts .wav audio files into Mel spectrograms and classifies them using a fine-tuned ResNet-34 model.
This approach treats spectrograms as images, enabling the use of powerful pretrained CNNs for sound recognition.

The model performs:
1. Audio preprocessing
2. Spectrogram generation
3. Data augmentation
4. CNN-based classification
5. Output generation in submission.csv

-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# PRE - PROCESSING PIPIELINE

- 🎧 Input: Audio (.wav) files  
- 🔄 Resample audio to 22,050 Hz  
- 🔊 Convert stereo to mono  
- 🎵 Compute Mel Spectrogram (64 Mel bins)  
- 🛠️ Data augmentation (for training dataset only):  
  - Frequency masking  
  - Time masking  
- 🔄 Normalize and resize to (3 × 224 × 224) to be a suitable input for ResNet-34 
- 🖼️ Use ResNet-34 pretrained on ImageNet as backbone  
- 🔧 Custom classifier head  
- 🏷️ Output: Predicted class label

------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

📂 Dataset Structure

The dataset inside Google Drive is following this structure:

```
the-frequency-quest/
 ├── train/
 │     └── train/
 │           ├── class_1/
 │           ├── class_2/
 │           └── ...
 ├── test/
 │     └── test/
 │           ├── audio_1.wav
 │           ├── audio_2.wav
 │           └── ...
 ├── cache_mel/        # Directory has been created to save pre-computed mel spectrograms (efficient execution & faster speed)

```
 -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------

 🔧 Preprocessing Details
-> Resampling :
All audio is resampled to 22,050 Hz.

-> Mono conversion :
Stereo audio → averaged into a single channel.

-> Mel spectrogram generation :
Using torchaudio:
n_mels = 64
n_fft = 1024
hop_length = 512
sample_rate = 22,050 Hz

-> Normalization :
Spectrograms standardized channel-wise.

-> Padding / Truncation: 
Each spectrogram is fixed to 128 frames.

-> SpecAugment (training only) :
1. Frequency masking
2. Time masking

-> Caching :
Processed Mel spectrograms saved to cache_mel/ for faster and efficient training.

------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

🧱 Model Architecture

-> Base Model

-> ResNet-34 pretrained on ImageNet( Dataset containing 1000 classes )

-> All layers unfrozen for full fine-tuning( Hybrid mode of fine tuning )

-> Input Adjustment

-> Mel spectrograms are repeated across 3 channels and resized to 224×224.( suitable for ResNet-34 Model )

-> Custom Classification Head :
# Code snippet
nn.Linear(in_features, 256)

nn.ReLU()

nn.Dropout(0.3)

nn.Linear(256, num_classes)

-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------

📈 Training Configuration

Loss : CrossEntropyLoss( for multi class classification )

Optimizer	Adam : (lr = 3e-4)

Scheduler :	ReduceLROnPlateau

Batch Size : 32

Epochs : 15

Train/Val Split	: 80/20

Device : GPU if available

During each epoch:
1. Training loss is computed
2. Validation accuracy is measured
3. Best model is saved as: best_model.pth

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

🧪 Testing & Submission Generation

After training:
1. The best model is loaded
2. Predictions are generated for test audio files
3. Labels are mapped back to class names
4. Predictions are saved in a ".csv" file ("submission.csv")
