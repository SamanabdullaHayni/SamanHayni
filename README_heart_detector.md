# Heart Sound Detection and Analysis System

A comprehensive Python-based system for detecting and analyzing heart sounds based on standard auscultation points on the chest.

## Overview

This system implements a heart sound detector that can:
- Generate synthetic heart sounds for testing
- Apply bandpass filtering to isolate heart sound frequencies
- Detect individual heartbeats using envelope detection
- Compute frequency spectrum analysis
- Digitize signals for transmission
- Visualize comprehensive analysis results

## Features

### 1. Auscultation Point Mapping
Maps standard clinical auscultation points on the chest:
- **A (Aortic)**: 2nd ICS (Intercostal Space) right
- **P (Pulmonic)**: 2nd ICS left
- **T (Tricuspid)**: 4th ICS left
- **M (Mitral)**: 5th ICS midclavicular

### 2. Heart Sound Generation
Generates synthetic heart sounds based on physiological parameters:
- **S1 (lub)**: 30-45 Hz, duration ~100ms
- **S2 (dub)**: 50-70 Hz, duration ~70ms
- Configurable heart rate (default: 75 BPM)

### 3. Signal Processing
- **Butterworth bandpass filter**: 20-200 Hz range
- **Hilbert transform**: For envelope detection
- **FFT analysis**: Frequency spectrum computation
- **Peak detection**: Identifies individual heartbeats

### 4. Data Export
- **WAV file export**: 16-bit digitized audio
- **Visualization**: Comprehensive 6-panel analysis plot

## Installation

### Requirements
- Python 3.7+
- NumPy
- Matplotlib
- SciPy

### Install Dependencies
```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage
```python
python heart_sound_detector.py
```

This will:
1. Generate a synthetic heart sound signal at 72 BPM
2. Filter and analyze the signal
3. Detect heartbeats
4. Save results as `heart_sound.wav` and `heart_sound_analysis.png`

### Using in Your Code
```python
from heart_sound_detector import HeartSoundDetector

# Initialize detector
detector = HeartSoundDetector()

# Generate heart sound at specific rate
t, signal = detector.generate_synthetic_heart_sound(heart_rate=80)

# Filter the signal
filtered = detector.bandpass_filter(signal)

# Detect beats
peaks, envelope, heart_rate = detector.detect_heart_beats(filtered)
print(f"Detected heart rate: {heart_rate:.1f} BPM")

# Get chest coordinates for visualization
coords = detector.calculate_chest_coordinates(chest_width_cm=30, chest_height_cm=40)
```

## Technical Details

### Mathematical Models

#### Heart Sound Model
```
s(t) = A1 * exp(-α1*t) * sin(2π*f1*t) + A2 * exp(-α2*t) * sin(2π*f2*t)
```
Where:
- A1, A2 = amplitudes of S1 and S2
- α1, α2 = decay rates
- f1, f2 = fundamental frequencies

#### Envelope Detection
```
A(t) = sqrt(s(t)^2 + H[s(t)]^2)
```
Where:
- s(t) = original signal
- H[s(t)] = Hilbert transform of signal
- A(t) = amplitude envelope

#### Chest Coordinate Calculation
```
x_abs = x_center + (x_rel * chest_width_cm / 20)
y_abs = y_center + (y_rel * chest_height_cm / 30)
```
Normalization factors (20, 30) are based on average adult chest dimensions.

## Output

### Console Output
```
Generating synthetic heart sound...
Applying bandpass filter...
Detecting heart beats...
Detected heart rate: 72.0 BPM
Number of beats detected: 5
Computing frequency spectrum...
Digitizing and saving...
Saved to heart_sound.wav
Creating visualization...
Saved visualization to heart_sound_analysis.png

Auscultation Point Coordinates (for 30x40 cm chest):
A (Aortic): x=15.0cm, y=22.7cm - 2nd ICS right
P (Pulmonic): x=15.0cm, y=22.7cm - 2nd ICS left
T (Tricuspid): x=13.5cm, y=18.7cm - 4th ICS left
M (Mitral): x=18.0cm, y=16.0cm - 5th ICS midclavicular
```

### Visualization
The generated `heart_sound_analysis.png` contains 6 panels:
1. **Auscultation Points on Chest**: Visual mapping of listening points
2. **Original Heart Sound Signal**: Raw generated signal
3. **Filtered Signal with Beat Detection**: Processed signal with detected beats
4. **Frequency Spectrum**: Power spectral density analysis
5. **Digitized Signal**: 16-bit quantized waveform
6. **Heart Rate Variability**: Beat-to-beat heart rate analysis

## Configuration

### Sampling Parameters
- Sample rate: 4000 Hz (standard for heart sound)
- Duration: 5 seconds
- Bit depth: 16-bit for digitization

### Filter Parameters
- Low cutoff: 20 Hz
- High cutoff: 200 Hz
- Filter order: 4 (Butterworth)

## Medical Context

The system is based on standard medical auscultation practices:
- **ICS**: Intercostal Space (the space between ribs)
- **Aortic area**: Best for detecting aortic valve sounds
- **Pulmonic area**: Best for detecting pulmonary valve sounds
- **Tricuspid area**: Best for detecting tricuspid valve sounds
- **Mitral area**: Best for detecting mitral valve sounds (apex of heart)

## Error Resolution

The original code had a syntax error at line 397 (extra closing parenthesis). This has been corrected in the current version.

### Original Issue
```python
print(f"{point} ({data['name']}): x={data['x']:.1f}cm, y={data['y']:.1f}cm - {data['rib']}")  # Extra )
```

### Fixed Version
```python
print(f"{point} ({data['name']}): x={data['x']:.1f}cm, y={data['y']:.1f}cm - {data['rib']}")
```

## License

This project is part of the SamanHayni personal portfolio.

## Author

Saman Abdullah Hayni

## Contributing

Feel free to submit issues or pull requests for improvements.
