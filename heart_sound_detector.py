import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.fft import fft, fftfreq
import wave
import struct

class HeartSoundDetector:
    """
    Heart Sound Detection and Analysis System
    Based on standard auscultation points on the chest
    """
    
    def __init__(self):
        # Define auscultation points based on anatomical landmarks
        # Coordinates are relative to chest center (sternum midpoint)
        self.auscultation_points = {
            'A': {'name': 'Aortic', 'position': (1, 2), 'rib': '2nd ICS right'},
            'P': {'name': 'Pulmonic', 'position': (-1, 2), 'rib': '2nd ICS left'},
            'T': {'name': 'Tricuspid', 'position': (-1, -1), 'rib': '4th ICS left'},
            'M': {'name': 'Mitral', 'position': (2, -3), 'rib': '5th ICS midclavicular'}
        }
        
        # Sampling parameters
        self.sample_rate = 4000  # Hz (standard for heart sound)
        self.duration = 5  # seconds
        
    def calculate_chest_coordinates(self, chest_width_cm, chest_height_cm):
        """
        Calculate absolute positions on chest based on anatomical measurements
        
        Equations:
        x_abs = x_center + (x_rel * chest_width_cm / 20)
        y_abs = y_center + (y_rel * chest_height_cm / 30)
        
        Where:
        - x_center, y_center = center of sternum
        - x_rel, y_rel = relative coordinates from standard points
        - Normalization factors (20, 30) are based on average adult chest
        """
        coordinates = {}
        x_center = chest_width_cm / 2
        y_center = chest_height_cm / 2
        
        for point, data in self.auscultation_points.items():
            x_rel, y_rel = data['position']
            x_abs = x_center + (x_rel * chest_width_cm / 20)
            y_abs = y_center + (y_rel * chest_height_cm / 30)
            
            coordinates[point] = {
                'x': x_abs,
                'y': y_abs,
                'name': data['name'],
                'rib': data['rib']
            }
        
        return coordinates
    
    def generate_synthetic_heart_sound(self, heart_rate=75):
        """
        Generate synthetic heart sound for testing
        
        Heart sound model:
        S1 (lub): 30-45 Hz, duration ~100ms
        S2 (dub): 50-70 Hz, duration ~70ms
        
        Equation for each beat:
        s(t) = A1 * exp(-α1*t) * sin(2π*f1*t) + A2 * exp(-α2*t) * sin(2π*f2*t)
        
        Where:
        - A1, A2 = amplitudes of S1 and S2
        - α1, α2 = decay rates
        - f1, f2 = fundamental frequencies
        """
        beat_period = 60.0 / heart_rate  # seconds per beat
        t = np.linspace(0, self.duration, int(self.sample_rate * self.duration))
        signal_data = np.zeros_like(t)
        
        # Parameters for S1 (lub)
        f1 = 37.5  # Hz
        A1 = 1.0
        alpha1 = 30
        duration_s1 = 0.1  # 100ms
        
        # Parameters for S2 (dub)
        f2 = 60  # Hz
        A2 = 0.7
        alpha2 = 40
        duration_s2 = 0.07  # 70ms
        s2_delay = 0.12  # S2 occurs 120ms after S1
        
        # Generate beats
        num_beats = int(self.duration / beat_period)
        for beat in range(num_beats):
            beat_start = beat * beat_period
            
            # S1 sound
            mask_s1 = (t >= beat_start) & (t < beat_start + duration_s1)
            t_s1 = t[mask_s1] - beat_start
            signal_data[mask_s1] += A1 * np.exp(-alpha1 * t_s1) * np.sin(2 * np.pi * f1 * t_s1)
            
            # S2 sound
            s2_start = beat_start + s2_delay
            mask_s2 = (t >= s2_start) & (t < s2_start + duration_s2)
            t_s2 = t[mask_s2] - s2_start
            signal_data[mask_s2] += A2 * np.exp(-alpha2 * t_s2) * np.sin(2 * np.pi * f2 * t_s2)
        
        # Add noise
        noise = np.random.normal(0, 0.05, len(signal_data))
        signal_data += noise
        
        # Normalize
        signal_data = signal_data / np.max(np.abs(signal_data))
        
        return t, signal_data
    
    def bandpass_filter(self, data, lowcut=20, highcut=200):
        """
        Apply Butterworth bandpass filter
        
        Transfer function:
        H(s) = (s/Q)^n / ((s/Q)^n + ... + 1)
        
        Where:
        - n = filter order
        - Q = quality factor
        - s = complex frequency
        """
        nyquist = 0.5 * self.sample_rate
        low = lowcut / nyquist
        high = highcut / nyquist
        
        # Design filter
        b, a = signal.butter(4, [low, high], btype='band')
        
        # Apply filter
        filtered_data = signal.filtfilt(b, a, data)
        
        return filtered_data
    
    def detect_heart_beats(self, signal_data):
        """
        Detect heart beats using envelope detection and peak finding
        
        Envelope equation (Hilbert transform):
        A(t) = sqrt(s(t)^2 + H[s(t)]^2)
        
        Where:
        - s(t) = original signal
        - H[s(t)] = Hilbert transform of signal
        - A(t) = amplitude envelope
        """
        # Calculate envelope using Hilbert transform
        analytic_signal = signal.hilbert(signal_data)
        amplitude_envelope = np.abs(analytic_signal)
        
        # Find peaks
        distance = int(0.4 * self.sample_rate)  # Minimum 0.4s between beats
        peaks, properties = signal.find_peaks(
            amplitude_envelope, 
            distance=distance,
            prominence=0.3
        )
        
        # Calculate heart rate
        if len(peaks) > 1:
            intervals = np.diff(peaks) / self.sample_rate
            heart_rate = 60.0 / np.mean(intervals)
        else:
            heart_rate = 0
        
        return peaks, amplitude_envelope, heart_rate
    
    def compute_spectrum(self, signal_data):
        """
        Compute frequency spectrum using FFT
        
        DFT equation:
        X(k) = Σ x(n) * exp(-j*2π*k*n/N)
        
        Power spectral density:
        PSD(f) = |X(f)|^2 / N
        
        Where:
        - X(k) = frequency domain representation
        - x(n) = time domain signal
        - N = number of samples
        """
        N = len(signal_data)
        yf = fft(signal_data)
        xf = fftfreq(N, 1/self.sample_rate)
        
        # Only positive frequencies
        positive_freq_idx = xf > 0
        xf = xf[positive_freq_idx]
        yf = yf[positive_freq_idx]
        
        # Power spectral density
        psd = np.abs(yf)**2 / N
        
        return xf, psd
    
    def digitize_for_transmission(self, signal_data, bits=16):
        """
        Convert analog signal to digital format for transmission
        
        Quantization equation:
        Q(x) = round(x * (2^bits - 1) / (x_max - x_min))
        
        Where:
        - x = input signal value
        - bits = bit depth
        - x_max, x_min = signal range
        """
        # Normalize to [-1, 1]
        normalized = signal_data / np.max(np.abs(signal_data))
        
        # Quantize
        max_value = 2**(bits-1) - 1
        quantized = np.round(normalized * max_value).astype(np.int16)
        
        return quantized
    
    def save_as_wav(self, signal_data, filename='heart_sound.wav'):
        """
        Save signal as WAV file for transmission
        """
        # Convert to 16-bit integer
        quantized = self.digitize_for_transmission(signal_data)
        
        # Write WAV file
        with wave.open(filename, 'w') as wav_file:
            wav_file.setnchannels(1)  # Mono
            wav_file.setsampwidth(2)  # 2 bytes (16 bits)
            wav_file.setframerate(self.sample_rate)
            
            # Write frames
            for sample in quantized:
                wav_file.writeframes(struct.pack('h', sample))
        
        print(f"Saved to {filename}")
        return filename
    
    def visualize_analysis(self, t, signal_data, filtered_data, peaks, 
                          envelope, xf, psd, chest_width=30, chest_height=40):
        """
        Create comprehensive visualization
        """
        fig = plt.figure(figsize=(16, 12))
        
        # 1. Chest location mapping
        ax1 = plt.subplot(2, 3, 1)
        coords = self.calculate_chest_coordinates(chest_width, chest_height)
        
        # Draw chest outline
        chest_x = [0, chest_width, chest_width, 0, 0]
        chest_y = [0, 0, chest_height, chest_height, 0]
        ax1.plot(chest_x, chest_y, 'k-', linewidth=2)
        
        # Plot auscultation points
        colors = {'A': 'red', 'P': 'blue', 'T': 'green', 'M': 'orange'}
        for point, data in coords.items():
            ax1.plot(data['x'], data['y'], 'o', color=colors[point], 
                    markersize=15, label=f"{point}: {data['name']}")
            ax1.text(data['x'], data['y']-2, data['rib'], 
                    ha='center', fontsize=8)
        
        ax1.set_xlabel('Width (cm)')
        ax1.set_ylabel('Height (cm)')
        ax1.set_title('Auscultation Points on Chest')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_aspect('equal')
        
        # 2. Original signal
        ax2 = plt.subplot(2, 3, 2)
        ax2.plot(t, signal_data, 'b-', linewidth=0.5)
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Amplitude')
        ax2.set_title('Original Heart Sound Signal')
        ax2.grid(True, alpha=0.3)
        
        # 3. Filtered signal with detected beats
        ax3 = plt.subplot(2, 3, 3)
        ax3.plot(t, filtered_data, 'g-', linewidth=0.5)
        ax3.plot(t, envelope, 'r--', linewidth=1, label='Envelope')
        ax3.plot(t[peaks], filtered_data[peaks], 'ro', markersize=8, 
                label='Detected Beats')
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('Amplitude')
        ax3.set_title('Filtered Signal with Beat Detection')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Frequency spectrum
        ax4 = plt.subplot(2, 3, 4)
        ax4.plot(xf, 10*np.log10(psd), 'b-')
        ax4.set_xlabel('Frequency (Hz)')
        ax4.set_ylabel('Power (dB)')
        ax4.set_title('Frequency Spectrum')
        ax4.set_xlim([0, 300])
        ax4.grid(True, alpha=0.3)
        
        # 5. Digitized signal (zoomed)
        ax5 = plt.subplot(2, 3, 5)
        zoom_samples = 1000
        quantized = self.digitize_for_transmission(filtered_data[:zoom_samples])
        ax5.step(range(zoom_samples), quantized, 'r-', linewidth=0.5)
        ax5.set_xlabel('Sample number')
        ax5.set_ylabel('Quantized value')
        ax5.set_title('Digitized Signal (16-bit)')
        ax5.grid(True, alpha=0.3)
        
        # 6. Heart rate analysis
        ax6 = plt.subplot(2, 3, 6)
        if len(peaks) > 1:
            intervals = np.diff(peaks) / self.sample_rate
            heart_rates = 60.0 / intervals
            ax6.plot(range(len(heart_rates)), heart_rates, 'mo-', linewidth=2)
            ax6.axhline(y=np.mean(heart_rates), color='r', linestyle='--', 
                       label=f'Mean: {np.mean(heart_rates):.1f} BPM')
            ax6.set_xlabel('Beat number')
            ax6.set_ylabel('Heart Rate (BPM)')
            ax6.set_title('Heart Rate Variability')
            ax6.legend()
            ax6.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig

# Main execution
if __name__ == "__main__":
    # Initialize detector
    detector = HeartSoundDetector()
    
    # Generate synthetic heart sound
    print("Generating synthetic heart sound...")
    t, raw_signal = detector.generate_synthetic_heart_sound(heart_rate=72)
    
    # Apply bandpass filter
    print("Applying bandpass filter...")
    filtered_signal = detector.bandpass_filter(raw_signal)
    
    # Detect heart beats
    print("Detecting heart beats...")
    peaks, envelope, heart_rate = detector.detect_heart_beats(filtered_signal)
    print(f"Detected heart rate: {heart_rate:.1f} BPM")
    print(f"Number of beats detected: {len(peaks)}")
    
    # Compute frequency spectrum
    print("Computing frequency spectrum...")
    frequencies, psd = detector.compute_spectrum(filtered_signal)
    
    # Save as WAV file
    print("Digitizing and saving...")
    detector.save_as_wav(filtered_signal)
    
    # Visualize results
    print("Creating visualization...")
    fig = detector.visualize_analysis(t, raw_signal, filtered_signal, 
                                      peaks, envelope, frequencies, psd)
    plt.savefig('heart_sound_analysis.png', dpi=300, bbox_inches='tight')
    print("Saved visualization to heart_sound_analysis.png")
    plt.show()
    
    # Display chest coordinates
    print("\nAuscultation Point Coordinates (for 30x40 cm chest):")
    coords = detector.calculate_chest_coordinates(30, 40)
    for point, data in coords.items():
        print(f"{point} ({data['name']}): x={data['x']:.1f}cm, y={data['y']:.1f}cm - {data['rib']}")
