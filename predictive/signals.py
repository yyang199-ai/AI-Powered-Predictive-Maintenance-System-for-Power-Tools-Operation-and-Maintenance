"""Optional raw-window extractor; same causal smoothing for offline and live use."""
import numpy as np
from scipy.signal import lfilter


def quality_score(signal, sample_times, sample_rate, saturation_limit=None):
    x=np.asarray(signal,dtype=float);times=np.asarray(sample_times,dtype=float)
    if len(x)<4 or len(x)!=len(times) or sample_rate<=0:return 0.0
    if not np.isfinite(x).all() or not np.isfinite(times).all():return 0.0
    if np.any(np.diff(times)<=0) or np.max(np.abs(np.diff(times)-1/sample_rate))>0.25/sample_rate:return 0.0
    if np.ptp(x)<1e-10:return 0.0
    if saturation_limit is not None and np.mean(np.abs(x)>=saturation_limit)>0.01:return 0.0
    return 1.0


def waveform_features(signal,sample_rate,filter_state=None):
    x=np.asarray(signal,dtype=float)
    if sample_rate<=0 or len(x)<8 or not np.isfinite(x).all():raise ValueError('波形至少8个有限采样点，采样率必须大于0。')
    # Persistent zi enables identical chunked and continuous causal processing.
    b=np.ones(3)/3
    filtered,state=lfilter(b,[1],x,zi=np.zeros(2) if filter_state is None else filter_state)
    centered=filtered-filtered.mean();rms=np.sqrt(np.mean(centered**2));std=max(centered.std(),1e-8)
    window=np.hanning(len(centered));spectrum=np.abs(np.fft.rfft(centered*window))*2/max(window.sum(),1e-8)
    frequency=np.fft.rfftfreq(len(centered),1/sample_rate)
    peak=int(np.argmax(spectrum[1:])+1)
    return {'rms':float(rms),'peak_to_peak':float(np.ptp(centered)),'kurtosis_pearson':float(np.mean((centered/std)**4)),
            'crest_factor':float(np.max(np.abs(centered))/max(rms,1e-8)),'spectral_peak_hz':float(frequency[peak]),
            'spectral_peak_amplitude':float(spectrum[peak])},state
