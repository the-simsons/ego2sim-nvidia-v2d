"""Video-to-mocap time offset by correlating a motion signal seen in the video with the mocap object's speed.
Scoring side: the offset is a ground-truth quantity and must never be fed to reconstruction."""
import numpy as np


def mocap_speed(t, pos, smooth_frames=9):
    v = np.r_[0, np.linalg.norm(np.diff(pos, axis=0), axis=1)] / np.r_[1, np.diff(t)]
    return np.convolve(np.nan_to_num(v), np.ones(smooth_frames) / smooth_frames, mode="same")


def best_lag(video_t, video_signal, mocap_t, mocap_signal, lags=np.arange(-4.0, 4.0, 0.001), min_overlap=0.75):
    """Lag (s) maximising the normalised correlation, with mocap time = video time + lag. Returns (lag, corr, scores)."""
    def z(x):
        return (x - x.mean()) / (x.std() + 1e-9)
    scores = np.full(len(lags), -2.0)
    for k, lag in enumerate(lags):
        tq = video_t + lag
        inside = (tq >= mocap_t[0]) & (tq <= mocap_t[-1])
        if inside.mean() < min_overlap:
            continue
        s = np.interp(tq[inside], mocap_t, mocap_signal)
        scores[k] = float(np.mean(z(video_signal[inside]) * z(s)))
    k = int(np.argmax(scores))
    return float(lags[k]), float(scores[k]), scores
