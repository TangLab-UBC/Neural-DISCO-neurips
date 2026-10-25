import numpy as np


def _total_window(input_window, output_window, delay_factor, target_mode):
    if target_mode == "future":
        return input_window + output_window + delay_factor
    if target_mode == "input_trailing":
        return input_window
    raise ValueError(f"Unsupported target_mode: {target_mode}")


def _target_bounds(t, input_window, output_window, delay_factor, target_mode):
    if target_mode == "future":
        out_start = t + input_window + delay_factor
        out_end = out_start + output_window
        return out_start, out_end
    if target_mode == "input_trailing":
        out_end = t + input_window
        out_start = out_end - output_window
        return out_start, out_end
    raise ValueError(f"Unsupported target_mode: {target_mode}")


def window_condition_array(array, input_window, output_window, delay_factor, target_mode="future"):
    total_window = _total_window(input_window, output_window, delay_factor, target_mode)
    samples = []
    for trial in array:
        n_time = trial.shape[0]
        for t in range(n_time - total_window + 1):
            samples.append(trial[t : t + input_window].T)
    if not samples:
        raise ValueError("No valid spike windows constructed")
    return np.stack(samples, axis=0).astype(np.float32)


def window_condition_context_array(array, input_window, output_window, delay_factor, context_window, target_mode="future"):
    total_window = _total_window(input_window, output_window, delay_factor, target_mode)
    samples = []
    for trial in array:
        n_time = trial.shape[0]
        for t in range(n_time - total_window + 1):
            input_end = t + input_window
            context_start = max(0, input_end - context_window)
            context = np.zeros((context_window, trial.shape[1]), dtype=trial.dtype)
            valid = trial[context_start:input_end]
            context[-len(valid) :] = valid
            samples.append(context.T)
    if not samples:
        raise ValueError("No valid spike context windows constructed")
    return np.stack(samples, axis=0).astype(np.float32)


def window_condition_target(array, input_window, output_window, delay_factor, target_mode="future"):
    total_window = _total_window(input_window, output_window, delay_factor, target_mode)
    samples = []
    for trial in array:
        n_time = trial.shape[0]
        for t in range(n_time - total_window + 1):
            out_start, out_end = _target_bounds(
                t, input_window, output_window, delay_factor, target_mode
            )
            samples.append(trial[out_start:out_end].reshape(-1))
    if not samples:
        raise ValueError("No valid behaviour windows constructed")
    return np.stack(samples, axis=0).astype(np.float32)


def window_condition_spike_target(array, input_window, output_window, delay_factor, target_mode="future"):
    total_window = _total_window(input_window, output_window, delay_factor, target_mode)
    samples = []
    for trial in array:
        n_time = trial.shape[0]
        for t in range(n_time - total_window + 1):
            out_start, out_end = _target_bounds(
                t, input_window, output_window, delay_factor, target_mode
            )
            samples.append(trial[out_start:out_end].T)
    if not samples:
        raise ValueError("No valid spike targets constructed")
    return np.stack(samples, axis=0).astype(np.float32)
