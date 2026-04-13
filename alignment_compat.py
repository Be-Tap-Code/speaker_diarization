import numpy as np
import torch

from ctc_forced_aligner import (
    generate_emissions as _generate_emissions,
    get_alignments,
    get_spans,
    postprocess_results,
    preprocess_text,
)

try:
    from ctc_forced_aligner import load_alignment_model as _load_alignment_model
except ImportError:
    _load_alignment_model = None
    from ctc_forced_aligner import AlignmentSingleton


def load_alignment_model(device: str, dtype: torch.dtype | None = None):
    if _load_alignment_model is not None:
        return _load_alignment_model(device, dtype=dtype)

    del device, dtype
    alignment = AlignmentSingleton()
    return alignment.alignment_model, alignment.alignment_tokenizer


def generate_emissions(alignment_model, audio_waveform, batch_size: int = 4):
    if isinstance(audio_waveform, torch.Tensor):
        if hasattr(alignment_model, "run"):
            audio_waveform = audio_waveform.detach().cpu().numpy()
        else:
            audio_waveform = audio_waveform.to(alignment_model.dtype).to(alignment_model.device)
    elif not hasattr(alignment_model, "run"):
        audio_waveform = torch.from_numpy(np.asarray(audio_waveform)).to(alignment_model.dtype)
        audio_waveform = audio_waveform.to(alignment_model.device)
    else:
        audio_waveform = np.asarray(audio_waveform, dtype=np.float32)

    return _generate_emissions(
        alignment_model,
        audio_waveform,
        batch_size=batch_size,
    )
