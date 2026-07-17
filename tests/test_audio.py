import numpy as np

from video_extraction.audio.segmentation import segment_rallies
from video_extraction.statistics import build_llm_statistics


def test_segment_rallies_splits_on_silence_and_preserves_hits():
    times = np.array([
        10.0, 11.0, 12.0, 13.0,
        30.0, 31.0, 32.0, 33.0,
    ])
    energies = np.arange(8, dtype=float)

    rallies = segment_rallies(times, energies)

    assert len(rallies) == 2
    assert rallies[0]["hit_times"] == [10.0, 11.0, 12.0, 13.0]
    assert rallies[1]["hit_energies"] == [4.0, 5.0, 6.0, 7.0]


def test_segment_rallies_clamps_video_end():
    times = np.array([10.0, 11.0, 12.0, 13.0])
    energies = np.ones(4)

    rallies = segment_rallies(times, energies, total_duration=13.5)

    assert rallies[0]["end"] == 13.5


def test_statistics_include_compact_audio_timing():
    report = [{
        "index": 1,
        "start": 1.0,
        "end": 4.0,
        "audio": {
            "sample_rate": 22050,
            "hit_times": [1.5, 2.5, 3.5],
            "hit_energies": [0.4, 0.6, 0.5],
        },
    }]

    stats = build_llm_statistics(report)

    assert stats["data_quality"]["audio"]["hit_count"] == 3
    assert stats["segments"][0]["audio"]["mean_hit_interval_seconds"] == 1.0
    assert (
        "candidate rally timing and audio hit analysis"
        in stats["analysis_capabilities"]["supported"]
    )
