"""Grounded captioning, the method described in the paper.

Frames are cut for *every* segment and fed alongside the facts, simulation frames at training time
and real frames at inference. The prompt makes the facts authoritative and lets the frames fill in
only what the facts leave out, so the model learns to write from the answers and merely glance at
the pixels — which is what has to survive the sim-to-real gap.

`traffic_jepa.captioning` is the other stage. It writes each caption from the VQA answers alone and
only falls back to frames where a segment has no answers. That one scored higher on the test set,
so it is what the scripts run by default.

Prompts, cleanup and the facts loader are imported from `traffic_jepa.captioning`. Only what is
specific to the grounded method lives here.
"""
