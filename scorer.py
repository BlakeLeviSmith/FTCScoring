"""
DECODE score calculation engine.

Per FTC 2025-2026 Competition Manual §10.5:
- CLASSIFIED / OVERFLOW are assessed THROUGHOUT the match (cumulative).
- PATTERN is assessed at end-of-period, against artifacts on the RAMP
  retained by the GATE, comparing each index to the MOTIF color.

Classified/overflow counts are taken as cumulative totals from the
RampTracker. Pattern comparison uses the live ramp snapshot sorted by
spatial position along the ramp (caller provides the ordered list).
"""

import config


class ScoreKeeper:
    """Tracks scores for a single alliance RAMP."""

    def __init__(self):
        self.motif_name = "GPP"
        self.motif_pattern = list(config.MOTIFS["GPP"])
        self.classified_count = 0
        self.overflow_count = 0
        self.pattern_matches = []
        self.ramp_colors = []

    def set_motif(self, motif_name):
        motif_name = motif_name.upper()
        if motif_name in config.MOTIFS:
            self.motif_name = motif_name
            self.motif_pattern = list(config.MOTIFS[motif_name])

    def update(self, ramp_colors_by_position,
               classified_total=None, overflow_total=None):
        """
        Args:
            ramp_colors_by_position: list of "G"/"P" currently on the ramp,
                sorted gate→square. Used ONLY for PATTERN matching.
            classified_total: cumulative CLASSIFIED count from RampTracker.
                If None, falls back to len(ramp_colors).
            overflow_total: cumulative OVERFLOW count from RampTracker.
                If None, falls back to 0.
        """
        self.ramp_colors = ramp_colors_by_position[:9]

        if classified_total is not None:
            self.classified_count = classified_total
        else:
            self.classified_count = len(self.ramp_colors)

        self.overflow_count = overflow_total if overflow_total is not None else 0

        self.pattern_matches = []
        for i, color in enumerate(self.ramp_colors):
            if i < len(self.motif_pattern):
                self.pattern_matches.append(color == self.motif_pattern[i])
            else:
                self.pattern_matches.append(False)

    def get_scores(self):
        pattern_match_count = sum(1 for m in self.pattern_matches if m)
        classified_points = self.classified_count * config.POINTS_CLASSIFIED_TELEOP
        overflow_points = self.overflow_count * config.POINTS_OVERFLOW_TELEOP
        pattern_points = pattern_match_count * config.POINTS_PATTERN_MATCH
        total = classified_points + overflow_points + pattern_points

        return {
            "motif_name": self.motif_name,
            "motif_pattern": self.motif_pattern,
            "ramp_colors": self.ramp_colors,
            "classified_count": self.classified_count,
            "classified_points": classified_points,
            "overflow_count": self.overflow_count,
            "overflow_points": overflow_points,
            "pattern_matches": self.pattern_matches,
            "pattern_match_count": pattern_match_count,
            "pattern_points": pattern_points,
            "total_points": total,
        }
