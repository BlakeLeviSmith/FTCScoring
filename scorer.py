"""
DECODE score calculation engine.
Takes detected ball patterns and compares against the active MOTIF.
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
        """Set the active MOTIF (GPP, PGP, or PPG)."""
        motif_name = motif_name.upper()
        if motif_name in config.MOTIFS:
            self.motif_name = motif_name
            self.motif_pattern = list(config.MOTIFS[motif_name])

    def update(self, detected_colors):
        """
        Update scores based on detected ball colors on the RAMP.

        Args:
            detected_colors: list of color strings ["G", "P", "P", ...] in order
                            from Gate (position 1) to Square (position 9).
                            Max 9 items for CLASSIFIED, extras are OVERFLOW.
        """
        self.ramp_colors = detected_colors[:9]  # Max 9 CLASSIFIED
        self.classified_count = len(self.ramp_colors)
        self.overflow_count = max(0, len(detected_colors) - 9)

        # Calculate pattern matches
        self.pattern_matches = []
        for i, color in enumerate(self.ramp_colors):
            if i < len(self.motif_pattern):
                self.pattern_matches.append(color == self.motif_pattern[i])
            else:
                self.pattern_matches.append(False)

    def get_scores(self):
        """
        Return current score breakdown.

        Returns:
            dict with all scoring info
        """
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
