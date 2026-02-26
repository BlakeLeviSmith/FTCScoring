# DECODE Scoring Rules Reference

Quick reference for the FTC 2025-2026 DECODE game scoring system.
Source: DECODE_Competition_Manual_TU24.pdf, Sections 8-10.

## Game Overview

- 2 ALLIANCES of 2 teams each compete
- 30-second AUTO period + 8-second transition + 2-minute TELEOP period
- ROBOTS collect purple and green ball ARTIFACTS, score them into GOAL
- ARTIFACTS exit GOAL through CLASSIFIER (SQUARE -> RAMP -> GATE)
- PATTERN of balls on RAMP is compared against randomly selected MOTIF

## ARTIFACTS

- **Type**: 5-inch Gopher ResisDent polypropylene balls (4.9 +/- 0.25 in)
- **Colors**: Purple (P) and Green (G)
- **Quantity per match**: 24 Purple + 12 Green = 36 total
- **Not perfectly spherical** - plan for size variation

## CLASSIFIER Components

### SQUARE
- Location at top of RAMP where scoring is assessed
- ARTIFACT passes through SQUARE first after exiting GOAL

### RAMP
- Aluminum extrusion structure below SQUARE
- Holds up to **9 CLASSIFIED ARTIFACTS** before overflow begins
- Balls sit in order from GATE (position 1) to SQUARE (position 9)

### GATE
- At bottom of RAMP, prevents CLASSIFIED artifacts from exiting
- Push-to-open mechanism (ROBOT activates it)
- When opened, CLASSIFIED artifacts roll out into SECRET TUNNEL ZONE
- OVERFLOW artifacts pass over the top of closed GATE

## Artifact Classification

| Status | Definition |
|---|---|
| **CLASSIFIED** | Enters GOAL top, exits archway, passes through SQUARE, moves directly to RAMP (doesn't roll over other balls) |
| **OVERFLOW** | Passes through SQUARE but rolls over one or more artifacts on RAMP |
| **Not scored** | Doesn't enter GOAL through top, doesn't exit archway, or doesn't pass through SQUARE |

## MOTIFS and PATTERNS

### The 3 MOTIFS (shown on OBELISK via AprilTags)

| MOTIF | AprilTag ID | 3-letter code | Full 9-position pattern |
|---|---|---|---|
| GPP | 21 | G-P-P | G,P,P,G,P,P,G,P,P |
| PGP | 22 | P-G-P | P,G,P,P,G,P,P,G,P |
| PPG | 23 | P-P-G | P,P,G,P,P,G,P,P,G |

The MOTIF is randomized by FIELD STAFF before each match after DRIVE TEAM setup.

### PATTERN Scoring
- Each RAMP position (1-9) has an expected color from the MOTIF
- Position 1 is at the GATE end, Position 9 is at the SQUARE end
- If artifact color at position N matches MOTIF color at position N = PATTERN points
- Artifacts must be CLASSIFIED and retained by GATE
- PATTERN is assessed at end of AUTO and end of TELEOP

### Example (MOTIF = GPP)
```
Position:  1  2  3  4  5  6  7  8  9
Expected:  G  P  P  G  P  P  G  P  P
Actual:    G  G  P  G  P  P  G  P  -
Match?:    Y  N  Y  Y  Y  Y  Y  Y  -
Points:    2  0  2  2  2  2  2  2  0 = 14 PATTERN points
```

## Point Values Table

| Action | AUTO pts | TELEOP pts | Notes |
|---|---|---|---|
| **LEAVE** | 3 | - | Robot moves off LAUNCH LINE |
| **CLASSIFIED** | 3 | 3 | Artifact on RAMP properly |
| **OVERFLOW** | 1 | 1 | Artifact passed over others |
| **DEPOT** | - | 1 | Artifact over DEPOT tape at base of GOAL |
| **PATTERN** | 2 | 2 | Per artifact matching MOTIF position |
| **Partially to BASE** | - | 5 | Robot partially in BASE ZONE |
| **Fully to BASE** | - | 10 | Robot fully in BASE ZONE |
| **Both ROBOTS to BASE** | - | 10 | Alliance bonus if both fully returned |

## Ranking Points (RP)

| RP Type | Regular Events | Regional Championships |
|---|---|---|
| **MOVEMENT RP** | LEAVE+BASE >= 16 | >= 21 |
| **GOAL RP** | Artifacts scored >= 36 | >= 42 |
| **PATTERN RP** | PATTERN points >= 18 | >= 22 |
| **WIN** | 3 RP | 3 RP |
| **TIE** | 1 RP | 1 RP |

## Penalties

| Penalty | Effect |
|---|---|
| MINOR FOUL | +5 points to opponent |
| MAJOR FOUL | +15 points to opponent |
| YELLOW CARD | Warning, carries forward |
| RED CARD | DISQUALIFIED for that match |

## Match Timing

| Event | Timer | Audio Cue |
|---|---|---|
| Match start | 2:30 | "Cavalry Charge" |
| AUTO ends | 2:00 | "Buzzer x 3" |
| Transition | 0:07 to 0:01 | "Drivers, pick up controllers, 3-2-1" |
| TELEOP begins | 2:00 | "3 Bells" |
| Final 20 seconds | 0:20 | "Train Whistle" |
| Match end | 0:00 | "3-second Buzzer" |

## What Our Vision System Needs to Detect

For each alliance RAMP (Red and Blue):
1. **Count of CLASSIFIED artifacts** on the RAMP (0-9)
2. **Color of each artifact** at each position (P or G)
3. **Order of artifacts** from GATE (pos 1) to SQUARE (pos 9)
4. **Count of OVERFLOW** (artifacts that passed over others)

With MOTIF input, calculate:
5. **PATTERN score** (how many positions match)
6. **Total artifact points** (CLASSIFIED + OVERFLOW + PATTERN)
7. **Whether GOAL RP and PATTERN RP thresholds are met**
