# archive/

Older code we pivoted away from. Kept in tree because it's useful
context for "why did we end up with the current design", not because
it's part of the shipping pipeline.

## hsv_pivot/

A mid-project alternative approach that bypassed YOLO entirely and
counted balls by HSV color thresholding + connected-component blobs
crossing a user-drawn line. Calibration via a click-and-circle UI:
draw a circle on a reference ball, the median HSV inside the circle
becomes the color, π·r² becomes the per-ball pixel area, blob area
divided by ball area gives cluster counts.

Worked OK for stable scenes but couldn't handle the metal rod cutting
through balls (which split each ball into two half-blobs) and the
motion-blur trails (which read as 2-3 ball clusters from a single fast
ball). We took the lessons (per-clip calibration UX, ground-truth
labeling flow, persistent JSON results) back to the YOLO pipeline.

`hsv_pivot/templates/index.html`, `hsv_pivot/app.py`,
`hsv_pivot/hsv_counter.py` are the three artifacts.

## app.py / config.py / csrt_tracker.py / etc. (root of archive)

Pre-CSRT versions of the main pipeline. Several pivots happened:
ByteTrack-only → CSRT + ghost resurrection → tracker dedup. The
shipping versions of these files live at the project root.
