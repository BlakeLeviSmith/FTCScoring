# roi_configs/

Per-clip zone definitions. One JSON per clip, named `<clipname>.json`.

```json
{
  "red":  {"roi": [...], "gate": [...], ...},  // polygons in normalized coords
  "blue": {"roi": [...], "gate": [...], ...},
  "start_frame": 840          // optional: skip counting before this frame
}
```

`start_frame` is for clips whose opening seconds have an unstable camera
or pre-match handling — auto-counter won't fire any events before
`source_frame >= start_frame`, and the benchmark drops GT events from
that pre-window so they don't show up as artificial misses.

The global `roi_config.json` (one level up) is the seed used the first
time you open a new clip — once you save zones in the dashboard, they
write to the per-clip file here.
