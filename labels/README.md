# labels/

Ground-truth event logs for each match clip, recorded by hand in
`python app.py --replay <clip> --label --alliance <red|blue>` mode.

One JSON per clip: `<clipname>.json`. Each file holds:

```json
{
  "video": "<clipname>",
  "counts": {
    "red":  {"classified": N, "overflow": N},
    "blue": {"classified": N, "overflow": N}
  },
  "events": [
    {"seq": 1, "frame": 458, "t": 1778640141.8,
     "alliance": "red", "line": "classified"},
    ...
  ]
}
```

`frame` is the source-clip frame index at the moment the human clicked,
which is what the benchmark matches auto-events against (within ±150
frames by default, see `/api/benchmark` in `app.py`).

`*.bak` files are pre-edit backups (gitignored).
