# Radar example

`example_240km.png` is the unchanged sample previously stored at the repository
root as `dpsri_240km_2026051910300000dBR.dpsri.png`. It is an old 240 km image,
not current weather or a 70 km training input.

Use an explicit image path with `scripts/inspect_legacy_png.py` for the legacy
colour-decoder diagnostic. Source-aware model decoding is tested separately in
`tests/test_radar_codec.py`.
