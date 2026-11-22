# Example X-rays

Put 8-10 images from the NIH ChestX-ray14 **test** split here (NIH data is CC0) and list them in
`manifest.json`:

```json
[
  { "file": "00000013_005.png", "label": "Cardiomegaly" },
  { "file": "00000032_001.png", "label": "Effusion" }
]
```

An empty manifest hides the gallery. Pick images from `test_list.txt` only, never from the
training or validation data, and avoid files over about 1 MB.
