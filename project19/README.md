# Project 19: Hear a black hole merger (LIGO audio gallery)

**Question:** What did the first black-hole collision ever detected (2015) actually "sound" like in LIGO's detectors?

This folder is the Thursday prototype: an audio gallery with spectrograms, playable chirps and a short explainer of why the pitch rises, for three events with very different masses.

| Event | Masses (Suns) | What it sounds like |
|---|---|---|
| GW190521 | 98.4 + 57.2 | Heaviest: a short, deep thump |
| GW150914 | 35.6 + 30.6 | The first detection (Sept 14, 2015): a quick upward "whoop" |
| GW190814 | 23.3 + 2.6 | Lightest: a longer chirp that climbs higher |

## How to run it

From the top folder of the repo:

```bash
python3 project19/make_gallery.py
```

It downloads about 1 MB of data per event (cached in `data/`, so only once) and takes about 15 seconds.

## What it makes

- `index.html`: the gallery. Pictures and sounds are built into the page, so this one file works by itself. GitHub shows HTML as code, so download it ("Download raw file") and open it in a web browser.
- `audio/GWxxxxxx.wav`: 3 seconds of real data around each merger, at real speed. The merger is 2 seconds in.
- `audio/GWxxxxxx_x4.wav`: the same sound played 4 times faster (two octaves higher), so it's easier to hear on laptop speakers.
- `../plots/19_GWxxxxxx_spectrogram.png`: each chirp, with the textbook prediction drawn on top.
- `../plots/19_chirp_comparison.png`: all three side by side.

## How it works

1. **Data:** 32 seconds of LIGO Livingston strain data around each event, plus its masses and merger time, from GWOSC's event API.
2. **Spectrogram:** a whitened Q-transform from gwpy, which shows how loud each pitch is over time.
3. **Prediction:** the simplest (leading-order) form of Einstein's equations gives the chirp's frequency at time τ before merger: f = (1/π) · (5 / 256τ)^(3/8) · (G·Mc/c³)^(−5/8), where Mc is the "chirp mass" LIGO measured. It isn't fitted to the data. GWOSC rounds merger times to 0.1 s, so the curve can sit a few hundredths of a second off.
4. **Sound:** LIGO records 4096 samples per second, which is already an audio format. We whiten the data, keep 25–400 Hz and save it as a WAV file.

## Next step (stretch goal)

Repeat for GW250114, the loudest event so far (download about 100 MB): add it to `EVENTS` in `make_gallery.py` once its data are on GWOSC.

## Data source

Gravitational Wave Open Science Center (GWOSC), https://gwosc.org. Event catalogs GWTC-1 and GWTC-2.1.

"This research has made use of data or software obtained from the Gravitational Wave Open Science Center (gwosc.org), a service of the LIGO Scientific Collaboration, the Virgo Collaboration, and KAGRA."
