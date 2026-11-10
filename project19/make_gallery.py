"""
Project 19: Hear a black hole merger -- the audio gallery (Thursday prototype).

For three black-hole mergers seen by LIGO, this script:
  1. downloads 32 seconds of LIGO-Livingston data around each merger (about 1 MB each),
  2. makes a spectrogram (a picture of pitch vs. time) of the "chirp",
  3. draws the textbook prediction for how the pitch should rise on top of it,
  4. saves each chirp as a WAV sound file you can play,
  5. builds project19/index.html: one page with all the pictures, sounds and an explainer.

Run it from the top folder of the repo:   python3 project19/make_gallery.py

Data source: Gravitational Wave Open Science Center (GWOSC), https://gwosc.org
  Event list and parameters: https://gwosc.org/eventapi/
  "This research has made use of data or software obtained from the Gravitational Wave
  Open Science Center (gwosc.org), a service of the LIGO Scientific Collaboration, the
  Virgo Collaboration, and KAGRA."
"""
import base64
import json
import os

import matplotlib
matplotlib.use('Agg')                      # draw to files, no screen needed
import matplotlib.pyplot as plt
import numpy as np
import requests
from gwpy.timeseries import TimeSeries
from scipy.io import wavfile

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
# The three events, from heaviest to lightest. Heavier pairs merge at lower pitch
# and spend less time in LIGO's hearing range, so their chirp is short and deep.
# Each event points at its entry in a GWOSC catalog (catalog name / event / version).
# Those entries list the event's masses, merger time and small 32-second data files.
EVENTS = {
    'GW190521': ('GWTC-2.1-confident/GW190521/v4',
                 'Heaviest: ~150 Suns in total. Over in a blink, deep "thump".'),
    'GW150914': ('GWTC-1-confident/GW150914/v3',
                 'The first ever detection (Sept 14, 2015). ~65 Suns in total.'),
    'GW190814': ('GWTC-2.1-confident/GW190814/v3',
                 'Lightest: a 23-Sun black hole eating a 2.6-Sun mystery object. Long, rising chirp.'),
}

# What each spectrogram shows, written after looking at the plots.
RESULTS = {
    'GW190521': 'These black holes are so big that they touch while still orbiting slowly. '
                'The signal ends below 100 Hz and is over almost instantly, so the prediction '
                'line is very short too.',
    'GW150914': 'The signal sweeps from about 35 Hz to about 300 Hz in the last tenth of a second. '
                'The dashed prediction follows the shape of the real chirp, sitting a few '
                'hundredths of a second early because GWOSC rounds merger times to 0.1 s.',
    'GW190814': 'The chirp is visible for the whole second, rising from about 45 Hz to about '
                '150 Hz. The prediction runs a little above the data: the simple formula is least '
                'accurate for very unequal masses like 23 and 2.6 Suns.',
}

AUTHOR = ''                                # put your name here, e.g. AUTHOR = 'Your Name'

DATA_DIR = 'data'                          # downloads are cached here (git-ignored)
PLOT_DIR = 'plots'                         # every figure goes here
OUT_DIR = 'project19'                      # WAV files and the HTML gallery go here
AUDIO_DIR = os.path.join(OUT_DIR, 'audio')

# gwosc.org's certificate is signed by "emSign Root CA - G1", which is in the operating
# system's list but not in Python's certifi list. So we tell requests to use the system
# list. Certificate checking stays ON; we only choose which trusted list to use.
SYSTEM_CERTS = '/etc/ssl/certs/ca-certificates.crt'
VERIFY = SYSTEM_CERTS if os.path.exists(SYSTEM_CERTS) else True

# Physical constants (SI units)
G = 6.674e-11        # gravitational constant
C = 2.998e8          # speed of light, m/s
M_SUN = 1.989e30     # mass of the Sun, kg


# ---------------------------------------------------------------------------
# Step 1: download (once) and load the data
# ---------------------------------------------------------------------------
def download(url, filename):
    """Download url into data/filename, unless we already have it."""
    os.makedirs(DATA_DIR, exist_ok=True)
    path = os.path.join(DATA_DIR, filename)
    if not os.path.exists(path):
        print(f'  downloading {url}')
        r = requests.get(url, verify=VERIFY, timeout=120)
        r.raise_for_status()                       # stop with a clear error if it failed
        with open(path, 'wb') as f:
            f.write(r.content)
    return path


def get_event_info(name, catalog_entry):
    """Ask GWOSC's event API for the event's merger time, masses and data-file links."""
    url = f'https://gwosc.org/eventapi/json/{catalog_entry}/'
    path = download(url, f'{name}_event.json')
    with open(path) as f:
        info = json.load(f)
    return list(info['events'].values())[0]      # the entry holds exactly one event


def get_livingston_data(event):
    """Download the 32-second, 4096-samples-per-second LIGO Livingston (L1) file."""
    files = [s for s in event['strain']
             if s['detector'] == 'L1' and s['format'] == 'hdf5'
             and s['sampling_rate'] == 4096 and s['duration'] == 32]
    if not files:
        raise RuntimeError(f"No 32-second L1 file listed for {event['commonName']}")
    url = files[0]['url']
    path = download(url, os.path.basename(url))
    return TimeSeries.read(path, format='hdf5.gwosc')


# ---------------------------------------------------------------------------
# Step 2: the textbook prediction for the chirp
# ---------------------------------------------------------------------------
def predicted_frequency(seconds_before_merger, chirp_mass_suns):
    """
    Gravitational-wave frequency (Hz) a given time before merger, from the simplest
    ("Newtonian", leading-order) version of Einstein's equations:

        f = (1/pi) * (5 / (256 * tau))^(3/8) * (G * Mc / c^3)^(-5/8)

    tau = seconds left before merger, Mc = "chirp mass", a combination of the two masses
    that sets how fast the pitch climbs. It's only an approximation, and it breaks down
    in the last few milliseconds when the two objects actually touch.
    """
    mc_seconds = G * chirp_mass_suns * M_SUN / C**3          # chirp mass, in seconds
    return (1 / np.pi) * (5 / (256 * seconds_before_merger))**(3 / 8) * mc_seconds**(-5 / 8)


# ---------------------------------------------------------------------------
# Step 3: spectrogram with the prediction drawn on top
# ---------------------------------------------------------------------------
WINDOW = (-1.0, 0.1)      # seconds relative to merger, same for every event so they compare fairly
FREQ_RANGE = (20, 500)    # Hz, the band where LIGO hears best


def detector_chirp_mass(event):
    """
    LIGO measures the "redshifted" chirp mass: the Universe's expansion stretches the signal
    on its way to us, so we use the detector-frame mass = source mass x (1 + redshift).
    """
    return event['chirp_mass_source'] * (1 + event['redshift'])


def draw_spectrogram(ax, data, event, show_prediction=True):
    """Draw a whitened time-frequency map ("Q-transform") of the merger on ax."""
    gps = event['GPS']        # GWOSC's merger time, rounded to 0.1 s
    q = data.q_transform(outseg=(gps + WINDOW[0], gps + WINDOW[1]), frange=FREQ_RANGE)
    times = q.times.value - gps                    # seconds relative to the merger
    freqs = q.frequencies.value
    image = ax.pcolormesh(times, freqs, q.value.T, cmap='viridis', vmin=0, vmax=25,
                          shading='auto', rasterized=True)
    ax.set_yscale('log')
    ax.set_ylim(*FREQ_RANGE)
    ax.set_xlim(*WINDOW)
    ax.set_xlabel('Seconds from merger (catalog time, rounded to 0.1 s)')
    ax.set_ylabel('Frequency (Hz)')

    if show_prediction:
        mc_detector = detector_chirp_mass(event)
        tau = np.linspace(0.005, -WINDOW[0], 500)          # seconds before merger
        f = predicted_frequency(tau, mc_detector)
        keep = f < 300                                     # where the simple formula still works
        ax.plot(-tau[keep], f[keep], color='white', ls='--', lw=1.5,
                label=f'Textbook prediction (chirp mass {mc_detector:.0f} Suns)')
        ax.legend(loc='upper left', fontsize=8, facecolor='black', labelcolor='white')
    return image


def total_mass(event):
    return event['mass_1_source'] + event['mass_2_source']


def make_event_figure(name, data, event):
    fig, ax = plt.subplots(figsize=(8, 4))
    image = draw_spectrogram(ax, data, event)
    fig.colorbar(image, ax=ax, label='Normalized energy')
    ax.set_title(f"{name}: {event['mass_1_source']:.1f} + {event['mass_2_source']:.1f} Suns merging\n"
                 'LIGO Livingston data from GWOSC')
    path = os.path.join(PLOT_DIR, f'19_{name}_spectrogram.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return path


def make_comparison_figure(all_data):
    """All three chirps side by side, with the same axes, heaviest on the left."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), sharey=True)
    for ax, (name, (data, event)) in zip(axes, all_data.items()):
        image = draw_spectrogram(ax, data, event, show_prediction=False)
        ax.set_title(f'{name}\ntotal mass ~{total_mass(event):.0f} Suns')
        ax.set_xlabel('Seconds from merger')       # shorter label, so neighbours don't overlap
    for ax in axes[1:]:
        ax.set_ylabel('')
    fig.colorbar(image, ax=axes, label='Normalized energy', fraction=0.02)
    fig.suptitle('Lighter pairs chirp longer and climb higher (LIGO Livingston, data: GWOSC; '
                 'merger times rounded to 0.1 s)', y=1.04)
    path = os.path.join(PLOT_DIR, '19_chirp_comparison.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Step 4: turn the data into sound
# ---------------------------------------------------------------------------
def make_wav(name, data, event, speedup=1):
    """
    Save 3 seconds around the merger as a WAV file.
    Whitening flattens LIGO's loud low-frequency noise, and the band-pass keeps 25-400 Hz,
    where the chirp is. LIGO's data are already "sound waves": 4096 samples per second.
    speedup=4 plays it 4 times faster, which raises every pitch by two octaves so it's
    easier to hear on laptop speakers. We take a 4 times longer stretch of data for it,
    so the sound still lasts 3 seconds, with the merger at 2 seconds.
    """
    gps = event['GPS']
    clip = data.whiten(4, 2).bandpass(25, 400).crop(gps - 2.0 * speedup, gps + 1.0 * speedup)
    samples = clip.value
    samples = samples / np.max(np.abs(samples))            # loudest point = full volume
    fade = np.linspace(0, 1, len(samples) // 30)           # 0.1 s fade in/out, no clicks
    samples[:len(fade)] *= fade
    samples[-len(fade):] *= fade[::-1]
    rate = int(clip.sample_rate.value) * speedup
    os.makedirs(AUDIO_DIR, exist_ok=True)
    suffix = '' if speedup == 1 else f'_x{speedup}'
    path = os.path.join(AUDIO_DIR, f'{name}{suffix}.wav')
    wavfile.write(path, rate, (samples * 32767 * 0.9).astype(np.int16))
    return path


# ---------------------------------------------------------------------------
# Step 5: the HTML gallery
# ---------------------------------------------------------------------------
def embed(path, mime):
    """Put a file inside the HTML page itself, so the page works on its own."""
    with open(path, 'rb') as f:
        return f'data:{mime};base64,{base64.b64encode(f.read()).decode()}'


def make_html(cards, comparison_png):
    """Build the full project report: one page with the write-up, plots and sounds."""
    # One row per event for the data table.
    data_rows = ''
    for c in cards:
        e = c['event']
        data_rows += f"""
    <tr><td>{c['name']}</td><td>{e['catalog.shortName'].replace('-confident', '')}</td>
        <td>{e['mass_1_source']:.1f} + {e['mass_2_source']:.1f}</td>
        <td>{e['mass_1_source'] + e['mass_2_source']:.0f}</td>
        <td>~{e['luminosity_distance'] * 3.26e-3:.1f}</td>
        <td>20{c['name'][2:4]}-{c['name'][4:6]}-{c['name'][6:8]}</td></tr>"""

    # One card per event: what it shows, its spectrogram and its two sounds.
    card_html = ''
    for c in cards:
        card_html += f"""
  <section class="card">
    <h3>{c['name']}</h3>
    <p class="blurb">{c['blurb']}</p>
    <p>{RESULTS[c['name']]}</p>
    <img src="{embed(c['png'], 'image/png')}" alt="Spectrogram of {c['name']}">
    <div class="players">
      <p><b>Real speed</b><br><audio controls src="{embed(c['wav'], 'audio/wav')}"></audio></p>
      <p><b>4&times; faster</b> (two octaves higher, easier on laptop speakers)<br>
         <audio controls src="{embed(c['wav_fast'], 'audio/wav')}"></audio></p>
    </div>
  </section>"""

    byline = f'{AUTHOR} &middot; ' if AUTHOR else ''
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Hear a black hole merger: LIGO audio gallery</title>
<style>
  body {{ font-family: system-ui, sans-serif; max-width: 960px; margin: 2em auto; padding: 0 1em;
         line-height: 1.55; color: #222; }}
  h1 {{ margin-bottom: 0.1em; }}
  h2 {{ margin-top: 2em; border-bottom: 1px solid #ddd; padding-bottom: 0.2em; }}
  .byline {{ color: #666; margin-top: 0; }}
  .answer {{ background: #f4f6fb; border-radius: 10px; padding: 0.8em 1.2em; }}
  .card {{ border: 1px solid #ddd; border-radius: 10px; padding: 0.5em 1.5em 1em; margin: 1.5em 0; }}
  .blurb {{ font-style: italic; }}
  .players {{ display: flex; flex-wrap: wrap; gap: 0 2em; }}
  .players p {{ flex: 1 1 320px; }}
  img {{ max-width: 100%; }}
  audio {{ width: 100%; max-width: 380px; }}
  table.data {{ border-collapse: collapse; margin: 1em 0; }}
  table.data th, table.data td {{ border-bottom: 1px solid #ddd; padding: 0.3em 1em 0.3em 0; text-align: left; }}
  .formula {{ font-family: Georgia, serif; font-size: 1.1em; text-align: center; margin: 1em 0; }}
  .small {{ font-size: 0.9em; color: #555; }}
</style>
</head>
<body>
<h1>Hear a black hole merger</h1>
<p class="byline">{byline}CS85 Project 19 &middot; Data: LIGO, via the Gravitational Wave Open Science Center</p>

<h2>Question and answer</h2>
<p><b>Question:</b> What did the first black-hole collision ever detected (GW150914, September 14, 2015)
actually "sound" like in LIGO's detectors?</p>
<div class="answer">
<p><b>Answer:</b> A quick upward "whoop". In the last tenth of a second the signal sweeps from about
35&nbsp;Hz to about 300&nbsp;Hz, then stops as the two black holes become one. Comparing three mergers
shows that mass sets the sound: heavier pairs make a shorter, deeper chirp; lighter pairs chirp longer
and climb higher.</p>
</div>
<p>Listen below. Use headphones if you can: the chirps are deep, and laptop speakers struggle below
about 100&nbsp;Hz.</p>

<h2>Why the pitch rises</h2>
<p>Losing energy makes the orbit faster, and a faster orbit loses energy even faster.</p>
<ol>
  <li>Two orbiting black holes send out gravitational waves, ripples in space itself.</li>
  <li>The waves go up and down twice per orbit, so their pitch is twice the orbit rate.</li>
  <li>The waves carry energy away, so the black holes sink closer together.</li>
  <li>Closer orbits are faster orbits (Mercury circles the Sun faster than Earth does), so the pitch goes up.</li>
  <li>Faster orbits lose energy faster, so steps 3 and 4 run away until the black holes touch and merge.
      That rising "whoop" is called a <b>chirp</b>.</li>
</ol>
<p><b>Mass sets the sound.</b> Heavy black holes are big, so they touch while still orbiting slowly:
a short, low chirp. Light ones get much closer before touching, so they orbit faster and spend longer
in LIGO's hearing range: a longer chirp that climbs higher.</p>

<h2>Results</h2>
<p>On the same axes, the 156-Sun pair is a short blob below 100&nbsp;Hz, the 66-Sun pair sweeps up to
about 300&nbsp;Hz in 0.1&nbsp;s, and the 26-Sun pair rises for the whole second shown.</p>
<img src="{embed(comparison_png, 'image/png')}" alt="Three spectrograms side by side">
<p>Each picture below is a spectrogram: time runs left to right, pitch runs up, and brighter means
louder. The white dashed line is the textbook prediction from the masses LIGO measured; it is
<i>not</i> fitted to the picture. In each sound file the merger is 2 seconds in. The sound is mostly
hiss, with the chirp a short "whoop" on top; for GW190521 it is only a low thump.</p>
{card_html}

<h2>Data</h2>
<p>All data come from the <a href="https://gwosc.org">Gravitational Wave Open Science Center (GWOSC)</a>:
32 seconds of LIGO Livingston strain data around each event (about 1&nbsp;MB each, 4096 samples per
second), plus masses, distances and merger times from GWOSC's event catalogs. The three events were
picked to span a wide range of masses.</p>
<table class="data">
  <tr><th>Event</th><th>Catalog</th><th>Masses (Suns)</th><th>Total (Suns)</th>
      <th>Distance* (billion light-years)</th><th>Date (UTC)</th></tr>{data_rows}
</table>
<p class="small">*Luminosity distance, worked out from how faint the signal is. For far-away events it
is larger than the distance the waves travelled, because the Universe expanded on the way.</p>

<h2>Method</h2>
<p>One Python script, <code>project19/make_gallery.py</code>, does every step in about 15 seconds.</p>
<ol>
  <li><b>Download:</b> ask GWOSC's event API for each event's details and data file, and cache them.</li>
  <li><b>Spectrogram:</b> gwpy's Q-transform makes a whitened picture of pitch versus time, from
      1&nbsp;s before to 0.1&nbsp;s after the merger.</li>
  <li><b>Prediction:</b> draw the textbook chirp on top, from LIGO's measured chirp mass M<sub>c</sub>.
      The simplest (leading-order) form of Einstein's equations gives the frequency a time &tau; before
      merger:
      <div class="formula">f = (1/&pi;) &middot; (5 / 256&tau;)<sup>3/8</sup> &middot;
      (G&thinsp;M<sub>c</sub> / c<sup>3</sup>)<sup>&minus;5/8</sup></div>
      LIGO measures the redshifted chirp mass, so the script uses the source chirp mass &times; (1 + redshift).</li>
  <li><b>Sound:</b> LIGO records 4096 samples per second, which is already an audio format. Whiten the
      data (make the noise equally loud at every pitch), keep 25&ndash;400&nbsp;Hz, and save 3&nbsp;s
      around the merger as a WAV file, plus a copy played 4&times; faster.</li>
</ol>

<h2>Limitations</h2>
<ul>
  <li>GWOSC rounds merger times to 0.1&nbsp;s, so "0" on the time axis is approximate and the prediction
      can sit a few hundredths of a second off.</li>
  <li>The prediction is the simplest version of the formula. It fails in the last instant, when the two
      objects touch, and is least accurate for very unequal masses (GW190814).</li>
  <li>Only one detector (LIGO Livingston) is used, and the sounds are dominated by detector noise.</li>
</ul>

<h2>Next steps</h2>
<ul>
  <li>Stretch goal: add GW250114, the loudest event so far (download about 100&nbsp;MB).</li>
  <li>Add LIGO Hanford's data, to show the same chirp arriving at two detectors a few milliseconds apart.</li>
</ul>

<h2>Sources</h2>
<p class="small">Data: Gravitational Wave Open Science Center, <a href="https://gwosc.org">gwosc.org</a>;
event parameters from the GWTC-1 and GWTC-2.1 catalogs via the
<a href="https://gwosc.org/eventapi/">GWOSC event API</a>. "This research has made use of data or
software obtained from the Gravitational Wave Open Science Center (gwosc.org), a service of the LIGO
Scientific Collaboration, the Virgo Collaboration, and KAGRA."<br>
Software: gwpy, matplotlib, numpy, scipy, requests. Code:
<a href="https://github.com/ryannoor41/cs85-astro/blob/claude/funny-gauss-6k7ayb/project19/make_gallery.py">project19/make_gallery.py</a>.</p>
</body>
</html>
"""
    path = os.path.join(OUT_DIR, 'index.html')
    with open(path, 'w') as f:
        f.write(html)
    return path


# ---------------------------------------------------------------------------
# Run everything
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    os.makedirs(PLOT_DIR, exist_ok=True)
    all_data, cards = {}, []
    for name, (catalog_entry, blurb) in EVENTS.items():
        print(f'{name}:')
        event = get_event_info(name, catalog_entry)
        data = get_livingston_data(event)
        all_data[name] = (data, event)
        cards.append({
            'name': name, 'blurb': blurb, 'event': event,
            'png': make_event_figure(name, data, event),
            'wav': make_wav(name, data, event),
            'wav_fast': make_wav(name, data, event, speedup=4),
        })
        print(f"  saved {cards[-1]['png']}, {cards[-1]['wav']}, {cards[-1]['wav_fast']}")

    print('saved', make_comparison_figure(all_data))
    print('saved', make_html(cards, os.path.join(PLOT_DIR, '19_chirp_comparison.png')))
