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
    card_html = ''
    for c in cards:
        e = c['event']
        card_html += f"""
  <section class="card">
    <h2>{c['name']}</h2>
    <p class="blurb">{c['blurb']}</p>
    <table>
      <tr><td>Masses</td><td>{e['mass_1_source']:.1f} + {e['mass_2_source']:.1f} Suns</td></tr>
      <tr><td>Distance*</td><td>~{e['luminosity_distance'] * 3.26e-3:.1f} billion light-years</td></tr>
      <tr><td>Date (UTC)</td><td>20{c['name'][2:4]}-{c['name'][4:6]}-{c['name'][6:8]}</td></tr>
    </table>
    <img src="{embed(c['png'], 'image/png')}" alt="Spectrogram of {c['name']}">
    <p><b>Real speed:</b><br><audio controls src="{embed(c['wav'], 'audio/wav')}"></audio></p>
    <p><b>4&times; faster (two octaves higher, easier on laptop speakers):</b><br>
       <audio controls src="{embed(c['wav_fast'], 'audio/wav')}"></audio></p>
  </section>"""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Hear a black hole merger: LIGO audio gallery</title>
<style>
  body {{ font-family: system-ui, sans-serif; max-width: 960px; margin: 2em auto; padding: 0 1em;
         line-height: 1.5; color: #222; }}
  h1 {{ margin-bottom: 0.2em; }}
  .card {{ border: 1px solid #ddd; border-radius: 10px; padding: 1em 1.5em; margin: 1.5em 0; }}
  .blurb {{ font-style: italic; }}
  img {{ max-width: 100%; }}
  audio {{ width: 100%; max-width: 420px; }}
  table td {{ padding: 0 1em 0 0; }}
  .note {{ background: #f4f6fb; border-radius: 10px; padding: 0.8em 1.2em; }}
  footer {{ font-size: 0.9em; color: #555; margin-top: 3em; }}
</style>
</head>
<body>
<h1>Hear a black hole merger</h1>
<p>Real data from the LIGO Livingston detector, turned into sound. Use headphones if you can:
the chirps are deep, and laptop speakers struggle below about 100&nbsp;Hz.</p>

<div class="note">
<h3>Why does the pitch rise?</h3>
<p>Two black holes orbiting each other send out gravitational waves, ripples in space itself.
The waves go up and down <b>twice per orbit</b>, so their pitch is twice the orbit rate.
The waves carry energy away, so the black holes sink closer together. Closer orbits are
faster orbits (just as Mercury goes around the Sun faster than Earth), so the pitch goes up.
Faster orbits lose energy even faster, so it runs away: the pitch sweeps upward more and more
quickly until the two black holes touch and merge. That rising "whoop" is called a <b>chirp</b>.</p>
<p><b>Mass sets the sound.</b> Heavy black holes are big, so they touch while still orbiting
fairly slowly: the chirp ends at a low pitch and is over almost instantly (GW190521).
Light ones can get much closer before touching, so they orbit faster and spend longer
in LIGO's hearing range: a longer chirp that climbs higher (GW190814).</p>
<p>The white dashed line on each picture is the textbook prediction, worked out from Einstein's
equations in their simplest form, using the masses LIGO measured. It's not fitted to the
picture, yet it follows the bright track. It's not perfect: it is the simplest version of the
formula (it's least accurate for very unequal masses, like GW190814), it fails in the last instant
when the two objects touch, and GWOSC's merger times are rounded to 0.1&nbsp;s, so the curve can sit
a few hundredths of a second off.</p>
<p><i>What you hear is mostly noise:</i> the data were "whitened" (noise at every pitch made
equally loud) and filtered to 25&ndash;400&nbsp;Hz. Listen for the short upward "whoop" about
two seconds in, over the hiss. For GW190521 it's only a low thump.</p>
</div>

<h2>All three side by side</h2>
<img src="{embed(comparison_png, 'image/png')}" alt="Three spectrograms side by side">
{card_html}

<footer>
<p><b>Data:</b> Gravitational Wave Open Science Center (GWOSC), <a href="https://gwosc.org">gwosc.org</a>:
32&nbsp;s of LIGO Livingston strain data around each event, with masses, distances and merger times
from the GWOSC event catalog (GWTC-1 and GWTC-2.1).
*Distances are "luminosity distances", worked out from how faint the signal is. For far-away
events this is larger than the distance the waves actually travelled, because the Universe
expanded while they were on their way. This research has made use of data or software
obtained from GWOSC, a service of the LIGO Scientific Collaboration, the Virgo Collaboration,
and KAGRA.</p>
<p>Made with gwpy, matplotlib and scipy for CS85. Code: <code>project19/make_gallery.py</code>.</p>
</footer>
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
