# Media dependency notices and source references

Reviewed 2026-10-12 for Sweep's private Windows desktop installer with av 19.0.1
(cp312-abi3-win_amd64) and vosk 0.3.45. These files supplement notices present in
installed Python distributions. They do not establish complete release compliance.
The Vosk wheel supplied no COPYING/NOTICE; the PyAV wheel supplied its own BSD
license but omitted separate notices for its bundled native libraries.

`inventory.json` records every saved notice's exact source URL, byte count and
SHA-256, plus the 27 DLLs observed in `av.libs` and `vosk`. No DLLs or models are
stored in this directory. The inventory applies to these observed wheels; changing
a wheel requires repeating the inventory and notice review.

## Source and version evidence

PyAV's [v19.0.1 vendor manifest](https://github.com/PyAV-Org/PyAV/blob/v19.0.1/scripts/ffmpeg-latest.json)
pins [pyav-ffmpeg 9.0.2-1](https://github.com/PyAV-Org/pyav-ffmpeg/releases/tag/9.0.2-1).
Its commit is `bac889417e26be29615cb6a3f38318151fed55cd`. The
[dependency recipe](https://github.com/PyAV-Org/pyav-ffmpeg/blob/bac889417e26be29615cb6a3f38318151fed55cd/scripts/pkg.py)
provides dependency source archives and hashes; the
[build script](https://github.com/PyAV-Org/pyav-ffmpeg/blob/bac889417e26be29615cb6a3f38318151fed55cd/scripts/build-ffmpeg.py)
records build options and copying compiler runtime DLLs from the build host.
Observed FFmpeg reports 9.0.2, consistent with that manifest. Dependency versions
below are recipe evidence, not independently reproduced binary provenance.

| Notice directory | Reference version / scope |
| --- | --- |
| PyAV | 19.0.1; license and authors copied from installed wheel |
| vosk | Official vosk-api v0.3.45 COPYING (Apache 2.0) |
| ffmpeg | Official FFmpeg n9.0.2 license overview and GPL/LGPL texts |
| x264 | VideoLAN commit b35605ace3ddf7c1a5d67a2eb553f034aef41d55 COPYING |
| x265 | Multicoreware 4.3 COPYING |
| opus | 1.6.1 COPYING |
| dav1d | 1.5.4 COPYING |
| SVT-AV1 | 4.2.0 LICENSE.md |
| libvpx | 1.17.0 LICENSE and PATENTS |
| libwebp | 1.6.0 COPYING and PATENTS; includes sharpyuv/webpmux family |
| vmaf | 3.2.1 LICENSE |
| libvpl | 2.16.0 LICENSE and third-party-programs.txt |
| lamer | 3.101.0 LICENSE; enabled MP3 library in recipe, potentially embedded |
| png | 1.6.58 LICENSE; recipe dependency, potentially embedded |
| AMF | 1.5.2 LICENSE.txt; enabled header dependency |
| gcc | Official GCC COPYING3 and COPYING.RUNTIME at pinned review commit; DLL compiler versions unverified |
| winpthreads | Official mingw-w64 COPYING at pinned review commit; bundled DLL version unverified |
| iconv | GNU LGPL 2.1 reference text; bundled DLL version and exact source notices unverified |
| zlib | Official 1.3.1 LICENSE reference; bundled DLL version unverified |

## Unresolved distribution boundaries

The FFmpeg runtime reports "LGPL version 3 or later", but its configuration enables
libx264 and libx265. The vendor
[FFmpeg patch](https://github.com/PyAV-Org/pyav-ffmpeg/blob/bac889417e26be29615cb6a3f38318151fed55cd/patches/ffmpeg.patch)
moves those libraries from the GPL list to the version-3 list. The original
[FFmpeg license overview](https://github.com/FFmpeg/FFmpeg/blob/n9.0.2/LICENSE.md)
still classifies their combination under GPL, and their COPYING files retained
here contain GPLv2. The reported LGPL string therefore does not settle the linked
build's licensing. No alternative commercial rights have been established.

This directory is a notice supplement, not a corresponding-source distribution,
written source offer, or determination that Sweep can be publicly redistributed
with these binaries. Before public release, resolve the exact linked licenses,
source/build/patch availability, relinking conditions and any additional notices.
Preserve the dependency recipe and vendor patch when preparing corresponding source.

Exact source versions for MinGW/GCC, iconv and zlib DLLs remain unverified. Vosk's
[build Makefile](https://github.com/alphacep/vosk-api/blob/v0.3.45/src/Makefile)
can statically combine Kaldi, OpenFst, OpenBLAS and related math libraries; the
Windows wheel does not provide a full static dependency provenance/notice bundle.
NV codec header notices and any other embedded dependency notices also remain to
be reconciled with the exact build. These gaps are not repaired by copying generic
license texts.

The separately downloaded small English speech model is not part of this installer
payload; its publisher's model terms and verified setup archive are tracked by the
speech setup implementation. Other Sweep dependencies retain their own collected
notices outside this directory.
