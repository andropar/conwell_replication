"""Drive the LAION-fMRI download for all 5 subjects.

Run via screen so the long-running fetch survives a disconnect:

    export LAION_FMRI_ROOT=/path/to/laion_fmri
    mkdir -p "$LAION_FMRI_ROOT/.laion_fmri"
    screen -S laion_dl -L -Logfile "$LAION_FMRI_ROOT/.laion_fmri/download.log" \
        python scripts/download_laion_fmri.py

Filter rationale: extension in {nii.gz, tsv, json} keeps betas, trial TSVs,
noise ceilings, and metadata sidecars while skipping the ~1.2 GB-per-session
GLMsingle .h5 model files and the figure PNGs.
"""

import sys
import time

from laion_fmri.config import get_data_dir
from laion_fmri.download import download

SUBJECTS = ("sub-01", "sub-03", "sub-05", "sub-06", "sub-07")
EXTENSIONS = ["nii.gz", "tsv", "json"]
# ses-31..34 had public access revoked upstream — listing still returns them
# but HeadObject/GetObject 403s. Restrict to ses-01..30 plus subject-level
# summaries ("averages" sentinel keeps no-ses files like noise ceilings).
SESSIONS = [f"ses-{i:02d}" for i in range(1, 31)] + ["averages"]
N_JOBS = 8


def main():
    data_dir = get_data_dir()
    print(f"[{time.strftime('%H:%M:%S')}] data_dir = {data_dir}")
    print(f"[{time.strftime('%H:%M:%S')}] subjects = {list(SUBJECTS)}")
    print(f"[{time.strftime('%H:%M:%S')}] extension filter = {EXTENSIONS}")
    print(f"[{time.strftime('%H:%M:%S')}] ses filter = ses-01..ses-30 + averages")
    print(f"[{time.strftime('%H:%M:%S')}] n_jobs = {N_JOBS}")
    print(flush=True)

    for sub in SUBJECTS:
        t0 = time.time()
        print(f"[{time.strftime('%H:%M:%S')}] >>> {sub} starting", flush=True)
        download(
            subject=sub,
            ses=SESSIONS,
            extension=EXTENSIONS,
            include_stimuli=False,
            n_jobs=N_JOBS,
        )
        dt = time.time() - t0
        print(
            f"[{time.strftime('%H:%M:%S')}] <<< {sub} done in {dt/60:.1f} min",
            flush=True,
        )

    print(f"[{time.strftime('%H:%M:%S')}] all subjects complete", flush=True)


if __name__ == "__main__":
    sys.exit(main())
