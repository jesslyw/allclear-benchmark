"""
Intersection of emrdm_samples.json and vpint2_samples.json.

Samples eligible for both filters. Entries are taken from the EMRDM set
because those carry the S1 frames (VPint2 entries store s1 = []), and the
S2 frames are identical in both.

Output: intersection_samples.json (same format as the input sample files)
"""

import json
from pathlib import Path

EMRDM = Path(__file__).parent / "emrdm_samples.json"
VPINT2 = Path(__file__).parent / "vpint2_samples.json"
OUTPUT = Path(__file__).parent / "intersection_samples.json"


def main():
    with open(EMRDM) as f:
        emrdm = json.load(f)
    with open(VPINT2) as f:
        vpint2 = json.load(f)

    shared = [k for k in emrdm if k in vpint2]
    out = {k: emrdm[k] for k in shared}

    with open(OUTPUT, "w") as f:
        json.dump(out, f, indent=2)

    print(f"EMRDM samples       : {len(emrdm)}")
    print(f"VPint2 samples      : {len(vpint2)}")
    print(f"Intersection samples: {len(out)}")
    print(f"Written to: {OUTPUT}")


if __name__ == "__main__":
    main()
