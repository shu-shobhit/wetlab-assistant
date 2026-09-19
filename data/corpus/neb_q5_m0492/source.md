# PCR Using Q5 High-Fidelity 2X Master Mix (NEB #M0492)

Source: https://www.neb.com/en-us/protocols/2012/12/07/protocol-for-q5-high-fidelity-2x-master-mix-m0492
Accessed 6 September 2026. Read through a browser: NEB returns 403 to scripts.

The parts of the page that `parsed.yaml` is derived from, kept verbatim so the
parse can be checked against them. The page's general guidelines, videos and
product links are not reproduced.

## Overview

This protocol describes methods for PCR using Q5 High-Fidelity 2X Master Mix,
which offers high fidelity (~280X higher than Taq), resulting in ultra-low error
rates.

## Protocol: Reaction Setup

Assemble all reaction components on ice. Each component should be gently mixed
before adding to the reaction. The entire reaction should be mixed again to
ensure homogeneous, consistent mixture. Collect all liquid to the bottom of the
tube with a quick centrifuge spin if necessary. Overlay the sample with mineral
oil if using a PCR machine without a heated lid.

Quickly transfer the reactions to a thermocycler preheated to the denaturation
temperature (98°C) and begin thermocycling.

| Component | 25 µl Reaction | 50 µl Reaction | Final Concentration |
|---|---|---|---|
| Q5 High-Fidelity 2X Master Mix | 12.5 µl | 25 µl | 1X |
| 10 µM Forward Primer | 1.25 µl | 2.5 µl | 0.5 µM |
| 10 µM Reverse Primer | 1.25 µl | 2.5 µl | 0.5 µM |
| Template DNA | variable | variable | < 1,000 ng |
| Nuclease-Free Water | to 25 µl | to 50 µl | |

## Thermocycling Conditions for a Routine PCR

| STEP | TEMP | TIME |
|---|---|---|
| Initial Denaturation | 98°C | 30 seconds |
| 25-35 Cycles | 98°C | 5-10 seconds |
| | 50-72°C* | 10-30 seconds |
| | 72°C | 20-30 seconds/kb |
| Final Extension | 72°C | 2 minutes |
| Hold | 4-10°C | |

\* Use of the NEB Tm Calculator is highly recommended.

## Guidelines quoted where parsed.yaml depends on them

**Denaturation.** An initial denaturation of 30 seconds at 98°C is sufficient for
most amplicons from pure DNA templates. During thermocycling, a 5-10 second
denaturation at 98°C is recommended for most templates.

**Annealing.** Typically use a 10-30 second annealing step at 3°C above the Tm of
the lower Tm primer.

**Extension.** The recommended extension temperature is 72°C. Extension times are
generally 20-30 seconds per kb for complex, genomic samples. A final extension of
2 minutes at 72°C is recommended.

**Cycle number.** Generally, 25-35 cycles yield sufficient product. For genomic
amplicons, 30-35 cycles are recommended.

**Primers.** The best results are typically seen when using each primer at a
final concentration of 0.5 µM in the reaction.
