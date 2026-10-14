# Licence for `data/scale/sats_adult.json`

The repository code is MIT licensed (see `/LICENSE`). **This file is not.**

`sats_adult.json` reproduces the adult triage chart, the TEWS table, the priority
target times and the additional-investigation prompts from:

> *The South African Triage Scale (SATS) Training Manual 2012.* Western Cape
> Government Health / Emergency Medicine Society of South Africa (EMSSA).
> https://emssa.org.za/wp-content/uploads/2011/04/SATS-Manual-A5-LR-spreads.pdf

The SATS chart is distributed for non-commercial use with attribution and
share-alike terms (CC BY-NC-SA, as recorded in our research notes). The 2012
manual itself has no licence statement, so **check the current terms with EMSSA
before any commercial deployment**. Any changes to this file must be shared on
the same terms.

What we changed: we re-typed the chart into machine-readable JSON, added stable
IDs for each discriminator, and added a page reference to every value. We did not
change any threshold. Site-specific additions (recheck intervals, SpO2 threshold
for adults, TEWS-rise watch) live in `local_policy.json`. That file is MIT and
marked NEEDS CLINICAL SIGN-OFF.

Scope: adults only (older than 12 years or taller than 150 cm). The paediatric
TEWS is not implemented. The scorer refuses to score younger patients and sends
them to the nurse.
