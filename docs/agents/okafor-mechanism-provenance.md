# Okafor mechanism provenance check

Checked on 2026-10-09. This was a bounded, read-only source check. It did not
change the mechanism, an environment, a server, a model, or a dataset. No raw
mechanism was downloaded or published.

## Result

The paper attribution and the local file identity are supported. Equivalence
to an independently obtained original mechanism and redistribution rights are
still unverified. Keep the existing private-asset boundary.

| Claim | Evidence | Status |
| --- | --- | --- |
| The Fuel study specifies Okafor chemistry with 59 species and 356 reactions. | The authors' manuscript states these counts and cites the 2018 paper as reference 34. | Supported for the study's stated choice. |
| The recovered YAML has those counts. | A fresh Cantera 3.2.0 load returned 59 species and 356 reactions. Its SHA-256 matches the recorded study copy. | Supported for this exact local file. |
| The recovered YAML is chemically identical to the publisher's original release. | No independently obtained, authoritative mechanism package was available for comparison in this check. | Not established. |
| The raw YAML may be redistributed publicly. | No mechanism-specific permission or license was verified. | Not established. |

The first claim comes from [the Fuel manuscript, introduction and reference
34](https://arxiv.org/html/2507.08277v2). The local checks below support the
second claim. Matching counts alone do not prove matching chemistry.

## Primary-source trail

The original paper is Okafor et al., *Experimental and numerical study of the
laminar burning velocity of CH4-NH3-air premixed flames*, Combustion and Flame
187 (2018), 185-198, DOI
[10.1016/j.combustflame.2017.09.002](https://doi.org/10.1016/j.combustflame.2017.09.002).
The authors' [Tohoku laboratory publication list](https://www.ifs.tohoku.ac.jp/kobayashi/en/papers.html)
independently lists this paper under 2018.

The [author's institutional publication record](https://kyushu-u.elsevierpure.com/en/publications/experimental-and-numerical-study-of-the-laminar-burning-velocity--2/)
provides the abstract. It describes a mechanism based on GRI-Mech 3.0 and
Tian chemistry. Its visible access links point to the DOI and Scopus, not a
mechanism archive. Its metadata identifies The Combustion Institute's 2017
publisher copyright. This is not a mechanism redistribution license.

The [publisher's indexed article preview](https://www.sciencedirect.com/science/article/pii/S0010218017303322)
also exposes the title and abstract in search results. Direct page access,
DOI access, and the publisher API returned tool access errors during this
check. Thus, the full article and its supplement list were not inspected.
No authoritative supplemental package URL, release hash, or explicit mechanism
license was verified. This is a limited access finding, not proof that a
supplement or permission does not exist. No third-party mirror was accepted
as the original release.

## Exact local artifact

The existing ignored file is
`runs/sources/nh3ch4-study/Okafor2018_s59r356.yaml`.

- Size: 85,019 bytes.
- SHA-256: `26a27fb3c19c6000ed46d70947faeaf4813b6161ca7186fb6cc9ad55ede294f0`.
- Fresh read-only load: Cantera 3.2.0; 59 species; 356 reactions.
- Header: `ck2yaml`, Cantera 2.6.0, conversion date 2024-01-10, and input
  names `chem.inp`, `therm.dat`, and `trans.dat`.

The header is metadata inside the local file. It is not an independent
conversion record or proof of original-file lineage. The earlier checks of
matching server copies are in the
[source and runtime contract](flame-source-and-runtime-contract.md#mechanism).

## What would close the gap?

Obtain the original package through the publisher or an author-controlled
source. Record the package URL, retrieval date, version, hashes, and applicable
permission. Then compare ordered species, thermodynamic and transport data,
reaction equations, rates, units, third-body efficiencies, and duplicate
flags. A conversion can change file bytes without changing chemistry; a
semantic comparison is therefore separate from a hash comparison.

Until then, use the precise description: **the recovered user-provided
59-species/356-reaction mechanism, attributed to Okafor 2018 and pinned by
SHA-256**. Do not describe it as a publisher-verified release. Do not publish
the YAML in Git, Pages, attachments, or image layers.
