# Private per-finding read-outs from the column-inversion probes (2026-10-23)

Probe score = base - (2*AUC_col - 1)/12, so AUC_col = .5 + 6*(base - probe). Displayed scores are 3-decimal, so each AUC carries about +-.003.

| finding | base | public AUC | private AUC | private - public |
|---|---|---|---|---|
| Synovitis | F1 | 0.914 | 0.860 | -0.054 |
| PF OA | F1 | 0.938 | 0.926 | -0.012 |
| Lateral OA | F1 | 0.920 | 0.944 | +0.024 |
| Contusion | G5 | 0.938 | 0.914 | -0.024 |
| Lateral Meniscus | G5 | 0.968 | 0.950 | -0.018 |

Sum over these five findings: -0.084, i.e. -0.0070 of the macro drop; the blends dropped .011-.012 macro (F1 .960 -> .948, G5 .961 -> .950), so the other seven findings account for about -0.008 each on average.

Reading: the private split is markedly harder on Synovitis (-.05) and Contusion (-.02), a little harder on Lateral Meniscus and PF OA, and easier on Lateral OA (+.02). Synovitis is the finding whose expert definition departs furthest from the report labels (I-183), so its private AUC of ~.86 against ~.91 public is the clearest sign that the public/private gap was a label-definition gap, not a modelling one.
