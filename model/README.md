# Sesta sottomissione — ensemble RF-DETR-Medium a 3 membri

## Cosa spedisce

`model/ensemble/` contiene i tre membri scelti misurando **tutti i 63
sottoinsiemi** dei sei allenati in questo round, non ereditandone la cardinalità
dal round precedente:

| file | membro interno | pretrain esterno |
|---|---|---|
| `clevis_sarrarp.pth` | `clevis14_sarrarp` | SAR-RARP50 wrist (1 classe) |
| `clevis_mixA.pth` | `clevis14_mixed` | EndoVis 2018 + SAR-RARP50 (1 classe) |
| `clevis_mixC.pth` | `clevis14_mix2c` | wrist + strumento intero (2 classi) |

`model/whole_instrument/` è spedito ma **mai caricato**: contiene detector dello
strumento INTERO, un oggetto diverso. `inference.py` glob-a solo
`model/ensemble/`, apposta.

## Il numero

Diagonale 7-fold leave-2-out sulla GT cat1, mAP@[.5:.05:.95], class-aware:

| configurazione | senza TTA | con flip TTA |
|---|---:|---:|
| tutti e sei | 0.6413 | 0.6424 |
| **questi tre** | 0.6392 | **0.6414** |
| n=4 migliore | 0.6404 | 0.6407 |
| membro singolo migliore | 0.6288 | — |

Sopra i tre membri la curva è **piatta**: da n=3 a n=6 si guadagnano 0.0010 con
TTA, contro un errore standard sulla media dei 7 fold di circa 0.022. Il n=4
finisce perfino sotto il n=3 pur contenendone due membri — un'inversione che da
sola dimostra che si sta leggendo rumore. Spediamo tre membri perché il quarto,
il quinto e il sesto costano inferenza reale e non comprano accuratezza
misurabile.

### ⚠ Il numero sopra è una STIMA, non una misura di questi pesi

Il round ha attraversato **tre corpus diversi**, perché `stage_9` scrive
`train: .../clevis_class_det-corpus/images` — la directory viva, che cambia a
ogni ricostruzione:

| stadio | corpus | cosa ne è uscito |
|---|---:|---|
| k-fold (stage 7, 26/08) | 9.814 frame / 21.452 box | la diagonale **0,6414** citata sopra |
| training finale (stage 9, 28/08) | 10.486 frame / 23.033 box | **i `.pth` in questo pacchetto** |
| corpus vivo (29/08) | 12.814 frame / 28.552 box | niente, ancora |

Quindi 0,6414 descrive un modello allenato su 9.814 frame, mentre questi pesi ne
hanno visti 10.486. È la stima migliore disponibile e la direzione è favorevole
(più dati), ma **non è una misura di ciò che viene spedito**. Se il risultato di
leaderboard si discosta parecchio, la prima ipotesi da controllare è questo
disallineamento, non il modello.

Dettagli in `runs/clevis_selftrain/round_s44_tooltip/corpus_PROVENANCE.json`. Il
round successivo deve congelare uno snapshot del corpus prima dello stage 7 e
usare quello anche per lo stage 9.

## Come sono stati ottenuti

Corpus di self-training a 10.486 frame (6.569 revisionati a mano, 3.917 ammessi
dal gate di confidenza), pretrain per membro sul corpus auto-gated unito alle
8.300 review umane del progetto "Auto labeling" permutate in tassonomia cat1,
poi finetune sui 7 video cat1 più il corpus. Le epoche di ciascun membro vengono
dalla mediana dei `best_epochs` per fold del k-fold: 19 / 19 / 14.

## Il filtro post-fusione

Il `Dockerfile` imposta `CLEVIS_MIN_SCORE=0`, `CLEVIS_MAX_PER_FRAME=10`,
`CLEVIS_CROSS_CLASS_IOU=1.0`. Non sono i default di `inference.py`: lo sweep a
96 punti misura 0.3631 con questi contro 0.3258 con i default. La quinta li aveva
persi ed è costato 0.037.


## Verifica del pacchetto (30/08)

Rifatta dopo aver scoperto che il tarball dell'immagine era **troncato**
(1,26 GB invece di 3,33, `gzip -t` in errore) mentre `do_save.sh` dichiarava
successo: lo script aveva `set -e` ma non `set -o pipefail`, e in
`docker save | gzip > file` bash guardava solo l'uscita di gzip.

Ora `do_save.sh` ha `pipefail` e rilegge ciò che ha scritto; `do_test_run.sh`
passa `--gpus` (il commento lo dichiarava, il codice no) e gira su un video vero
invece che sullo stub da 30 byte del template.

Stato verificato:

| artefatto | esito |
|---|---|
| `example-algorithm-*.tar.gz` | 3,33 GB, `gzip -t` **integro** |
| `nmodel.tar.gz` | 0,35 GB, `gzip -t` **integro**, i 3 `.pth` dell'ensemble |
| `do_test_run.sh` su `3_fps1` | 297 frame, **918 box**, identico a `do_run_val.sh` |
| formato output | `type`/`version`/`boxes` corretti, 297/297 frame, z=0,5, p∈[0,1] |

`model/whole_instrument/` **non è più nel tarball caricabile**: `inference.py`
glob-a solo `model/ensemble/*.pth`, quindi erano 250 MB (il 44% del pacchetto)
che il container non apre mai.
