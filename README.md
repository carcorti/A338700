# OEIS A338700

**Exhaustive search establishing a(9) > 5·10¹⁴ for OEIS sequence A338700**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Result

[OEIS A338700](https://oeis.org/A338700) records the smallest prime *p* such
that *p* and the *n* − 1 following primes are all Sophie Germain primes.

This repository presents an exhaustive search of the interval
[1 372 604 395 439, 5 · 10¹⁴] that found no run of 9 consecutive Sophie Germain
primes in this interval, establishing the verified lower bound:

> **a(9) > 5 · 10¹⁴**

Full methodology and results are described in the accompanying paper:

> Carlo Corti, *A computer-verified lower bound for the ninth term of OEIS
> sequence A338700*, 2026. [`paper/A338700.pdf`](paper/A338700.pdf)

---

## Repository structure

```
.
├── CITATION.cff
├── LICENSE
├── README.md
├── data/
│   └── b338700.txt             # OEIS b-file, terms a(1)–a(8)
├── logs/
│   └── a338700_campaign.log    # selected campaign log (10 sessions)
├── paper/
│   ├── A338700.tex           # LaTeX source
│   └── A338700.pdf           # compiled paper
└── src/
    ├── Makefile
    └── a338700.c            # production source (C17, OpenMP)
```

---

## Build

Requires GCC ≥ 13 with OpenMP and C17 support.

```bash
cd src
make release          # -O2 -march=native (default)
make release-znver4   # -O2 -march=znver4 -mtune=znver4  (Ryzen 9 7940HS)
make release-lto      # -O3 -march=native -flto
```

---

## Validation

Reproduces all known terms a(1)–a(8) exactly before any campaign.

```bash
make validate-small   # [0, 1e7],  verifies a(1)–a(5)  (seconds)
make validate-medium  # [0, 5e8],  verifies a(1)–a(6)  (seconds)
make validate-full    # [0, 2e12], verifies a(1)–a(8)  (hours)
```

---

## Running a campaign

```bash
./a338700 --low 1372604395439 --high 500000000000000 \
          --target 9 --max-run 16 --threads 16 --mode double
```

To resume from a checkpoint:

```bash
./a338700 --resume <statefile> --threads 16
```

---

## Campaign summary

| Parameter                   | Value                         |
| --------------------------- | ----------------------------- |
| Search window               | [1 372 604 395 439, 5 · 10¹⁴] |
| Total wall-clock time       | 52 h 49 min (10 sessions)     |
| Primes scanned              | 15 186 821 470 495            |
| Sophie Germain primes found | 598 620 723 232               |
| Longest run in window       | length 8, starting at a(8)    |
| Result                      | a(9) not found                |

---

## Reproducibility

The computation is fully reproducible using the provided source code,
Makefile, and parameters described above.

The repository includes:

- source code;
- data file (`data/b338700.txt`);
- selected campaign logs.

All validation steps are described and can be executed as shown.

---

## Known terms

| n   | a(n)              |
| --- | ----------------- |
| 1   | 2                 |
| 2   | 2                 |
| 3   | 2                 |
| 4   | 1 433 849         |
| 5   | 9 816 899         |
| 6   | 445 480 319       |
| 7   | 298 098 924 131   |
| 8   | 1 372 604 395 439 |

---

## Hardware

AMD Ryzen 9 7940HS · 16 threads · 64 GB DDR5 · Linux Mint 22.3

---

## Citation

If you use this code or results, please cite via `CITATION.cff` or:

```
Carlo Corti, "A computer-verified lower bound for the ninth term of
OEIS sequence A338700", 2026.

GitHub: https://github.com/carcorti/A338700  
DOI: 10.5281/zenodo.20039591
```

---

## License

MIT — see [LICENSE](LICENSE).
