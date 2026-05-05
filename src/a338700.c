/*
 * a338700.c — v5 search engine for OEIS A338700
 *
 * Sequence: A338700, "Start of the first run of n or more consecutive primes
 * that are all Sophie Germain primes" (p prime, 2p+1 prime).
 *
 * Primary algorithm: segmented single-domain double-sieve.
 *   Bitmap P over odd integers in [lo, hi] encodes primality of p.
 *   Bitmap Q over the same odd integers encodes "2p+1 has no small prime
 *   factor"; equivalently, for each odd base prime r, Q[i] is cleared whenever
 *   p ≡ (r-1)/2 (mod r) except when 2p+1 == r itself (in which case 2p+1 is
 *   prime and must not be filtered out).
 *
 * Fallback algorithm: deterministic Miller–Rabin on 2p+1, using the
 * Jaeschke–Sorenson witness set, proven sufficient for n < 3.317·10^24.
 *
 * v5 changes over v4 (after six-way HPC review: ChatGPT, DeepSeek, Gemini,
 * Grok, Kimi, Qwen; 2026-04-24):
 *   - Word-scan in scan(): replaced bit-by-bit loop with __builtin_ctzll over
 *     64-bit words. ~97% of iterations in v4 were wasted (prime density
 *     ≈ 1/30 at N ~ 10^14); word-scan skips directly to set bits.
 *   - block_summary_t padded to 64 bytes to eliminate real false sharing
 *     across adjacent sbatch[] entries under schedule(dynamic, 1).
 *   - records_t aligned to 64 bytes with padding so thread_rec[] entries
 *     start on fresh cache lines.
 *   - sieve_q() now limits base-prime iteration to r <= sqrt(2*hi+1)
 *     (per-block qlim), not the global sqrt(2*high+1).
 *   - OMP schedule now reads OMP_SCHEDULE via schedule(runtime); Carlo can
 *     benchmark static vs dynamic vs guided without recompiling.
 *   - Corrected comment: A338700 is nondecreasing (a(n+1) >= a(n)), not
 *     strictly increasing.
 *   - Corrected comment: state_checksum is byte-wise 8-lane XOR.
 *
 * Deferred (benchmark-driven, not correctness): per-thread P/Q buffer reuse,
 * software prefetch, Montgomery multiplication, PGO.
 *
 * Build: see Makefile.
 * Author: Sofia (Claude Opus 4.7), 2026-04-24, framework OEIS v5.
 */

#define _POSIX_C_SOURCE 200809L
#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <libgen.h>
#include <limits.h>
#include <math.h>
#include <omp.h>
#include <stdarg.h>
#include <stdbool.h>
#include <stdint.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

/* ---------------------------------------------------------------------------
 * Constants
 * ------------------------------------------------------------------------- */

#define A8                     1372604395439ULL
#define MAX_RUN                64u      /* hard ceiling on reported run length */
#define DEFAULT_HIGH           500000000000000ULL  /* 5·10^14 */
#define DEFAULT_BLOCK_ODDS     (1ULL<<22)          /* ~4.2M odd integers per block;
                                                     512 KB per bitmap, 1 MB per thread */
#define DEFAULT_BATCH_BLOCKS   4096u    /* ~34 GB scanned per batch at default block size */
#define DEFAULT_CHECKPOINT_SEC 600.0    /* 10 minutes */
#define DEFAULT_PROGRESS_SEC   60.0     /* 1 minute */
#define STATE_MAGIC            0x33383741u /* "A338" little-endian */
/* v5: version bumped to 2 because records_t now has 56 bytes of padding for
 * 64-byte alignment, which changes the on-disk layout of state_t. Old v4
 * checkpoints are rejected cleanly ("unknown version") rather than silently
 * loading with wrong offsets. */
#define STATE_VERSION          2u

/* ---------------------------------------------------------------------------
 * Configuration
 * ------------------------------------------------------------------------- */

typedef struct {
    uint64_t low;            /* lower search bound (inclusive) */
    uint64_t high;           /* upper search bound (inclusive) */
    uint64_t block_odds;     /* odd integers per block */
    uint32_t batch_blocks;   /* blocks per batch (bounds per-batch summary memory) */
    unsigned threads;        /* 0 = use OpenMP default */
    unsigned target;         /* primary n we are searching for */
    unsigned max_run;        /* how many a(k) values to report simultaneously */
    bool     mr;             /* true = Miller-Rabin fallback, false = double-sieve */
    double   checkpoint_sec;
    double   progress_sec;
    const char *results_file; /* NULL = no JSONL output */
    const char *state_file;   /* NULL = no checkpointing */
} cfg_t;

/* ---------------------------------------------------------------------------
 * Global records: first-seen start of a run of length k, for k = 1..MAX_RUN
 * ------------------------------------------------------------------------- */

typedef struct {
    uint64_t start[MAX_RUN + 1];
    /* v5: pad to a multiple of 64 bytes so each thread_rec[t] starts on a
     * fresh cache line. MAX_RUN+1 = 65 uint64_t = 520 bytes, which already
     * crosses 8 full lines; the final line is only 8 bytes used. Padding to
     * 576 bytes makes the array stride a full 9 cache lines and makes the
     * layout explicit. Cost: 56 bytes × nthreads ≈ 896 bytes at 16 threads. */
    uint8_t _pad[64 - ((MAX_RUN + 1) * 8) % 64];
} __attribute__((aligned(64))) records_t;

static inline void rec_init(records_t *r) {
    memset(r, 0, sizeof(*r));
}

/* Register that a run of length `len` starts at `start`. We keep only the
 * smallest `start` ever seen for each length k <= len (since runs of length
 * len are also runs of length k for all k <= len). */
static inline void rec_update(records_t *r, uint32_t len, uint64_t start, unsigned maxrun) {
    if (len > maxrun) len = maxrun;
    for (uint32_t k = 1; k <= len; k++)
        if (!r->start[k] || start < r->start[k])
            r->start[k] = start;
}

static inline void rec_merge(records_t *dst, const records_t *src, unsigned maxrun) {
    for (unsigned k = 1; k <= maxrun; k++)
        if (src->start[k] && (!dst->start[k] || src->start[k] < dst->start[k]))
            dst->start[k] = src->start[k];
}

/* ---------------------------------------------------------------------------
 * Reduced block summary (~40 bytes).
 * Used for cross-block reconciliation only. Internal runs are accumulated
 * into per-thread records_t and merged at end of run/at each checkpoint.
 * ------------------------------------------------------------------------- */

/* v5: padded to 64 bytes and aligned to eliminate false sharing across
 * adjacent sbatch[] entries. With schedule(dynamic, 1), two threads can
 * write to sbatch[k] and sbatch[k+1] concurrently; if both fit in the same
 * cache line, each write invalidates the other's L1 copy. Padding to 64 B
 * guarantees each summary occupies exactly one cache line. */
typedef struct {
    uint64_t first_prime;   /* smallest prime in block, or 0 if none */
    uint64_t prefix_start;  /* start of the SG-run at the beginning of the block */
    uint64_t suffix_start;  /* start of the SG-run at the end of the block */
    uint32_t num_primes;    /* total primes in block */
    uint16_t prefix_len;    /* length of SG-run starting at first_prime */
    uint16_t suffix_len;    /* length of SG-run ending at last prime */
    uint8_t  all_sg;        /* 1 iff every prime in this block is Sophie Germain */
    uint8_t  _pad[64 - 33]; /* pad to full cache line (total 64 bytes) */
} __attribute__((aligned(64))) block_summary_t;

/* ---------------------------------------------------------------------------
 * Small utilities
 * ------------------------------------------------------------------------- */

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + 1e-9 * ts.tv_nsec;
}

/* Integer square root via bisection. Works for all n in uint64_t range. */
static uint64_t isqrt64(uint64_t n) {
    uint64_t lo = 0, hi = 1ULL << 32;
    while (lo + 1 < hi) {
        uint64_t m = lo + (hi - lo) / 2;
        if (m <= n / m) lo = m;
        else            hi = m;
    }
    return lo;
}

static inline uint64_t ceildiv(uint64_t a, uint64_t b) { return a / b + (a % b != 0); }
static inline uint64_t oddceil(uint64_t x)             { return (x & 1) ? x : x + 1; }
/* Largest odd integer <= x. For x = 0 returns 0 (and oddcount handles the
 * empty-range case). ChatGPT's review spotted that the v2 implementation
 * `x | 1ULL` returned the odd *ceil*, not floor, wasting at most one bit in
 * oddcount when hi is even. Benign in practice (the scan discards p > hi)
 * but semantically wrong. */
static inline uint64_t oddfloor(uint64_t x)            { return (x & 1) ? x : (x ? x - 1 : 0); }

static inline uint64_t oddcount(uint64_t lo, uint64_t hi) {
    uint64_t a = oddceil(lo), b = oddfloor(hi);
    return b < a ? 0 : ((b - a) >> 1) + 1;
}

/* Smallest integer >= low that is congruent to res mod mod.
 * Precondition: res < mod. */
static inline uint64_t first_cong(uint64_t low, uint64_t mod, uint64_t res) {
    uint64_t r = low % mod;
    return low + (res >= r ? res - r : mod - (r - res));
}

static bool parse_u64(const char *s, uint64_t *out) {
    if (!s || !*s) return false;
    /* v4 (ChatGPT §4.8): reject leading '-' explicitly. strtoull() wraps
     * negatives to large positives, giving confusing "too large" errors for
     * a user who typed a minus by mistake. */
    if (s[0] == '-') return false;
    errno = 0;
    char *e = NULL;
    unsigned long long v = strtoull(s, &e, 10);
    if (errno || e == s || *e) return false;
    *out = (uint64_t)v;
    return true;
}

static bool parse_u(const char *s, unsigned *out) {
    uint64_t v;
    if (!parse_u64(s, &v) || v > UINT32_MAX) return false;
    *out = (unsigned)v;
    return true;
}

static bool parse_f(const char *s, double *out) {
    errno = 0;
    char *e = NULL;
    double v = strtod(s, &e);
    if (errno || e == s || *e) return false;
    *out = v;
    return true;
}

/* ---------------------------------------------------------------------------
 * Base primes: ordinary Eratosthenes on [2, lim]. lim is capped at UINT32_MAX
 * but in practice lim ~ sqrt(2*high+1) ~ 3.2·10^7 for high = 5·10^14.
 * ------------------------------------------------------------------------- */

typedef struct {
    uint64_t *p;
    size_t    n;
    uint32_t  limit;
} primes_t;

static bool base_primes(uint64_t lim, primes_t *t) {
    memset(t, 0, sizeof(*t));
    if (lim > UINT32_MAX) return false;
    uint32_t L = (uint32_t)lim;
    if (L < 2) return true;

    /* Bitmap over odd composites in [3, L]. */
    uint64_t bits  = (L >= 3) ? ((L - 3) / 2 + 1) : 0;
    uint64_t words = (bits + 63) / 64;
    uint64_t *comp = calloc(words ? words : 1, 8);
    if (!comp) return false;
#define SET(i) (comp[(i) >> 6] |= 1ULL << ((i) & 63))
#define GET(i) ((comp[(i) >> 6] >> ((i) & 63)) & 1ULL)

    uint32_t root = (uint32_t)isqrt64(L);
    for (uint32_t p = 3; p <= root; p += 2) {
        uint64_t ix = (p - 3) / 2;
        if (!GET(ix))
            for (uint64_t m = (uint64_t)p * p; m <= L; m += 2ULL * p)
                SET((m - 3) / 2);
    }

    size_t cap = (L > 100) ? (size_t)(1.4 * (double)L / log((double)L)) + 100 : 32;
    uint64_t *a = malloc(cap * 8);
    if (!a) { free(comp); return false; }
    size_t n = 0;
    a[n++] = 2;
    for (uint32_t p = 3; p <= L; p += 2) {
        uint64_t ix = (p - 3) / 2;
        if (!GET(ix)) {
            if (n == cap) {
                cap *= 2;
                uint64_t *q = realloc(a, cap * 8);
                if (!q) { free(a); free(comp); return false; }
                a = q;
            }
            a[n++] = p;
        }
    }
#undef SET
#undef GET
    free(comp);
    t->p = a;
    t->n = n;
    t->limit = L;
    return true;
}

static void free_primes(primes_t *t) { free(t->p); memset(t, 0, sizeof(*t)); }

/* ---------------------------------------------------------------------------
 * Simple bitset for segmented sieves.
 * ------------------------------------------------------------------------- */

typedef struct {
    uint64_t *w;
    uint64_t  nbits;
    uint64_t  nwords;
} bits_t;

static bool bits_alloc(bits_t *b, uint64_t n) {
    b->nbits  = n;
    b->nwords = (n + 63) / 64;
    if (posix_memalign((void **)&b->w, 64, (size_t)b->nwords * 8)) return false;
    return true;
}

static void bits_free(bits_t *b) { free(b->w); memset(b, 0, sizeof(*b)); }

static void bits_ones(bits_t *b) {
    memset(b->w, 0xff, (size_t)b->nwords * 8);
    if (b->nbits & 63) b->w[b->nwords - 1] &= ((1ULL << (b->nbits & 63)) - 1);
}

static inline void bits_clr(bits_t *b, uint64_t i) { b->w[i >> 6] &= ~(1ULL << (i & 63)); }
static inline void bits_set(bits_t *b, uint64_t i) { b->w[i >> 6] |=  (1ULL << (i & 63)); }
static inline bool bits_get(const bits_t *b, uint64_t i) { return (b->w[i >> 6] >> (i & 63)) & 1ULL; }

/* ---------------------------------------------------------------------------
 * P-sieve: clear bits for composite odd integers in [oddceil(lo), hi].
 * Bit index i represents the odd integer olo + 2i.
 * ------------------------------------------------------------------------- */

static void sieve_p(uint64_t lo, uint64_t hi, const primes_t *bp, bits_t *P) {
    uint64_t olo = oddceil(lo);
    bits_ones(P);
    if (olo == 1) bits_clr(P, 0);  /* 1 is not prime */

    for (size_t j = 1; j < bp->n; j++) {  /* skip r = 2, we only track odds */
        uint64_t r = bp->p[j];
        if (r > hi / r) break;              /* r^2 > hi: done */
        uint64_t s = r * r;                 /* first odd multiple of r >= r^2 */
        if (s < olo) s = ceildiv(olo, r) * r;
        if (!(s & 1)) s += r;               /* r is odd, so even + odd = odd */
        for (uint64_t m = s; m <= hi; m += 2 * r)
            bits_clr(P, (m - olo) >> 1);
    }
}

/* ---------------------------------------------------------------------------
 * Q-sieve: clear Q[i] iff 2p+1 is divisible by some base prime r, with the
 * single exception 2p+1 == r (in which case 2p+1 itself is prime).
 *
 * Why (r-1)/2? We want p such that 2p+1 ≡ 0 (mod r), i.e. p ≡ -1/2 (mod r).
 * Since 2·((r-1)/2) = r-1 ≡ -1 (mod r), we have p ≡ (r-1)/2 ⇒ 2p+1 ≡ 0.
 *
 * Why the exception? When 2p+1 equals the base prime r (i.e. p = (r-1)/2,
 * and p happens to be prime), the congruence still holds, but 2p+1 is prime
 * and the candidate must survive. Example: r=11, p=5: 2·5+1 = 11 = r is
 * prime; without the guard we would reject the genuine SG prime p=5.
 *
 * Completeness: base primes run up to sqrt(2·high+1), so every composite
 * 2p+1 has at least one factor in the list. After sieve_q, Q[i] implies
 * (p odd and 2p+1 prime or 2p+1 is one of the base primes r).
 * ------------------------------------------------------------------------- */

static void sieve_q(uint64_t lo, uint64_t hi, const primes_t *bp, bits_t *Q) {
    uint64_t olo = oddceil(lo);
    bits_ones(Q);

    /* v5: per-block sqrt limit instead of the global bp upper bound. Every
     * composite 2p+1 with p <= hi has a factor <= sqrt(2*hi+1), so iterating
     * beyond this point is wasted work — most relevant for early blocks in
     * validate-full runs where hi << high. Negligible effect for the
     * production campaign (low = a(8)) but zero-cost to add. */
    uint64_t qlim = isqrt64(2 * hi + 1);

    for (size_t j = 1; j < bp->n; j++) {   /* skip r = 2: 2p+1 is always odd */
        uint64_t r = bp->p[j];
        if (r > qlim) break;
        uint64_t res = (r - 1) / 2;        /* forbidden residue of p mod r */
        uint64_t p   = first_cong(olo, r, res);
        if (!(p & 1)) p += r;              /* r odd ⇒ p+r flips parity */
        for (; p <= hi; p += 2 * r) {
            if (2 * p + 1 != r)
                bits_clr(Q, (p - olo) >> 1);
        }
    }
}

/* ---------------------------------------------------------------------------
 * Miller–Rabin, deterministic for n < 2^64.
 * Witness set from Jaeschke (1993) / Sorenson–Webster (2017): sufficient
 * for n < 3.317·10^24, well above 2^64.
 * Modular multiplication uses __uint128_t; GCC emits a native DIV instruction
 * on x86-64, not a libgcc call.
 * ------------------------------------------------------------------------- */

static inline uint64_t mmul(uint64_t a, uint64_t b, uint64_t m) {
    return (uint64_t)((__uint128_t)a * b % m);
}

static uint64_t mpow(uint64_t a, uint64_t d, uint64_t m) {
    uint64_t r = 1;
    while (d) {
        if (d & 1) r = mmul(r, a, m);
        a = mmul(a, a, m);
        d >>= 1;
    }
    return r;
}

static bool isprime64(uint64_t n) {
    static const uint64_t small[] = {2,3,5,7,11,13,17,19,23,29,31,37};
    if (n < 2) return false;
    for (size_t i = 0; i < sizeof small / 8; i++) {
        if (n == small[i]) return true;
        if (n % small[i] == 0) return false;
    }
    /* Factor n-1 = 2^s · d with d odd. */
    uint64_t d = n - 1, s = 0;
    while (!(d & 1)) { d >>= 1; s++; }

    static const uint64_t bases[] = {2, 325, 9375, 28178, 450775, 9780504, 1795265022};
    for (size_t i = 0; i < sizeof bases / 8; i++) {
        uint64_t a = bases[i] % n;
        if (!a) continue;
        uint64_t x = mpow(a, d, n);
        if (x == 1 || x == n - 1) continue;
        bool comp = true;
        for (uint64_t r = 1; r < s; r++) {
            x = mmul(x, x, n);
            if (x == n - 1) { comp = false; break; }
        }
        if (comp) return false;
    }
    return true;
}

/* ---------------------------------------------------------------------------
 * Scan one block. Produces:
 *   - a block_summary_t (for cross-border reconciliation)
 *   - updates to thread_rec (internal-to-block runs)
 *   - increments to counters via pc/sgc output parameters
 *
 * Run accounting:
 *   prefix_len  = length of SG-run starting at first_prime (0 if first prime
 *                 is not SG, or if block has no primes).
 *   suffix_len  = length of SG-run ending at last_prime (0 if last prime is
 *                 not SG).
 *   all_sg      = every prime in the block is SG.
 * ------------------------------------------------------------------------- */

static void scan(uint64_t lo, uint64_t hi,
                 const bits_t *P, const bits_t *Q,
                 unsigned maxrun,
                 block_summary_t *s, records_t *thread_rec,
                 uint64_t *pc, uint64_t *sgc)
{
    memset(s, 0, sizeof(*s));
    s->all_sg = 1;
    uint64_t olo       = oddceil(lo);
    uint64_t run_start = 0;
    uint32_t run       = 0;
    bool     seen_bad  = false;  /* becomes true when a non-SG prime is seen */

    /* p = 2 is Sophie Germain (2·2+1 = 5) and needs special handling because
     * the odd-only bitmap cannot represent it. */
    if (lo <= 2 && hi >= 2) {
        s->num_primes   = 1;
        (*pc)++;
        (*sgc)++;
        s->first_prime  = 2;
        s->prefix_len   = 1;
        s->prefix_start = 2;
        s->suffix_len   = 1;
        s->suffix_start = 2;
        run       = 1;
        run_start = 2;
        rec_update(thread_rec, 1, 2, maxrun);
    }

    /* Main loop over odd candidates. v5: word-scan with __builtin_ctzll
     * replaces the v4 bit-by-bit iteration. For each 64-bit word of P we
     * extract set bits one at a time using ctzll (one clock on Zen 4), jump
     * directly to each set bit, and clear it with the `mask &= mask - 1`
     * idiom. At production N ~ 10^14, prime density is ~1/ln(N) ~ 1/30, so
     * ~97% of bits are zero and the v4 loop wasted its time on them.
     *
     * Invariants preserved byte-for-byte from the v4 version: for every set
     * bit with index i such that 3 <= p = olo + 2i <= hi, the body updates
     * num_primes, pc, sgc, first_prime, run/run_start/seen_bad/all_sg, and
     * the prefix/suffix fields in exactly the same order. The order of bits
     * within the scan is strictly monotonic (low word to high word, low bit
     * to high bit within a word), same as v4. */
    for (uint64_t w = 0; w < P->nwords; w++) {
        uint64_t word = P->w[w];
        while (word) {
            unsigned bit = (unsigned)__builtin_ctzll(word);
            uint64_t i = (w << 6) + bit;
            word &= word - 1;

            if (i >= P->nbits) break;  /* last-word tail bits already masked off
                                        * by bits_ones, but defend anyway */
            uint64_t p = olo + (i << 1);
            if (p < 3 || p > hi) continue;

            bool sg = bits_get(Q, i);
            s->num_primes++;
            (*pc)++;
            if (!s->first_prime) s->first_prime = p;

            if (sg) {
                (*sgc)++;
                if (!run) run_start = p;
                run++;
                rec_update(thread_rec, run, run_start, maxrun);
            } else {
                seen_bad = true;
                s->all_sg = 0;
                run = 0;
                run_start = 0;
            }

            /* Prefix: extends only while we have not yet seen any non-SG
             * prime. v4 (ChatGPT §4.6): saturate at maxrun. */
            if (s->num_primes == 1) {
                s->prefix_len   = sg ? 1 : 0;
                s->prefix_start = sg ? p : 0;
            } else if (!seen_bad && sg) {
                if (s->prefix_len < maxrun)
                    s->prefix_len = (uint16_t)(s->prefix_len + 1);
            }

            /* Suffix: current run saturated at maxrun. */
            s->suffix_len   = (uint16_t)(run < maxrun ? run : maxrun);
            s->suffix_start = run ? run_start : 0;
        }
    }

    if (!s->num_primes) s->all_sg = 0;
}

/* Driver that allocates the two bitmaps, sieves, and scans. */
static bool block_process(uint64_t lo, uint64_t hi,
                          const primes_t *bp, unsigned maxrun, bool mr,
                          block_summary_t *s, records_t *thread_rec,
                          uint64_t *pc, uint64_t *sgc)
{
    bits_t P = {0}, Q = {0};
    uint64_t n = oddcount(lo, hi);
    if (!bits_alloc(&P, n) || !bits_alloc(&Q, n)) {
        bits_free(&P); bits_free(&Q);
        return false;
    }
    sieve_p(lo, hi, bp, &P);
    if (mr) {
        /* Independent check path: start from empty Q and set bits only for
         * primes p where Miller-Rabin confirms 2p+1 is prime. */
        memset(Q.w, 0, (size_t)Q.nwords * 8);
        uint64_t olo = oddceil(lo);
        for (uint64_t i = 0; i < P.nbits; i++)
            if (bits_get(&P, i) && isprime64(2 * (olo + (i << 1)) + 1))
                bits_set(&Q, i);
    } else {
        sieve_q(lo, hi, bp, &Q);
    }
    scan(lo, hi, &P, &Q, maxrun, s, thread_rec, pc, sgc);
    bits_free(&P); bits_free(&Q);
    return true;
}

/* ---------------------------------------------------------------------------
 * Incremental reconciliation within a batch.
 *
 * Two kinds of runs need recording:
 *   (i)  Runs that fit entirely within a single block — handled by scan()
 *        writing into thread_rec as they are found.
 *   (ii) Runs that cross one or more block boundaries — handled here by
 *        walking summaries in order and maintaining a (carry, carry_start)
 *        state across blocks.
 *
 * Invariants maintained across calls to reconcile_advance:
 *   - carry: length of the SG-run currently extending from previous blocks
 *            into the next-to-process block. 0 if no such run.
 *   - carry_start: start prime of that run.
 *
 * Blocks cover contiguous integer intervals, so "the last prime of block i
 * and the first prime of block i+1 are consecutive primes" is automatic;
 * no freshness check is needed here.
 * ------------------------------------------------------------------------- */

typedef struct {
    uint64_t carry;
    uint64_t carry_start;
} carry_state_t;

static void reconcile_advance(const block_summary_t *s,
                              carry_state_t *cs,
                              records_t *global_rec,
                              unsigned maxrun)
{
    if (!s->num_primes) return;  /* empty block: carry propagates unchanged */

    /* Close a cross-border run if both sides exist. Saturate at maxrun using
     * 64-bit arithmetic to eliminate the uint32 cast of v2 (ChatGPT §4.4). */
    if (cs->carry && s->prefix_len) {
        uint64_t merged = cs->carry + (uint64_t)s->prefix_len;
        if (merged > maxrun) merged = maxrun;
        rec_update(global_rec, (uint32_t)merged, cs->carry_start, maxrun);
    }

    if (s->all_sg) {
        /* Whole block extends the run. */
        if (!cs->carry) cs->carry_start = s->first_prime;
        uint64_t extended = cs->carry + (uint64_t)s->num_primes;
        /* Saturate: further prefix extensions beyond maxrun do not change
         * any record. This prevents cs->carry from growing to UINT64_MAX. */
        cs->carry = extended > maxrun ? maxrun : extended;
    } else if (s->suffix_len) {
        /* Block has a new run at its tail. */
        cs->carry_start = s->suffix_start;
        cs->carry       = s->suffix_len;
    } else {
        cs->carry       = 0;
        cs->carry_start = 0;
    }
}

/* ---------------------------------------------------------------------------
 * Results JSONL: append one line each time a new best a(k) candidate appears.
 * Lines are self-describing and can be parsed independently. v4 additions
 * (ChatGPT §4.3, §4.4): checks every I/O return code, and includes `batch`
 * so consumers can deduplicate re-emitted lines after a crash/resume using
 * (k, a_k, low, high, mode, block_odds, batch) as a key.
 * ------------------------------------------------------------------------- */

static void results_append(const char *path, const cfg_t *cfg,
                           unsigned k, uint64_t val,
                           double progress, uint64_t primes_scanned,
                           uint64_t sg_found, uint64_t batch_idx)
{
    if (!path) return;
    FILE *f = fopen(path, "a");
    if (!f) {
        fprintf(stderr, "warning: cannot append to %s: %s\n", path, strerror(errno));
        return;
    }
    time_t t = time(NULL);
    struct tm tm;
    gmtime_r(&t, &tm);
    char ts[32];
    strftime(ts, sizeof ts, "%Y-%m-%dT%H:%M:%SZ", &tm);

    bool ok = true;
    int n = fprintf(f,
            "{\"ts\":\"%s\",\"k\":%u,\"a_k\":%" PRIu64 ",\"progress\":%.6f,"
            "\"primes_scanned\":%" PRIu64 ",\"sg_found\":%" PRIu64 ","
            "\"low\":%" PRIu64 ",\"high\":%" PRIu64 ",\"mode\":\"%s\","
            "\"block_odds\":%" PRIu64 ",\"batch\":%" PRIu64 "}\n",
            ts, k, val, progress, primes_scanned, sg_found,
            cfg->low, cfg->high, cfg->mr ? "mr" : "double",
            cfg->block_odds, batch_idx);
    if (n < 0)            ok = false;
    if (fflush(f) != 0)   ok = false;
    if (fclose(f) != 0)   ok = false;
    if (!ok)
        fprintf(stderr, "warning: I/O failure while appending to %s\n", path);
}

/* ---------------------------------------------------------------------------
 * Checkpoint state (binary, versioned, atomic via tmp+rename).
 *
 * Layout:
 *   uint32 magic
 *   uint32 version
 *   cfg_t  cfg               (pointers cleared before write)
 *   uint64 next_batch
 *   uint64 carry, carry_start
 *   uint64 total_primes, total_sg
 *   records_t global_rec
 *   uint64 checksum (byte-wise 8-lane XOR over preceding bytes)
 * ------------------------------------------------------------------------- */

typedef struct {
    uint32_t magic;
    uint32_t version;
    uint64_t cfg_low, cfg_high, cfg_block_odds;
    uint32_t cfg_batch_blocks, cfg_target, cfg_max_run, cfg_mr;
    uint64_t next_batch;
    uint64_t carry, carry_start;
    uint64_t total_primes, total_sg;
    records_t global_rec;
    uint64_t checksum;
} state_t;

/* Byte-wise XOR checksum. Using `const unsigned char *` is always legal in C
 * (strict aliasing explicitly allows character-type access to any object);
 * the v2 `const uint64_t *` cast was a strict-aliasing violation flagged by
 * ChatGPT §6.1. We keep an 8-byte running XOR to preserve the 64-bit field
 * in state_t::checksum. */
static uint64_t state_checksum(const state_t *st) {
    const unsigned char *p = (const unsigned char *)st;
    size_t n = offsetof(state_t, checksum);
    uint64_t x = 0;
    for (size_t i = 0; i < n; i++)
        x ^= (uint64_t)p[i] << (8 * (i & 7));
    return x;
}

/* fsync the directory that contains `path`. After rename() on POSIX, the
 * directory entry is only guaranteed durable after a directory fsync.
 * v4 (ChatGPT §4.5): returns bool so the caller can react. Failure is
 * reported as a warning in state_save rather than a hard error because the
 * file content itself is on disk; the only at-risk property is the
 * atomicity-across-power-loss of the rename. */
static bool fsync_parent_dir(const char *path) {
    char buf[PATH_MAX];
    int r = snprintf(buf, sizeof buf, "%s", path);
    if (r < 0 || (size_t)r >= sizeof buf) return false;
    char *dir = dirname(buf);
    int dfd = open(dir, O_DIRECTORY | O_RDONLY);
    if (dfd < 0) return false;
    bool ok = (fsync(dfd) == 0);
    if (close(dfd) != 0) ok = false;
    return ok;
}

static bool state_save(const char *path, const cfg_t *cfg,
                       uint64_t next_batch, carry_state_t cs,
                       uint64_t tp, uint64_t tsg,
                       const records_t *global_rec)
{
    if (!path) return true;
    state_t st = {0};
    st.magic   = STATE_MAGIC;
    st.version = STATE_VERSION;
    st.cfg_low          = cfg->low;
    st.cfg_high         = cfg->high;
    st.cfg_block_odds   = cfg->block_odds;
    st.cfg_batch_blocks = cfg->batch_blocks;
    st.cfg_target       = cfg->target;
    st.cfg_max_run      = cfg->max_run;
    st.cfg_mr           = cfg->mr ? 1 : 0;
    st.next_batch       = next_batch;
    st.carry            = cs.carry;
    st.carry_start      = cs.carry_start;
    st.total_primes     = tp;
    st.total_sg         = tsg;
    st.global_rec       = *global_rec;
    st.checksum         = state_checksum(&st);

    /* Build tmp path safely. */
    char tmp[PATH_MAX];
    int r = snprintf(tmp, sizeof tmp, "%s.tmp", path);
    if (r < 0 || (size_t)r >= sizeof tmp) {
        fprintf(stderr, "warning: state path too long\n");
        return false;
    }

    FILE *f = fopen(tmp, "wb");
    if (!f) {
        fprintf(stderr, "warning: cannot write %s: %s\n", tmp, strerror(errno));
        return false;
    }

    /* Check every I/O return value (ChatGPT §6.2). */
    bool ok = true;
    if (fwrite(&st, 1, sizeof st, f) != sizeof st) { ok = false; }
    if (ok && fflush(f) != 0)                      { ok = false; }

    int fd = fileno(f);
    if (ok && (fd < 0 || fsync(fd) != 0))          { ok = false; }
    if (fclose(f) != 0)                             { ok = false; }

    if (!ok) {
        fprintf(stderr, "warning: I/O failure writing %s\n", tmp);
        (void)unlink(tmp);
        return false;
    }

    if (rename(tmp, path) != 0) {
        fprintf(stderr, "warning: cannot rename %s -> %s: %s\n",
                tmp, path, strerror(errno));
        (void)unlink(tmp);
        return false;
    }

    /* After rename, the directory entry itself must be fsync'd for full
     * durability on POSIX. Omitting this can lose the rename on power loss
     * even though the file content is on disk. v4 (ChatGPT §4.5): warn if
     * the parent fsync fails; the state file is on disk and usable, but the
     * rename may not survive a simultaneous power loss. The checkpoint is
     * considered successful because the content is durable. */
    if (!fsync_parent_dir(path))
        fprintf(stderr, "warning: cannot fsync parent directory for %s "
                "(checkpoint written but rename may not be durable on power loss)\n", path);
    return true;
}

/* Forward declaration: state_load calls validate_cfg after loading the config
 * but before computing derived quantities (v4 change, ChatGPT §4.2). */
static bool validate_cfg(const cfg_t *c);

static bool state_load(const char *path, cfg_t *cfg_out,
                       uint64_t *next_batch, carry_state_t *cs_out,
                       uint64_t *tp, uint64_t *tsg,
                       records_t *global_rec_out)
{
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "error: cannot open %s: %s\n", path, strerror(errno)); return false; }
    state_t st;
    size_t rd = fread(&st, 1, sizeof st, f);
    fclose(f);
    if (rd != sizeof st)            { fprintf(stderr, "error: %s: short read\n",      path); return false; }
    if (st.magic   != STATE_MAGIC)  { fprintf(stderr, "error: %s: bad magic\n",       path); return false; }
    if (st.version != STATE_VERSION){ fprintf(stderr, "error: %s: unknown version\n", path); return false; }
    if (st.checksum != state_checksum(&st)) { fprintf(stderr, "error: %s: checksum mismatch\n", path); return false; }

    /* Adopt the config from the state file, but keep user-supplied runtime
     * knobs (threads, progress/checkpoint intervals, results path) as-is. */
    cfg_out->low          = st.cfg_low;
    cfg_out->high         = st.cfg_high;
    cfg_out->block_odds   = st.cfg_block_odds;
    cfg_out->batch_blocks = st.cfg_batch_blocks;
    cfg_out->target       = st.cfg_target;
    cfg_out->max_run      = st.cfg_max_run;
    cfg_out->mr           = st.cfg_mr ? true : false;
    *next_batch           = st.next_batch;
    cs_out->carry         = st.carry;
    cs_out->carry_start   = st.carry_start;
    *tp                   = st.total_primes;
    *tsg                  = st.total_sg;
    *global_rec_out       = st.global_rec;

    /* v4 (ChatGPT §4.2): validate the loaded config BEFORE doing any derived
     * arithmetic. Without this, a corrupt/crafted state file whose checksum
     * happens to match could trigger div-by-zero in the ceildiv below. */
    if (!validate_cfg(cfg_out)) {
        fprintf(stderr, "error: %s: invalid config in state file\n", path);
        return false;
    }

    /* Post-load sanity checks (ChatGPT §6.3 from v2 review). validate_cfg has
     * already ruled out the divide-by-zero cases, so these checks can assume
     * block_odds > 0, batch_blocks > 0, low <= high. */
    {
        uint64_t bspan  = cfg_out->block_odds * 2;
        uint64_t span   = cfg_out->high - cfg_out->low + 1;
        uint64_t nb     = ceildiv(span, bspan);
        uint64_t nbatch = ceildiv(nb, cfg_out->batch_blocks);
        if (*next_batch > nbatch) {
            fprintf(stderr, "error: %s: start_batch %" PRIu64 " > nbatch %" PRIu64 "\n",
                    path, *next_batch, nbatch);
            return false;
        }
    }
    /* Saturate carry to max_run. */
    if (cs_out->carry > cfg_out->max_run)
        cs_out->carry = cfg_out->max_run;

    return true;
}

/* ---------------------------------------------------------------------------
 * Progress reporting
 * ------------------------------------------------------------------------- */

static void fmt_hms(double s, char *out, size_t n) {
    if (s < 0 || !isfinite(s)) { snprintf(out, n, "--:--:--"); return; }
    long h = (long)(s / 3600.0);
    long m = (long)((s - 3600.0 * h) / 60.0);
    long q = (long)(s - 3600.0 * h - 60.0 * m);
    snprintf(out, n, "%02ld:%02ld:%02ld", h, m, q);
}

static void progress_report(double t0, uint64_t done_batches, uint64_t total_batches,
                            uint64_t tp, uint64_t tsg,
                            const records_t *global_rec, unsigned maxrun)
{
    double elapsed = now_sec() - t0;
    double frac    = total_batches ? (double)done_batches / (double)total_batches : 0.0;
    double eta     = (frac > 1e-9) ? elapsed * (1.0 - frac) / frac : -1.0;
    char te[16], ee[16];
    fmt_hms(elapsed, te, sizeof te);
    fmt_hms(eta,     ee, sizeof ee);

    unsigned best_k = 0;
    uint64_t best_v = 0;
    for (unsigned k = maxrun; k >= 1; k--)
        if (global_rec->start[k]) { best_k = k; best_v = global_rec->start[k]; break; }

    fprintf(stderr,
            "[%s] batch %" PRIu64 "/%" PRIu64 " (%.1f%%) | primes/s %.2e | SG/s %.2e | "
            "best k=%u at %" PRIu64 " | ETA %s\n",
            te, done_batches, total_batches, 100.0 * frac,
            elapsed > 0 ? (double)tp / elapsed : 0.0,
            elapsed > 0 ? (double)tsg / elapsed : 0.0,
            best_k, best_v, ee);
    fflush(stderr);
}

/* ---------------------------------------------------------------------------
 * Top-level runner with batch processing.
 * ------------------------------------------------------------------------- */

typedef struct {
    uint64_t total_batches;
    uint64_t total_primes;
    uint64_t total_sg;
    double   elapsed;
    records_t records;
    bool     ok;
} result_t;

static bool validate_cfg(const cfg_t *c) {
    if (c->low > c->high)                       { fprintf(stderr, "error: low > high\n"); return false; }
    if (c->high > (UINT64_MAX - 1) / 2)         { fprintf(stderr, "error: high too large, 2*high+1 would overflow\n"); return false; }
    if (c->block_odds == 0)                     { fprintf(stderr, "error: block_odds must be > 0\n"); return false; }
    /* v4 (ChatGPT §4.7 + §4.1): cap block_odds at 2^32 - 1. This is vastly
     * larger than any realistic production value (default 2^22 = 4 MiB block)
     * but small enough that (1) `num_primes` in a block cannot overflow
     * uint32_t (primes ~ block_odds/ln(N), so well under 2^32), and (2) the
     * per-block arithmetic `bi * bspan = bi * 2*block_odds` cannot overflow
     * for any physically realizable search span (bi < 2^31). */
    if (c->block_odds > ((uint64_t)UINT32_MAX))  { fprintf(stderr, "error: block_odds > 2^32-1 not supported\n"); return false; }
    if (c->batch_blocks == 0)                   { fprintf(stderr, "error: batch_blocks must be > 0\n"); return false; }
    if (c->max_run == 0 || c->max_run > MAX_RUN){ fprintf(stderr, "error: max_run out of range\n"); return false; }
    if (c->target == 0 || c->target > c->max_run){fprintf(stderr, "error: target out of range\n"); return false; }
    /* ChatGPT §6 patch: reject NaN/Inf and unreasonable values. */
    if (!isfinite(c->checkpoint_sec) || c->checkpoint_sec < 0.0) {
        fprintf(stderr, "error: checkpoint_sec must be finite and nonnegative\n");
        return false;
    }
    if (!isfinite(c->progress_sec) || c->progress_sec < 0.0) {
        fprintf(stderr, "error: progress_sec must be finite and nonnegative\n");
        return false;
    }
    /* Sanity bound on threads: > 256 is unrealistic on consumer/workstation
     * hardware and the cast to int would be UB for values > INT_MAX. */
    if (c->threads > 256) {
        fprintf(stderr, "error: threads > 256 is not supported\n");
        return false;
    }
    /* Base-prime sieve capacity: isqrt(2*high+1) must fit in UINT32_MAX so
     * base_primes() can succeed. For the nominal 5·10^14 campaign,
     * lim ≈ 3.16·10^7, well under the 4.29·10^9 limit. */
    uint64_t lim = isqrt64(2 * c->high + 1) + 1;
    if (lim > UINT32_MAX) {
        fprintf(stderr, "error: high too large for base-prime sieve (lim=%" PRIu64 ")\n", lim);
        return false;
    }
    return true;
}

static bool run(const cfg_t *cfg, uint64_t start_batch_in,
                carry_state_t cs_in, records_t global_rec_in,
                uint64_t tp_in, uint64_t tsg_in,
                result_t *out)
{
    if (!validate_cfg(cfg)) return false;
    memset(out, 0, sizeof *out);

    double t0 = now_sec();

    primes_t bp;
    uint64_t lim = isqrt64(2 * cfg->high + 1) + 1;
    if (!base_primes(lim, &bp)) {
        fprintf(stderr, "error: base prime generation failed\n");
        return false;
    }

    uint64_t span   = cfg->high - cfg->low + 1;
    uint64_t bspan  = cfg->block_odds * 2;
    /* Use ceildiv() helper instead of (span + bspan - 1) / bspan; the latter
     * overflows if span + bspan exceeds UINT64_MAX. Not reachable on the
     * production config but fixing this removes an extreme-parameter trap. */
    uint64_t nb     = ceildiv(span, bspan);
    uint64_t nbatch = ceildiv(nb, cfg->batch_blocks);

    int nthreads = cfg->threads ? (int)cfg->threads : omp_get_max_threads();
    omp_set_num_threads(nthreads);

    /* Per-thread records: runs found strictly inside a single block. */
    records_t *thread_rec = calloc((size_t)nthreads, sizeof(records_t));
    /* Per-batch summary array: small, reusable across batches. */
    block_summary_t *sbatch = calloc(cfg->batch_blocks, sizeof(block_summary_t));
    if (!thread_rec || !sbatch) {
        fprintf(stderr, "error: allocation failure\n");
        free(thread_rec); free(sbatch); free_primes(&bp);
        return false;
    }

    records_t global_rec = global_rec_in;
    carry_state_t cs     = cs_in;
    uint64_t total_primes = tp_in;
    uint64_t total_sg     = tsg_in;
    uint64_t fail_blocks  = 0;

    double last_progress   = now_sec();
    double last_checkpoint = now_sec();

    /* Snapshot of global records seen on the previous progress check, used to
     * emit results.jsonl lines only when something new appears. */
    records_t prev_reported = global_rec_in;

    if (start_batch_in > 0)
        fprintf(stderr, "resuming from batch %" PRIu64 "/%" PRIu64 "\n", start_batch_in, nbatch);
    fprintf(stderr, "campaign: low=%" PRIu64 " high=%" PRIu64 " target=%u max_run=%u threads=%d\n",
            cfg->low, cfg->high, cfg->target, cfg->max_run, nthreads);
    fprintf(stderr, "layout: block_odds=%" PRIu64 " nb=%" PRIu64 " batch_blocks=%u nbatch=%" PRIu64 "\n",
            cfg->block_odds, nb, cfg->batch_blocks, nbatch);

    for (uint64_t batch = start_batch_in; batch < nbatch; batch++) {
        uint64_t b0 = batch * cfg->batch_blocks;
        uint64_t b1 = b0 + cfg->batch_blocks;
        if (b1 > nb) b1 = nb;
        uint32_t bsize = (uint32_t)(b1 - b0);

        uint64_t batch_primes = 0, batch_sg = 0;
        uint64_t batch_fail   = 0;

        /* v5: schedule(runtime) honors the OMP_SCHEDULE env var at launch
         * time, so the operator can benchmark schedules without recompiling.
         * Default (if OMP_SCHEDULE is unset) is implementation-defined but
         * typically static. For explicit control on Carlo's 7940HS:
         *   OMP_SCHEDULE="static"        lowest overhead, uniform loads
         *   OMP_SCHEDULE="guided,4"      adaptive, good for mildly skewed
         *   OMP_SCHEDULE="dynamic,8"     original with less overhead than ,1
         * The reduction stays the same. */
#pragma omp parallel for schedule(runtime) reduction(+:batch_primes,batch_sg,batch_fail)
        for (uint32_t k = 0; k < bsize; k++) {
            uint64_t bi = b0 + k;
            /* v4 fix (ChatGPT §4.1): compute the block's high endpoint safely.
             * v3 used `hi = lo + bspan - 1; if (hi > cfg->high) hi = cfg->high;`
             * which can overflow before the clamp when bspan is large. With
             * block_odds now capped at 2^32 in validate_cfg, `bi * bspan`
             * cannot overflow either (bi < 2^31 for any physically allocable
             * span). The `remaining` clamp also makes the final block
             * naturally fit the trailing partial span. */
            uint64_t lo = cfg->low + bi * bspan;
            uint64_t remaining = cfg->high - lo + 1;
            uint64_t this_span = (remaining < bspan) ? remaining : bspan;
            uint64_t hi = lo + this_span - 1;
            int tid = omp_get_thread_num();
            uint64_t a = 0, b = 0;
            if (!block_process(lo, hi, &bp, cfg->max_run, cfg->mr,
                               &sbatch[k], &thread_rec[tid], &a, &b))
                batch_fail++;
            batch_primes += a;
            batch_sg     += b;
        }

        /* Any block failure invalidates the whole campaign: we cannot
         * reconcile across a hole. Fail loudly. */
        if (batch_fail) {
            fprintf(stderr, "error: %" PRIu64 " block(s) failed in batch %" PRIu64 "\n",
                    batch_fail, batch);
            fail_blocks += batch_fail;
            free(thread_rec); free(sbatch); free_primes(&bp);
            return false;
        }

        total_primes += batch_primes;
        total_sg     += batch_sg;

        /* Sequential cross-border reconciliation within this batch. */
        for (uint32_t k = 0; k < bsize; k++)
            reconcile_advance(&sbatch[k], &cs, &global_rec, cfg->max_run);

        /* Merge per-thread internal records into a running "observable" view.
         * We do this before emitting results so that results.jsonl reflects
         * both internal and cross-border findings. */
        records_t observable = global_rec;
        for (int t = 0; t < nthreads; t++)
            rec_merge(&observable, &thread_rec[t], cfg->max_run);

        /* Emit results for any newly-dropped best values. */
        if (cfg->results_file) {
            double progress = nbatch ? (double)(batch + 1) / (double)nbatch : 1.0;
            for (unsigned kk = 1; kk <= cfg->max_run; kk++) {
                if (observable.start[kk] &&
                    (!prev_reported.start[kk] || observable.start[kk] < prev_reported.start[kk])) {
                    results_append(cfg->results_file, cfg, kk, observable.start[kk],
                                   progress, total_primes, total_sg, batch + 1);
                }
            }
        }
        prev_reported = observable;

        /* Periodic progress report. */
        double now = now_sec();
        if (now - last_progress >= cfg->progress_sec || batch + 1 == nbatch) {
            progress_report(t0, batch + 1, nbatch, total_primes, total_sg,
                            &observable, cfg->max_run);
            last_progress = now;
        }

        /* Periodic checkpoint: fold thread_rec into global_rec, then save. */
        if (cfg->state_file && (now - last_checkpoint >= cfg->checkpoint_sec)) {
            for (int t = 0; t < nthreads; t++) {
                rec_merge(&global_rec, &thread_rec[t], cfg->max_run);
                rec_init(&thread_rec[t]);
            }
            if (state_save(cfg->state_file, cfg, batch + 1, cs,
                           total_primes, total_sg, &global_rec))
                fprintf(stderr, "checkpoint saved to %s at batch %" PRIu64 "\n",
                        cfg->state_file, batch + 1);
            last_checkpoint = now;
        }
    }

    /* Final merge of thread-local records into the global view. */
    for (int t = 0; t < nthreads; t++)
        rec_merge(&global_rec, &thread_rec[t], cfg->max_run);

    /* Final checkpoint so that a later --resume picks up "done". */
    if (cfg->state_file)
        state_save(cfg->state_file, cfg, nbatch, cs, total_primes, total_sg, &global_rec);

    out->total_batches = nbatch;
    out->total_primes  = total_primes;
    out->total_sg      = total_sg;
    out->elapsed       = now_sec() - t0;
    out->records       = global_rec;
    out->ok            = (fail_blocks == 0);

    free(thread_rec);
    free(sbatch);
    free_primes(&bp);
    return out->ok;
}

/* ---------------------------------------------------------------------------
 * Oracle sets and validation
 *
 * Reference values from OEIS A338700 (retrieved 2026-04-24):
 *   a(1)=a(2)=a(3) = 2
 *   a(4) = 1 433 849
 *   a(5) = 9 816 899
 *   a(6) = 445 480 319
 *   a(7) = 298 098 924 131
 *   a(8) = 1 372 604 395 439
 *
 * Three validation levels:
 *   small:  [0, 10^7]            verifies a(1..5)         (< 1 s)
 *   medium: [0, 5·10^8]          verifies a(1..6)         (seconds)
 *   full:   [0, 2·10^12]         verifies a(1..8)         (hours)
 * ------------------------------------------------------------------------- */

typedef struct {
    const char *name;
    uint64_t    low, high;
    unsigned    target, max_run;
    uint64_t    expected[MAX_RUN + 1];
} oracle_t;

static const oracle_t ORACLES[] = {
    { "small",
      0ULL, 10000000ULL, 5, 5,
      { 0, 2, 2, 2, 1433849ULL, 9816899ULL } },
    { "medium",
      0ULL, 500000000ULL, 6, 6,
      { 0, 2, 2, 2, 1433849ULL, 9816899ULL, 445480319ULL } },
    { "full",
      0ULL, 2000000000000ULL, 8, MAX_RUN,
      { 0, 2, 2, 2, 1433849ULL, 9816899ULL, 445480319ULL,
        298098924131ULL, 1372604395439ULL } },
};

static int validate_mode(const cfg_t *user_cfg, const char *level) {
    const oracle_t *orc = NULL;
    for (size_t i = 0; i < sizeof ORACLES / sizeof ORACLES[0]; i++)
        if (!strcmp(level, ORACLES[i].name)) { orc = &ORACLES[i]; break; }
    if (!orc) { fprintf(stderr, "error: unknown validate level '%s' (use small|medium|full)\n", level); return 2; }

    cfg_t c = *user_cfg;
    c.low          = orc->low;
    c.high         = orc->high;
    c.target       = orc->target;
    c.max_run      = orc->max_run;
    c.results_file = NULL;  /* validation never writes to results file */
    c.state_file   = NULL;  /* or to state file */

    fprintf(stderr, "validate-%s: [%" PRIu64 ", %" PRIu64 "], target=%u\n",
            orc->name, c.low, c.high, c.target);

    result_t r;
    carry_state_t cs0 = {0, 0};
    records_t rec0; rec_init(&rec0);
    if (!run(&c, 0, cs0, rec0, 0, 0, &r)) {
        fprintf(stderr, "validate-%s: run failed\n", orc->name);
        return 1;
    }

    bool ok = true;
    for (unsigned k = 1; k <= orc->target; k++) {
        if (r.records.start[k] != orc->expected[k]) {
            fprintf(stderr, "  a(%u) = %" PRIu64 " (expected %" PRIu64 ") FAIL\n",
                    k, r.records.start[k], orc->expected[k]);
            ok = false;
        } else {
            fprintf(stderr, "  a(%u) = %" PRIu64 " OK\n", k, r.records.start[k]);
        }
    }
    fprintf(stderr, "validate-%s: %s  elapsed=%.3fs  primes=%" PRIu64 "  sg=%" PRIu64 "\n",
            orc->name, ok ? "PASS" : "FAIL", r.elapsed, r.total_primes, r.total_sg);
    return ok ? 0 : 1;
}

/* ---------------------------------------------------------------------------
 * CLI
 * ------------------------------------------------------------------------- */

static void usage(const char *arg0) {
    fprintf(stderr,
"Usage: %s [OPTIONS]\n"
"Search OEIS A338700 — start of runs of n+ consecutive Sophie Germain primes.\n"
"\n"
"  --low N              lower bound of search (default: a(8) = %llu)\n"
"  --high N             upper bound of search (default: %llu)\n"
"  --target K           run length to hunt for (default: 9)\n"
"  --max-run K          report a(1)..a(K) simultaneously (default: 16)\n"
"  --block-odds N       odd integers per block (default: 2^22)\n"
"  --batch-blocks N     blocks per OpenMP batch (default: %u)\n"
"  --threads N          OpenMP thread count (default: runtime)\n"
"  --mode double|mr     sieve mode (default: double)\n"
"\n"
"  --validate LEVEL     run oracle set (LEVEL = small | medium | full), then exit\n"
"  --bench              short benchmark (100M window above a(8)), then exit\n"
"\n"
"  --results FILE       append JSONL record for every improved a(k)\n"
"  --state FILE         save a resumable checkpoint here every --checkpoint-every sec\n"
"  --resume FILE        read state from FILE and continue\n"
"  --checkpoint-every S checkpoint interval in seconds (default: %.0f)\n"
"  --progress-every S   progress reporting interval in seconds (default: %.0f)\n"
"\n"
"Example production run:\n"
"  %s --low %llu --high 500000000000000 --target 9 --max-run 16 \\\n"
"    --threads 16 --mode double --results a338700.jsonl --state a338700.state\n",
        arg0,
        (unsigned long long)A8,
        (unsigned long long)DEFAULT_HIGH,
        DEFAULT_BATCH_BLOCKS,
        DEFAULT_CHECKPOINT_SEC,
        DEFAULT_PROGRESS_SEC,
        arg0,
        (unsigned long long)A8);
}

int main(int argc, char **argv) {
    cfg_t c = {
        .low = A8,
        .high = DEFAULT_HIGH,
        .block_odds = DEFAULT_BLOCK_ODDS,
        .batch_blocks = DEFAULT_BATCH_BLOCKS,
        .threads = 0,
        .target = 9,
        .max_run = 16,
        .mr = false,
        .checkpoint_sec = DEFAULT_CHECKPOINT_SEC,
        .progress_sec = DEFAULT_PROGRESS_SEC,
        .results_file = NULL,
        .state_file = NULL,
    };
    const char *validate_level = NULL;
    const char *resume_file = NULL;
    bool bench = false;

    /* v4: a small helper macro for uniform parse-error messages. Silent
     * return-2 made it hard for a user to see what went wrong when they
     * typed e.g. `--block-odds -1`. */
#define PARSE_ERR(flag, val) \
    do { fprintf(stderr, "error: invalid value for %s: '%s'\n", (flag), (val)); return 2; } while (0)

    for (int i = 1; i < argc; i++) {
        const char *a = argv[i];
        if      (!strcmp(a, "--low")           && i+1 < argc) { const char *v = argv[++i]; if (!parse_u64(v, &c.low))          PARSE_ERR("--low",          v); }
        else if (!strcmp(a, "--high")          && i+1 < argc) { const char *v = argv[++i]; if (!parse_u64(v, &c.high))         PARSE_ERR("--high",         v); }
        else if (!strcmp(a, "--target")        && i+1 < argc) { const char *v = argv[++i]; if (!parse_u (v, &c.target))        PARSE_ERR("--target",       v); }
        else if (!strcmp(a, "--max-run")       && i+1 < argc) { const char *v = argv[++i]; if (!parse_u (v, &c.max_run))       PARSE_ERR("--max-run",      v); }
        else if (!strcmp(a, "--block-odds")    && i+1 < argc) { const char *v = argv[++i]; if (!parse_u64(v, &c.block_odds))   PARSE_ERR("--block-odds",   v); }
        else if (!strcmp(a, "--batch-blocks")  && i+1 < argc) { const char *v = argv[++i]; unsigned u; if (!parse_u(v, &u) || u == 0) PARSE_ERR("--batch-blocks", v); c.batch_blocks = u; }
        else if (!strcmp(a, "--threads")       && i+1 < argc) { const char *v = argv[++i]; if (!parse_u (v, &c.threads))       PARSE_ERR("--threads",      v); }
        else if (!strcmp(a, "--mode")          && i+1 < argc) {
            const char *m = argv[++i];
            if      (!strcmp(m, "double")) c.mr = false;
            else if (!strcmp(m, "mr"))     c.mr = true;
            else { fprintf(stderr, "error: --mode must be double|mr\n"); return 2; }
        }
        else if (!strcmp(a, "--validate")      && i+1 < argc) { validate_level = argv[++i]; }
        else if (!strcmp(a, "--bench"))                        { bench = true; }
        else if (!strcmp(a, "--results")       && i+1 < argc) { c.results_file = argv[++i]; }
        else if (!strcmp(a, "--state")         && i+1 < argc) { c.state_file   = argv[++i]; }
        else if (!strcmp(a, "--resume")        && i+1 < argc) { resume_file    = argv[++i]; }
        else if (!strcmp(a, "--checkpoint-every") && i+1 < argc) { const char *v = argv[++i]; if (!parse_f(v, &c.checkpoint_sec)) PARSE_ERR("--checkpoint-every", v); }
        else if (!strcmp(a, "--progress-every")   && i+1 < argc) { const char *v = argv[++i]; if (!parse_f(v, &c.progress_sec))   PARSE_ERR("--progress-every",   v); }
        else if (!strcmp(a, "--help") || !strcmp(a, "-h"))    { usage(argv[0]); return 0; }
        else { fprintf(stderr, "error: unknown option '%s'\n", a); usage(argv[0]); return 2; }
    }
#undef PARSE_ERR

    if (c.max_run < c.target) c.max_run = c.target;

    /* --validate takes precedence over everything else. */
    if (validate_level) return validate_mode(&c, validate_level);

    /* --bench: a short 100M window benchmark above a(8). */
    if (bench) {
        c.low  = A8;
        c.high = A8 + 100000000ULL - 1;
        c.results_file = NULL;
        c.state_file   = NULL;
    }

    /* Initial state: either fresh or loaded from --resume. */
    uint64_t start_batch = 0;
    carry_state_t cs0 = {0, 0};
    records_t rec0; rec_init(&rec0);
    uint64_t tp0 = 0, tsg0 = 0;

    if (resume_file) {
        /* Resume overrides the config search window (low/high/block_odds/...)
         * with values from the state file, so they match exactly the run we
         * interrupted. Runtime options (threads, files, intervals) stay. */
        if (!state_load(resume_file, &c, &start_batch, &cs0, &tp0, &tsg0, &rec0))
            return 1;
        fprintf(stderr, "resume: start_batch=%" PRIu64 " carry=%" PRIu64 " carry_start=%" PRIu64
                " total_primes=%" PRIu64 " total_sg=%" PRIu64 "\n",
                start_batch, cs0.carry, cs0.carry_start, tp0, tsg0);
    }

    result_t r;
    if (!run(&c, start_batch, cs0, rec0, tp0, tsg0, &r))
        return 1;

    /* Final human-readable summary. The semantics of "FOUND a(k)" depend on
     * whether the window starts at 0 (globally proven) or at a user-specified
     * low > 0 (first-in-window candidate). For the production campaign,
     * low = a(8) is mathematically sufficient to find a(9) because A338700
     * is nondecreasing: a(n+1) >= a(n), with possible equality if the first
     * run of length n already has length >= n+1. A run of length 9 is also
     * a run of length 8, so the first run of length 9 is at position >= a(8).
     * The search includes a(8), so equality would be detected. Per ChatGPT's
     * v4 review §7.2 which flagged that v4's comment incorrectly claimed
     * strict inequality. */
    printf("A338700 v5 mode=%s low=%" PRIu64 " high=%" PRIu64
           " target=%u max_run=%u threads=%u\n",
           c.mr ? "mr" : "double", c.low, c.high, c.target, c.max_run, c.threads);
    printf("done %.3fs batches=%" PRIu64 " primes=%" PRIu64
           " sg=%" PRIu64 " span/s=%.3e\n",
           r.elapsed, r.total_batches, r.total_primes, r.total_sg,
           (double)(c.high - c.low + 1) / (r.elapsed > 0 ? r.elapsed : 1.0));

    const bool window_from_zero = (c.low == 0);
    const char *label = window_from_zero ? "a" : "first-in-window candidate for a";
    for (unsigned k = 1; k <= c.max_run; k++)
        if (r.records.start[k])
            printf("%s(%u) = %" PRIu64 "\n", label, k, r.records.start[k]);

    if (r.records.start[c.target]) {
        if (window_from_zero)
            printf("FOUND a(%u) = %" PRIu64 " (globally proven: window starts at 0)\n",
                   c.target, r.records.start[c.target]);
        else
            printf("FOUND first-in-window candidate for a(%u) = %" PRIu64
                   " in [%" PRIu64 ", %" PRIu64 "]\n"
                   "  note: globally proven only if no run of length %u exists in [0, %" PRIu64 ")\n",
                   c.target, r.records.start[c.target], c.low, c.high,
                   c.target, c.low);
    } else {
        printf("a(%u) not found in [%" PRIu64 ", %" PRIu64 "]\n",
               c.target, c.low, c.high);
    }

    return 0;
}
