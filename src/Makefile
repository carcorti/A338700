# Makefile for A338700
# Requires gcc with OpenMP and C17 support.

CC       = gcc
CSTD     = -std=c17
WARN     = -Wall -Wextra -Wpedantic -Wshadow -Wstrict-prototypes
OPT      = -O2 -march=native
OMP      = -fopenmp
CFLAGS   = $(CSTD) $(WARN) $(OPT) $(OMP)
LDFLAGS  = $(OMP) -lm

SRC      = a338700.c
BIN      = a338700

.PHONY: all release release-znver4 release-lto sanitize profile validate validate-small validate-medium validate-full bench bench-schedule bench-blockodds clean help

all: release

release: $(BIN)

$(BIN): $(SRC)
	$(CC) $(CFLAGS) -o $(BIN) $(SRC) $(LDFLAGS)

# Zen 4 tuned build for MinisForum UM790 Pro (Ryzen 9 7940HS).
release-znver4: $(SRC)
	$(CC) $(CSTD) $(WARN) -O2 -march=znver4 -mtune=znver4 $(OMP) \
	      -o $(BIN) $(SRC) $(LDFLAGS)

# LTO + O3 for benchmarking. Verify oracle passes before trusting.
release-lto: $(SRC)
	$(CC) $(CSTD) $(WARN) -O3 -march=native -flto $(OMP) \
	      -o $(BIN) $(SRC) $(LDFLAGS) -flto

# AddressSanitizer + UBSan. Diagnostic only.
sanitize: $(SRC)
	$(CC) $(CSTD) $(WARN) -O1 -g $(OMP) \
	      -fsanitize=address,undefined -o $(BIN) $(SRC) $(OMP) -lm

# gprof-compatible build.
profile: $(SRC)
	$(CC) $(CFLAGS) -pg -o $(BIN) $(SRC) $(LDFLAGS)

# Validation wrappers.
validate: validate-small validate-medium

validate-small: release
	./$(BIN) --validate small --threads 4 --mode double
	./$(BIN) --validate small --threads 4 --mode mr

validate-medium: release
	./$(BIN) --validate medium --threads 16 --mode double
	./$(BIN) --validate medium --threads 16 --mode mr

validate-full: release
	./$(BIN) --validate full --threads 16 --mode double

# Short benchmark above a(8).
bench: release
	./$(BIN) --bench --threads 16 --mode double
	./$(BIN) --bench --threads 16 --mode mr

# v5 benchmark helpers.
# Use these to pick OMP_SCHEDULE and --block-odds for the production campaign.
bench-schedule: release
	@echo "Benchmarking OMP_SCHEDULE variants..."
	OMP_SCHEDULE="static"       ./$(BIN) --bench --threads 16 --mode double
	OMP_SCHEDULE="static,1"     ./$(BIN) --bench --threads 16 --mode double
	OMP_SCHEDULE="guided,4"     ./$(BIN) --bench --threads 16 --mode double
	OMP_SCHEDULE="dynamic,1"    ./$(BIN) --bench --threads 16 --mode double
	OMP_SCHEDULE="dynamic,8"    ./$(BIN) --bench --threads 16 --mode double

bench-blockodds: release
	@echo "Benchmarking block_odds values (4 MB through 32 MB per thread)..."
	./$(BIN) --low 1372604395439 --high 1372804395439 --target 9 --threads 16 --mode double --block-odds 1048576
	./$(BIN) --low 1372604395439 --high 1372804395439 --target 9 --threads 16 --mode double --block-odds 2097152
	./$(BIN) --low 1372604395439 --high 1372804395439 --target 9 --threads 16 --mode double --block-odds 4194304
	./$(BIN) --low 1372604395439 --high 1372804395439 --target 9 --threads 16 --mode double --block-odds 8388608

clean:
	rm -f $(BIN) gmon.out

help:
	@echo "Targets:"
	@echo "  release           optimized build, -O2 -march=native (default)"
	@echo "  release-znver4    -O2 -march=znver4 (Ryzen 7940HS, gcc 13+)"
	@echo "  release-lto       -O3 -march=native -flto (benchmark candidate)"
	@echo "  sanitize          ASan + UBSan diagnostic build"
	@echo "  profile           gprof-compatible build"
	@echo "  validate          small + medium oracle sets"
	@echo "  validate-small    [0, 1e7], verifies a(1..5)"
	@echo "  validate-medium   [0, 5e8], verifies a(1..6)"
	@echo "  validate-full     [0, 2e12], verifies a(1..8)  (hours)"
	@echo "  bench             short benchmark window above a(8)"
	@echo "  bench-schedule    sweep OMP_SCHEDULE variants"
	@echo "  bench-blockodds   sweep --block-odds 2^20..2^23 on production window"
	@echo "  clean             remove binaries"
	@echo ""
	@echo "Environment:"
	@echo "  OMP_SCHEDULE      controls OMP schedule policy at runtime (v5)"
	@echo "                    e.g. 'static', 'guided,4', 'dynamic,1'"
	@echo "  OMP_PROC_BIND     consider 'spread' for long campaigns"
