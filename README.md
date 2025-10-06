
---

## Requirements

- GCC compiler
- `make`
- `gprof` (for profiling)
- Input files (`img0.pgm`, `img1.pgm`, ...) should be in `src/V1/`  

---

## Compile, Run, and Profile Example3

You can use the following **single copy-paste command** to compile everything, run `example3`, and generate the profiling report:

```bash
# Compile the library and examples with profiling enabled
make

# Run example3 to generate the gmon.out profiling file
./src/V1/example3

# Generate gprof report in src/V1/profile_example3.txt
gprof ./src/V1/example3 src/V1/gmon.out > profile_example3.txt

# Open the profile report
less profile_example3.txt
