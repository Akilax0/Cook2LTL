# AnyGrasp Installation Guide

**Environment:** Ubuntu, CUDA 12.8, Python 3.11, Conda

**Tested:** March 2026

---

## Step 1: Create Conda Environment

```bash
conda deactivate
conda env remove -n anygrasp -y

conda create -n anygrasp python=3.11 -y
conda activate anygrasp
```

## Step 2: Pin setuptools (v82+ removes pkg_resources)

```bash
pip install setuptools==75.0.0 wheel

# Verify
python -c "import pkg_resources; print('pkg_resources: OK')"
```

> **Why:** setuptools >= 82 removed `pkg_resources` from the main package.
> Many AnyGrasp dependencies (graspnetAPI, MinkowskiEngine, etc.) still need it.

## Step 3: Install CUDA Toolkit + OpenBLAS in Conda

```bash
conda install nvidia/label/cuda-12.8.0::cuda-toolkit -y
conda install nvidia/label/cuda-12.8.0::cuda-nvtx -y
conda install openblas-devel -c anaconda -y

export CUDA_HOME=$CONDA_PREFIX
```

```bash
# Verify
nvcc --version
# Should show 12.8
```

## Step 4: Install PyTorch (matching CUDA 12.8)

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

```bash
# Verify
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}, GPU: {torch.cuda.is_available()}')"
# Should show CUDA 12.8, GPU True
```

## Step 5: Install Python Dependencies

```bash
pip install numpy==1.24.4 pandas scipy pillow tqdm opencv-python transforms3d open3d trimesh
```

## Step 6: Install graspnetAPI

```bash
pip install graspnetAPI --no-deps
```

> **Why `--no-deps`:** graspnetAPI tries to install an old pandas (1.4.4) which
> fails to build on Python 3.11+ due to missing `distutils.msvccompiler`.
> We already installed pandas and all other deps manually in Step 5.

## Step 7: Build MinkowskiEngine

This is the hardest step. Multiple patches are needed for CUDA 12.8 compatibility.

### 7a. Clone and checkout

```bash
export CUDA_HOME=$CONDA_PREFIX
export MAX_JOBS=2

mkdir -p dependencies && cd dependencies
git clone https://github.com/chenxi-wang/MinkowskiEngine.git
cd MinkowskiEngine
git checkout cuda-12-1
```

### 7b. Stub out broken NVTX headers

The bundled NVTX/CUDF headers are incompatible with CUDA 12.8. Since NVTX is
only used for profiling and not needed for inference, we replace them with no-op stubs.

```bash
# Stub nvtx3.hpp
cat > src/3rdparty/cudf/detail/nvtx/nvtx3.hpp << 'EOF'
#pragma once
#define NVTX3_FUNC_RANGE()
#define NVTX3_FUNC_RANGE_IN(domain)
namespace nvtx3 {
template<typename T> struct domain_thread_range {};
}
EOF

# Stub ranges.hpp (adds CUDF_FUNC_RANGE macro)
cat > src/3rdparty/cudf/detail/nvtx/ranges.hpp << 'EOF'
#pragma once
namespace cudf {
struct libcudf_domain {};
using thread_range = struct {};
}
#define NVTX3_FUNC_RANGE_IN(domain)
#define NVTX3_FUNC_RANGE()
#define CUDF_FUNC_RANGE()
EOF
```

> **What these fix:**
> - `fatal error: nvtx3/nvToolsExt.h: No such file or directory`
> - `error: namespace "nvtx3" has no member "domain_thread_range"`
> - `error: identifier "NVTX3_FUNC_RANGE_IN" is undefined`
> - `error: identifier "CUDF_FUNC_RANGE" is undefined`

### 7c. Fix __to_address ambiguity (CUDA 12.8 + GCC 11 conflict)

CUDA 12.8 introduces `cuda::std::__to_address` which conflicts with `std::__to_address`
in GCC 11's `shared_ptr_base.h`. We need to explicitly qualify the call.

```bash
# Find the header in your conda environment
HEADER=$(find $CONDA_PREFIX -name "shared_ptr_base.h" -path "*/c++/11*/bits/*" 2>/dev/null | head -1)
echo "Patching: $HEADER"

# Back up
cp "$HEADER" "${HEADER}.bak"

# Fix: qualify __to_address with std:: namespace
sed -i 's/auto __raw = __to_address(__r.get());/auto __raw = std::__to_address(__r.get());/' "$HEADER"

# Verify
grep "std::__to_address" "$HEADER" | head -3
```

> **What this fixes:**
> - `error: more than one instance of overloaded function "std::__to_address" matches the argument list`

### 7d. Build

```bash
rm -rf build/

python setup.py install \
    --blas_include_dirs=${CONDA_PREFIX}/include \
    --blas_library_dirs=${CONDA_PREFIX}/lib \
    --blas=openblas
```

> **If you get `ld: cannot find -lopenblas`:**
> ```bash
> cp ${CONDA_PREFIX}/lib/libopenblas.so* $(python -c "import torch; print(torch.__path__[0])")/lib/
> ```
> Then retry the build command.

> **If you get ninja errors:**
> ```bash
> pip install ninja
> ```
> Or edit `setup.py` and change `use_ninja=True` to `use_ninja=False` in the `BuildExtension` call.

### 7e. Verify

```bash
cd ../..
python -c "import MinkowskiEngine; print(f'MinkowskiEngine: {MinkowskiEngine.__version__}')"
```

## Step 8: Clone AnyGrasp SDK + Install Requirements

```bash
git clone git@github.com:graspnet/anygrasp_sdk.git
cd anygrasp_sdk
pip install -r requirements.txt
```

## Step 9: Install PointNet2

```bash
export CUDA_HOME=$CONDA_PREFIX

cd pointnet2
python setup.py install
cd ..
```

> **If CUDA version mismatch error:**
> Make sure `CUDA_HOME=$CONDA_PREFIX` is set so it uses the conda CUDA 12.8,
> not whatever system CUDA is installed.

```bash
# Verify
python -c "import pointnet2; print('pointnet2: OK')"
```

## Step 10: Get License

AnyGrasp requires a per-machine license.

```bash
cd license_registration
python license_checker.py -f
```

This prints your machine's `feature_id`. Submit it at: https://graspnet.net/anygrasp.html

They will email you a `license/` folder.

> **Important:** The license is tied to your machine ID. If using Docker, pin
> the network config — the machine ID changes with network changes.

## Step 11: Install License + Checkpoint

```bash
# Copy license to both detection and tracking dirs
cp -r license/ grasp_detection/
cp -r license/ grasp_tracking/

# Place checkpoint (from download link in license email)
# grasp_detection/log/checkpoint_detection.tar
```

## Step 12: Test

```bash
cd grasp_detection
python demo.py --debug
```

Should show grasp detections on sample data with Open3D visualization.

---

## Quick Reference: Environment Variables

Always set these before building anything in the environment:

```bash
conda activate anygrasp
export CUDA_HOME=$CONDA_PREFIX
export MAX_JOBS=2
export CPLUS_INCLUDE_PATH=$CONDA_PREFIX/include:$CPLUS_INCLUDE_PATH
export C_INCLUDE_PATH=$CONDA_PREFIX/include:$C_INCLUDE_PATH
```

## Troubleshooting Summary

| Error | Fix |
|-------|-----|
| `No module named 'pkg_resources'` | `pip install setuptools==75.0.0` |
| `No module named 'distutils.msvccompiler'` | Use Python 3.11 (not 3.12+) |
| `nvtx3/nvToolsExt.h: No such file or directory` | Stub NVTX headers (Step 7b) |
| `namespace "nvtx3" has no member "domain_thread_range"` | Stub NVTX headers (Step 7b) |
| `identifier "CUDF_FUNC_RANGE" is undefined` | Add `#define CUDF_FUNC_RANGE()` to ranges.hpp stub (Step 7b) |
| `more than one instance of "std::__to_address"` | Patch shared_ptr_base.h (Step 7c) |
| `ld: cannot find -lopenblas` | Copy libopenblas.so to torch lib dir (Step 7d note) |
| CUDA version mismatch (PointNet2) | `export CUDA_HOME=$CONDA_PREFIX` |
| `Failed to build 'pandas'` | Install pandas separately, use `--no-deps` for graspnetAPI |
