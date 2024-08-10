# ReadMe

This is the repository for the *Remote Sensing of Environment* article: *Deep Point Cloud Regression for
Above-Ground Forest Biomass Estimation from Airborne LiDAR*.

When cloning the repository, make sure to also get submodules:
```
git clone --recurse-submodules https://github.com/greenipy/DPCR-AGB.git
```

We include **code**, **evaluation scripts**, **model weights** (soon), and the **dataset**.

Regarding the code:
We forked the [torch-points3d](https://github.com/nicolas-chaulet/torch-points3d) framework and added support for
regression tasks including datasets, tracking, and models on our own. In the process, we also simplified the usage of
package.

In addition, we also included our code to load the trained linear regression and random forest in
the `pointcloud_stats_method` folder. Just run the notebook `learn_with_stats.ipynb`.

Finally, the results/plots for each method can be seen in the `eval_scripts` folder within
the `eval_deep_learning_v2.ipynb`. The results for the network size experiment are in `eval_deep_learning_v2_size.ipynb`.

**results on the test set:**

| target          | model    | treeadd |    $R^2$ |       |     RMSE |         |      MAPE |           | mean bias |          |
|:----------------|:---------|:--------|---------:|------:|---------:|--------:|----------:|----------:|----------:|---------:|
|                 |          |         | *median* | *max* | *median* |   *min* |  *median* |     *min* |  *median* |    *min* |
| **biomass**     | KPConv   | False   |    0.800 | 0.815 |   45.264 |  43.540 |   396.685 |   272.288 |     0.460 |    0.389 |
|                 |          | True    |    0.780 | 0.803 |   47.526 |  44.975 |   467.581 |   246.927 |     3.660 |   -0.707 |
|                 | MSENet14 | False   |    0.825 | 0.829 |   42.373 |  41.806 |   299.497 |   192.777 |     0.666 |   -0.291 |
|                 |          | True    |    0.823 | 0.829 |   42.596 |  41.851 |   271.716 |   131.120 |     0.313 |    0.122 |
|                 | MSENet50 | False   |    0.827 | 0.835 |   42.140 |  41.083 |   469.104 |   174.245 |     0.837 |   -0.114 |
|                 |          | True    |    0.824 | 0.837 |   42.481 |  40.909 |   339.700 |   119.264 |     0.889 |    0.596 |
|                 | PointNet | False   |    0.770 | 0.772 |   48.565 |  48.288 |   889.293 |   625.091 |     0.539 |    0.119 |
|                 |          | True    |    0.766 | 0.768 |   48.932 |  48.753 |   896.835 |   622.713 |     2.464 |    1.774 |
|                 | RF       | False   |    0.754 | 0.754 |   50.188 |  50.158 |   625.439 |   616.635 |     1.470 |    1.459 |
|                 |          | True    |    0.151 | 0.157 |   93.238 |  92.930 |  7644.787 |  7423.094 |    47.625 |  -47.521 |
|                 | power    | False   |    0.761 | 0.761 |   49.509 |  49.509 |   365.606 |   365.606 |     2.027 |    2.027 |
|                 |          | True    |    0.034 | 0.034 |   99.478 |  99.478 |  7604.844 |  7604.844 |    57.525 |  -57.525 |
|                 | linear   | False   |    0.762 | 0.762 |   49.420 |  49.420 |   425.605 |   425.605 |     1.894 |    1.894 |
|                 |          | True    |    0.195 | 0.195 |   90.801 |  90.801 | 11448.501 | 11448.501 |    39.149 |  -39.149 |
| **wood volume** | KPConv   | False   |    0.799 | 0.805 |   85.434 |  84.255 |   103.866 |    85.633 |     0.377 |    0.285 |
|                 |          | True    |    0.778 | 0.792 |   89.808 |  87.002 |   126.543 |    85.812 |     7.885 |   -1.012 |
|                 | MSENet14 | False   |    0.823 | 0.826 |   80.309 |  79.631 |    99.105 |    72.597 |     0.515 |    0.389 |
|                 |          | True    |    0.821 | 0.825 |   80.750 |  79.716 |    84.473 |    70.097 |     2.577 |    1.829 |
|                 | MSENet50 | False   |    0.824 | 0.831 |   79.986 |  78.344 |   131.525 |    72.381 |     0.169 |    0.123 |
|                 |          | True    |    0.822 | 0.832 |   80.571 |  78.177 |   115.634 |    78.422 |     3.572 |    2.646 |
|                 | PointNet | False   |    0.777 | 0.781 |   90.183 |  89.198 |   205.366 |   162.049 |     1.991 |    1.369 |
|                 |          | True    |    0.773 | 0.776 |   90.844 |  90.220 |   236.383 |   174.903 |     5.708 |    4.578 |
|                 | RF       | False   |    0.757 | 0.757 |   94.091 |  94.070 |   223.652 |   222.600 |     3.979 |    3.955 |
|                 |          | True    |    0.192 | 0.197 |  171.475 | 170.930 |  1683.778 |  1676.524 |    85.629 |  -85.465 |
|                 | power    | False   |    0.763 | 0.763 |   92.819 |  92.819 |   223.654 |   223.654 |     4.497 |    4.497 |
|                 |          | True    |    0.120 | 0.120 |  178.973 | 178.973 |  1793.822 |  1793.822 |   101.104 | -101.104 |
|                 | linear   | False   |    0.766 | 0.766 |   92.292 |  92.292 |   171.483 |   171.483 |     4.602 |    4.602 |
|                 |          | True    |    0.243 | 0.243 |  166.034 | 166.034 |  1747.807 |  1747.807 |    72.340 |  -72.340 |

# Install torch-points3d

We setup our environment in the following way (conda is already installed):

1. go to `torch-points3d`
2. Make sure to install cuda 11.8 (don't forget to deselect the driver install if your drivers are current)
3. [cuda-toolkit](https://developer.nvidia.com/cuda-12-4-0-download-archive?target_os=Linux&target_arch=x86_64&Distribution=Debian&target_version=12&target_type=deb_local)

```
wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-wsl-ubuntu.pin
sudo mv cuda-wsl-ubuntu.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda-repo-wsl-ubuntu-11-8-local_11.8.0-1_amd64.deb
sudo dpkg -i cuda-repo-wsl-ubuntu-11-8-local_11.8.0-1_amd64.deb
sudo cp /var/cuda-repo-wsl-ubuntu-11-8-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cuda
```

3. after installing close and reopen the terminal to check if the PATH is set correctly with `echo $PATH`. It should
   **not** have `/usr/local/cuda-10.2` but should have something like `/usr/local/cuda-11.8` in there

5. install mamba (optional but highly recommended)

```
curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
bash Miniforge3-$(uname)-$(uname -m).sh
```

3. create conda environment:

```
mamba env create -f env.yml
```

or for cpu-version:

```
mamba env create -f env_cpu.yml
```

4. activate environment:

```
mamba activate pts
```

5. install missing pip packages for Minkowski networks (Windows is currently not supported, Failed: 2024-08-10).

* [Can't install with CUDA 12.1](https://github.com/NVIDIA/MinkowskiEngine/issues/543)
* [fix build with CUDA 12.2]([https://github.com/NVIDIA/MinkowskiEngine/issues/543](https://github.com/NVIDIA/MinkowskiEngine/pull/567))
* [Compilation: compiler finds both std::to_address and cuda::std::to_address]([https://github.com/NVIDIA/MinkowskiEngine/issues/596))

```
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
# mamba install cuda -c nvidia/label/cuda-12.4.0
# sudo apt install ninja-build cmake generate-ninja
# git clone https://github.com/NVIDIA/MinkowskiEngine.git
# cd MinkowskiEngine
# export CUDA_HOME=$(dirname $(dirname $(which nvcc)));
# sudo apt install libopenblas-dev
# export TORCH_CUDA_ARCH_LIST="8.9"
# unset CXX
mamba install nvcc_linux-64
python setup.py install --blas_include_dirs=${CONDA_PREFIX}/include --blas=openblas
# pip install -U git+https://github.com/NVIDIA/MinkowskiEngine -v --no-deps --config-settings="--blas_include_dirs=${CONDA_PREFIX}/include" --config-settings="--blas=openblas"

```

Installation error with CUDA below:

```
# WSL2
# Debian 12 (bookworm)
# Cuda 12.4

/usr/include/c++/12/bits/shared_ptr_base.h(1561): error: more than one instance of overloaded function "std::__to_address" matches the argument list:
            function template "_Tp *cuda::std::__4::__to_address(_Tp *) noexcept" (declared at line 277 of /usr/local/cuda-12.4/targets/x86_64-linux/include/cuda/std/detail/libcxx/include/__memory/pointer_traits.h)
            function template "_Tp *std::__to_address(_Tp *) noexcept" (declared at line 228 of /usr/include/c++/12/bits/ptr_traits.h)

/usr/include/c++/12/bits/shared_ptr_base.h(1563): error: no instance of overloaded function "std::__shared_ptr<_Tp, _Lp>::_M_enable_shared_from_this_with [with _Tp=concurrent_unordered_map<minkowski::coordinate<int32_t>, uint32_t, minkowski::detail::coordinate_murmur3<int32_t>, minkowski::detail::coordinate_equal_to<int32_t>, minkowski::detail::default_allocator<cuda::std::__4::pair<minkowski::coordinate<int32_t>, uint32_t>>>, _Lp=__gnu_cxx::_S_atomic]" matches the argument list      
            argument types are: (<error-type>)
     _M_enable_shared_from_this_with(__raw);

/usr/include/c++/12/bits/shared_ptr_base.h(1561): error: more than one instance of overloaded function "std::__to_address" matches the argument list:
            function template "_Tp *cuda::std::__4::__to_address(_Tp *) noexcept" (declared at line 277 of /usr/local/cuda-12.4/targets/x86_64-linux/include/cuda/std/detail/libcxx/include/__memory/pointer_traits.h)
            function template "_Tp *std::__to_address(_Tp *) noexcept" (declared at line 228 of /usr/include/c++/12/bits/ptr_traits.h)

/usr/include/c++/12/bits/shared_ptr_base.h(1563): error: no instance of overloaded function "std::__shared_ptr<_Tp, _Lp>::_M_enable_shared_from_this_with [with _Tp=concurrent_unordered_map<minkowski::coordinate<int32_t>, uint32_t, minkowski::detail::coordinate_murmur3<int32_t>, minkowski::detail::coordinate_equal_to<int32_t>, minkowski::detail::c10_allocator<cuda::std::__4::pair<minkowski::coordinate<int32_t>, uint32_t>>>, _Lp=__gnu_cxx::_S_atomic]" matches the argument list
            argument types are: (<error-type>)
     _M_enable_shared_from_this_with(__raw);

```


or for cpu-version:

```
mamba uninstall pytorch torchvision torchaudio cpuonly -c pytorch
mamba uninstall cudatoolkit
xargs mamba remove -y
cd conf
grep -Ril "CUDA" ./        # change all to -1
mamba list | grep sym |
conda install pytorch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 cpuonly -c pytorch
# pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cpu
pip uninstall sympy
mamba install 'pyg=2.5.2[build=py38_torch_2.2.0_cpu]' 'pytorch-cluster=1.6.3[build=py38_torch_2.2.0_cpu]' 'pytorch-scatter=2.1.2[build=py38_torch_2.2.0_cpu]' -c pyg -c pytorch -c conda-forge
cd MinkowskiEngine # 0.5.4
python setup.py install --blas_include_dirs=${CONDA_PREFIX}/include --blas=openblas --cpu_only
pip install -U git+https://github.com/NVIDIA/MinkowskiEngine -v --no-deps --config-settings="--blas_include_dirs=${CONDA_PREFIX}/include" --config-settings="--blas=openblas"

```

Install missing packages for cpu-version:

```
# ModuleNotFoundError: No module named 'torch_geometric'
# ModuleNotFoundError: No module named 'torch_cluster'
# ModuleNotFoundError: No module named 'torch_scatter'
pip install torch_geometric torch_cluster torch_scatter
# ImportError: libtiff.so.5: cannot open shared object file: No such file or directory
sudo apt install libtiff5-dev
sudo ln -s /usr/lib/x86_64-linux-gnu/libtiff.so ./libtiff.so.5 
```

5. compile KPConv scripts

```
sh compile_wrappers.sh
```

# Get the Data

to get the preprocessed lidar and nfi data, go to the torch-points3d folder (`cd torch-points3d`)  and download:

```
wget https://sid.erda.dk/share_redirect/bB1TBPTsEk
mv bB1TBPTsEk nfi_preprocessed_data.zip
unzip nfi_preprocessed_data.zip
```
the data should now be in `data/biomass/processed_nfi_reg/` given the root folder is torch-points3d.

# Training for Regression

run from within the torch-points3d folder.

*MSENet50:*

```
python -u train.py task=instance models=instance/minkowski_baseline model_name=SENet50 data=instance/NFI/reg data.transform_type=sparse_xy training=nfi/minkowski lr_scheduler=cosineawr update_lr_scheduler_on=on_num_batch
```

*MSENet14:*

```
python -u train.py task=instance models=instance/minkowski_baseline model_name=SENet14 data=instance/NFI/reg data.transform_type=sparse_xy training=nfi/minkowski lr_scheduler=cosineawr update_lr_scheduler_on=on_num_batch
```

*KPConv:*

```
python -u train.py task=instance models=instance/kpconv model_name=KPConv data=instance/NFI/reg training=nfi/kpconv data.transform_type=xy lr_scheduler=cosineawr update_lr_scheduler_on=on_num_batch
```

*PointNet:*

```
python -u train.py task=instance models=instance/minkowski_baseline model_name=MPointNet data=instance/NFI/reg training=nfi/pointnet data.transform_type=sparse_xy lr_scheduler=cosineawr update_lr_scheduler_on=on_num_batch
```

# Calibration batch normalization

to calibrate the trained models batch norm statistics. Note that the checkpoint directory has to be an absolute path,
e.g.: `checkpoint_dir=/home/user/torch-points3d/weights/SENet50/0`

for Minkowski or Pointnet (`model_name=SENet50`, `model_name=SENet14`, or `model_name=MPointNet`):

```
python calibrate_bn.py model_name=${model_name} checkpoint_dir=${checkpoint_dir} data=instance/NFI/reg num_workers=4 task=instance weight_name="total_BMag_ha_rmse" batch_size=64 num_workers=4 data.transform_type=sparse_xy epochs=20
```

for KPConv:

```
python calibrate_bn.py model_name=KPConv checkpoint_dir=${checkpoint_dir} data=instance/NFI/reg num_workers=4 task=instance weight_name="total_BMag_ha_rmse" batch_size=64 num_workers=4 data.transform_type=xy epochs=20
```

# Evaluating our models

run from within the torch-points3d folder. Note that the checkpoint directory has to be an absolute path,
e.g.: `PATHTOFRAMEWORK=/home/user/torch-points3d`
Also, there are 5 weights for each model (from different trials): `TRIAL=1`

*MSENet50:*

```
python eval.py model_name=SENet50 checkpoint_dir=${PATHTOFRAMEWORK}/weights/SENet50/${TRIAL}/ weight_name="latest" batch_size=32 num_workers=4 eval_stages=["train","val","test"] data.transform_type=sparse_xy_eval data=instance/NFI/reg task=instance
```

the save folder location is `weights/msenet50/eval`.

*MSENet14:*

```
python eval.py model_name=SENet14 checkpoint_dir=${PATHTOFRAMEWORK}/weights/SENet14/${TRIAL}/ weight_name="latest" batch_size=32 num_workers=4 eval_stages=["train","val","test"] data.transform_type=sparse_xy_eval data=instance/NFI/reg task=instance
```

the save folder location is `weights/msenet14/eval`.

*KPConv:*

```
python eval.py model_name=KPConv checkpoint_dir=${PATHTOFRAMEWORK}/weights/KPConv/${TRIAL}/ weight_name="latest" batch_size=32 num_workers=4 eval_stages=["train","val","test"] data.transform_type=xy_eval data=instance/NFI/reg task=instance
```

the save folder location is `weights/kpconv/eval`.

*PointNet:*

```
python eval.py model_name=MPointNet checkpoint_dir=${PATHTOFRAMEWORK}/weights/PointNet/${TRIAL}/ weight_name="latest" batch_size=32 num_workers=4 eval_stages=["train","val","test"] data.transform_type=sparse_xy_eval data=instance/NFI/reg task=instance
```

the save folder location is `weights/pointnet/eval`.

# Using tree-adding augmentations during test

same as before, but the transform type changes to use tree augmentations, e.g.:

```
python eval.py model_name=MPointNet checkpoint_dir=${PATHTOFRAMEWORK}/weights/pointnet/ weight_name="total_rmse" batch_size=32 num_workers=4 eval_stages=["train","val","test"] data.transform_type=sparse_xy_eval_treeadd
```
