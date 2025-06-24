# ConViTCast: Continuous Vision Transformer for Weather Forecasting

![architecture](images/weather.png)

# Requirements:
* python=3.8
* pytorch=1.12.1
* pytorch-lightning==1.8.0
* [torchdiffeq](https://github.com/rtqichen/torchdiffeq)
* zarr=2.13.6
* xarray
* dask[complete]
* netcdf4=1.6.2
* cartopy (for Visualization)
* [ClimaX](https://microsoft.github.io/ClimaX/) [1] (As the code follows coding practice of ClimaX)

# Data Preparation
First, download ERA5 data from [WeatherBench](https://dataserv.ub.tum.de/index.php/s/m1524895?path=%2F). Select the resolution of the data you want to download. Preprocess the netcdf data into small numpy file:

```python
python data/nc2np.py \
    --root_dir <path/to/data directory> \
    --save_dir <path/to/preprocess output directory> \
    --start_train_year 1979 --start_val_year 2018 \
    --start_test_year 2019 --end_year 2021 --num_shards 8
```
## [WeatherBench2](https://console.cloud.google.com/storage/browser/weatherbench2?pageState=(%22StorageObjectListTable%22:(%22f%22:%22%255B%255D%22)))
Download the WeatherBench2 data at 1.5 resolution using google cloud.

* First Download and install the Google Cloud SDK
```python
curl https://sdk.cloud.google.com | bash
```
* Initialize the SDK
```python
gcloud init
```
* Authenticate with Google Cloud
```python
gcloud auth login
```
* Download data
```python
gsutil -m cp -r "gs://weatherbench2/datasets/era5/1959-2022-6h-240x121_equiangular_with_poles_conservative.zarr" \
    <destination_folder>
```
* Convert zarr array to netcdf and regrid it to the required resolution
```python
python data/weatherbench2.py
python data/regrid.py
```

# Training
To train ConViTCast for Global Forecasting, use
```python
python train.py --config <path/to/config>
```

```
[1] Nguyen, T., Brandstetter, J., Kapoor, A., Gupta, J.K. and Grover, A., 2023. Climax: A foundation model for weather and climate. arXiv preprint arXiv:2301.10343.
```
Note: We are also constantly updating the repo to make it accustomed to WeatherBench2, and removing bugs and modifying certain parts.
