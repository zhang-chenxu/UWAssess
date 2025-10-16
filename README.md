# Datasets preparation

## Download datasets
Datasets downloading URL:
    
| Dataset Name | Link | Access |
|-----|---------------|--------|
| RoadwayFlooding | https://data.mendeley.com/datasets/t395bwcvbw/1 | Open Access |
| UWBenchV2 | https://github.com/zhang-chenxu/LSM-Adapter | Credentialed Access |

## Prepare datasets
After downloading the above dataset to /test_set/, please use the corresponding processing code in /dataset_preparation/ for it.


# Urban waterlogging assessment

## Prepare checkpoints
Download the [Adapted SAM2 Model](https://drive.google.com/file/d/198xdKX893UYHc7WN7Z-JT9cui5UETiKW/view?usp=drive_link) to /checkpoints/.

Download the [DeepSeek-VL2 Model](https://github.com/deepseek-ai/DeepSeek-VL2) to the current path.

## Testing
For urban waterlogging assessment task, you can use the following command to test any image (replace your_test_image_path with the path to your test images):
```shell
python test.py --test_path your_test_image_path
```

For visual perception and performance evaluation only, the following command can be used (replace your_test_dataset_path with the path to your test dataset):
```shell
python test_vision.py --test_path your_test_dataset_path
```

For report scoring, the following command can be used (modify your_api_secret_key and your_base_url):
```shell
python report_scoring.py --API_SECRET_KEY your_api_secret_key --BASE_URL your_base_url
```
