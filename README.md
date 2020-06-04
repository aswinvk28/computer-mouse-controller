# Computer Pointer Controller

*TODO:* Write a short introduction to your project

To move the mouse pointer based on the head pose, facial landmarks that use eye coordination, detected face and gaze estimation.

## Project Set Up and Installation
*TODO:* Explain the setup procedures to run your project. For instance, this can include your project directory structure, the models you need to download and where to place them etc. Also include details about how to install the dependencies your project requires.

```bash

    pip install virtualenv

    virtualenv .

    pip install -r requirements.txt

```

__Downloading OpenVINO Models__

```bash

    python /opt/intel/openvino/deployment_tools/tools/model_downloader/downloader.py --name face-detection-adas-binary-0001 --output_dir ./models/

    python /opt/intel/openvino/deployment_tools/tools/model_downloader/downloader.py --name head-pose-estimation-adas-0001 --output_dir ./models/

    python /opt/intel/openvino/deployment_tools/tools/model_downloader/downloader.py --name landmarks-regression-retail-0009 --output_dir ./models/

    python /opt/intel/openvino/deployment_tools/tools/model_downloader/downloader.py --name gaze-estimation-adas-0002 --output_dir ./models/

```

## Demo
*TODO:* Explain how to run a basic demo of your model.

```bash

    python exec.py --models="face,head_pose,facial_landmarks,gaze_estimation" --input_file "bin/demo.mp4" --device "CPU"

```

## Documentation
*TODO:* Include any documentation that users might need to better understand your project code. For instance, this is a good place to explain the command line arguments that your project supports.

The `exec.py` file has used `concurrent.futures.ThreadPoolExecutor`

The models used are:

- Face Detection

- Face Landmarks Detection

- Head Pose Estimation

- Gaze Estimation

## Benchmarks
*TODO:* Include the benchmark results of running your model on multiple hardwares and multiple model precisions. Your benchmarks can include: model loading time, input/output processing time, model inference time etc.

```bash

    cd scripts/

    # Face Detection

    python inference_model.py --model "face-detection-adas-binary-0001" --precision "FP16" --device "CPU" --image "images/image1.png" --prefix "/home/workspace/ir_models/intel/"

    # Face Landmarks Detection

    python inference_model.py --model "facial-landmarks-35-adas-0002" --precision "FP16" --device "CPU" --image "images/image1.png" --prefix "/home/workspace/ir_models/intel/"

    # Gaze Estimation Model

    python inference_model.py --model "gaze-estimation-adas-0002" --precision "FP16" --device "CPU" --image "images/image1.png" --prefix "/home/workspace/ir_models/intel/"

    # Head Pose Estimation Model

    python inference_model.py --model "head-pose-estimation-adas-0001" --precision "FP16" --device "CPU" --image "images/image1.png" --prefix "/home/workspace/ir_models/intel/"

```

## Results
*TODO:* Discuss the benchmark results and explain why you are getting the results you are getting. For instance, explain why there is difference in inference time for FP32, FP16 and INT8 models.

### Face Detection

`FP32`: This is a high precision model, so there will be higher inference time due to improved requirements for accuracy

`FP16`: 

`INT8`: 

### Face Landmarks Detection

`FP32`: This is a high precision model, so there will be higher inference time due to improved requirements for accuracy

`FP16`: 

`INT8`: 

### Gaze Estimation

`FP32`: This is a high precision model, so there will be higher inference time due to improved requirements for accuracy

`FP16`: 

`INT8`: 

### Head Pose Estimation

`FP32`: This is a high precision model, so there will be higher inference time due to improved requirements for accuracy

`FP16`: 

`INT8`: 

## Stand Out Suggestions
This is where you can provide information about the stand out suggestions that you have attempted.



### Async Inference
If you have used Async Inference in your code, benchmark the results and explain its effects on power and performance of your project.



### Edge Cases
There will be certain situations that will break your inference flow. For instance, lighting changes or multiple people in the frame. Explain some of the edge cases you encountered in your project and how you solved them to make your project more robust.

