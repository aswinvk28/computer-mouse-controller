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

    # Face Detection, Head Pose Estimation, Facial Landmarks Detection, Gaze Estimation

    python inference_model.py --models "face,head_pose,facial_landmarks,gaze_estimation" --precision "FP16" --device "CPU" --image "../images/image1.png" --prefix "/home/workspace/ir_models/intel/" --iterations 100

    Model...:  face
    Model Load Time is:  0.2290022373199463
    Inference Time is:  0.0218103551864624
    Input/Output Time is:  0.0008311271667480469

    Model...:  head_pose
    Model Load Time is:  0.09927248954772949
    Inference Time is:  0.0018021202087402345
    Input/Output Time is:  0.00013875961303710938

    Model...:  facial_landmarks
    Model Load Time is:  0.42183756828308105
    Inference Time is:  0.003168463706970215
    Input/Output Time is:  0.00013375282287597656
    
    Model...:  gaze_estimation
    Model Load Time is:  0.13030791282653809
    Inference Time is:  0.0023927545547485353
    Input/Output Time is:  0.00022840499877929688

```

## Results
*TODO:* Discuss the benchmark results and explain why you are getting the results you are getting. For instance, explain why there is difference in inference time for FP32, FP16 and INT8 models.

### Face Detection, Face Landmarks Detection, Gaze Estimation, Head Pose Estimation

```log

    Precision:  FP16
    Model...:  face
    Model Load Time is:  1.2867493629455566
    Inference Time is:  0.0026137804985046386
    Input/Output Time is:  0.017719745635986328

    Model...:  head_pose
    Model Load Time is:  0.7920746803283691
    Inference Time is:  0.003512754440307617
    Input/Output Time is:  0.00013685226440429688

    Model...:  facial_landmarks
    Model Load Time is:  0.2831246852874756
    Inference Time is:  0.002412700653076172
    Input/Output Time is:  0.0002562999725341797

```

```log

    Precision:  FP32
    Model...:  face
    Model Load Time is:  0.30683088302612305
    Inference Time is:  0.002059483528137207
    Input/Output Time is:  0.00015091896057128906

    Model...:  head_pose
    Model Load Time is:  0.6290264129638672
    Inference Time is:  0.003314807415008545
    Input/Output Time is:  0.00013494491577148438

    Model...:  facial_landmarks
    Model Load Time is:  0.33220458030700684
    Inference Time is:  0.0025447630882263184
    Input/Output Time is:  0.00022602081298828125

```

```log

    Precision:  INT1
    Model...:  face
    Model Load Time is:  0.31473755836486816
    Inference Time is:  0.017162725925445557
    Input/Output Time is:  0.0005006790161132812

```

```log

    Precision:  INT8
    Model...:  face
    Model Load Time is:  0.2635071277618408
    Inference Time is:  0.0015732979774475098
    Input/Output Time is:  0.00023293495178222656

```

## Stand Out Suggestions
This is where you can provide information about the stand out suggestions that you have attempted.



### Async Inference
If you have used Async Inference in your code, benchmark the results and explain its effects on power and performance of your project.



### Edge Cases
There will be certain situations that will break your inference flow. For instance, lighting changes or multiple people in the frame. Explain some of the edge cases you encountered in your project and how you solved them to make your project more robust.
