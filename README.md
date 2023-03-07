# Image Quantization

This repository includes code to quantize an image to the specified number of colors.
Initial colors can be picked from the interactive window or randomly.

**Usage**
```
-h, --help            show this help message and exit
-file FILE            path to input file
--cluster_count CLUSTER_COUNT
                      total number of clusters (default 5)
--max_iter MAX_ITER   maximum number of iteration until convergence (default 10)
--random_init         cluster centers will be randomly initialized. Do not set this flag to choose colors from the interactive window
--save                processed image will be saved
--no_display          processed image will not be displayed

```

**Example Command**
```
python main.py -file sample.jpg --save --random_init --cluster_count 10
```

Input             |  Quantized (#C=2) |  Quantized (#C=5) |  Quantized (#C=10)
:-------------------------:|:-------------------------:|:-------------------------:|:-------------------------:
![](https://github.com/mervekantarci/image_quantization/blob/main/sample.jpg)  |  ![](https://github.com/mervekantarci/image_quantization/blob/main/sample_quantized_2.jpg) |  ![](https://github.com/mervekantarci/image_quantization/blob/main/sample_quantized_5.jpg)|  ![](https://github.com/mervekantarci/image_quantization/blob/main/sample_quantized_10.jpg)

