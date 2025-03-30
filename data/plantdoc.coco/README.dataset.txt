# PlantDoc > resize-416x416
https://public.roboflow.ai/object-detection/plantdoc

Provided by [Singh et. al 2019](https://arxiv.org/pdf/1911.10317.pdf)
License: CC BY 4.0

# Overview

The PlantDoc dataset was originally published by researchers at the Indian Institute of Technology, and described in depth in [their paper](https://arxiv.org/pdf/1911.10317.pdf). One of the paper’s authors, Pratik Kayal, shared the object detection dataset available [on GitHub](https://github.com/pratikkayal/PlantDoc-Dataset).

 PlantDoc is a dataset of 2,569 images across 13 plant species and 30 classes (diseased and healthy) for image classification and object detection. There are 8,851 labels. Read more about how the version available on Roboflow improves on the original version [here](https://blog.roboflow.ai/introducing-an-improved-plantdoc-dataset-for-plant-disease-object-detection/).

And here's an example image:

![Tomato Blight](https://i.imgur.com/fGlQ0kG.png)

`Fork` this dataset (upper right hand corner) to receive the raw images, or (to save space) grab the 416x416 export.

# Use Cases

As the researchers from IIT stated in their paper, “plant diseases alone cost the global economy around US$220 billion annually.” Training models to recognize plant diseases earlier dramatically increases yield potential.

The dataset also serves as a useful open dataset for benchmarks. The researchers trained both object detection models like MobileNet and Faster-RCNN and image classification models like VGG16, InceptionV3, and InceptionResnet V2.

The dataset is useful for advancing general agriculture computer vision tasks, whether that be health crop classification, plant disease classification, or plant disease objection. 

# Using this Dataset

This dataset follows [Creative Commons 4.0 protocol](https://creativecommons.org/licenses/by/4.0/). You may use it commercially without Liability, Trademark use, Patent use, or Warranty.

Provide the following citation for the original authors:

```
@misc{singh2019plantdoc,
    title={PlantDoc: A Dataset for Visual Plant Disease Detection},
    author={Davinder Singh and Naman Jain and Pranjali Jain and Pratik Kayal and Sudhakar Kumawat and Nipun Batra},
    year={2019},
    eprint={1911.10317},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```

# About Roboflow

[Roboflow](https://roboflow.ai) makes managing, preprocessing, augmenting, and versioning datasets for computer vision seamless.

Developers reduce 50% of their code when using Roboflow's workflow, automate annotation quality assurance, save training time, and increase model reproducibility.

#### [![Roboflow Workmark](https://i.imgur.com/WHFqYSJ.png =350x)](https://roboflow.ai)
