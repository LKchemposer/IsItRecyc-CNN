# IsItRecyc-CNN
This project is Khoa Lam's passion project at the Metis data science bootcamp in NYC. Recycling contamination is not only an environmental but also an economic issue as recycling companies often redirect contaminated bales of recyclables to landfills. As a result, it increases human waste output and costs businesses resources. Here, I used a convolutional neural network (CNN) to predict if an object is recyclable from its image. My project aims to help consumers minimize recycling contamination. This goal is a shared goal with other projects and organizations (e.g., [TrashNet](https://github.com/garythung/trashnet), [Multilayer Hybrid Deep-Learning Method for Waste Classification and Recycling](https://www.hindawi.com/journals/cin/2018/5060857/), and [ZenRobotics](https://zenrobotics.com/)). This project, however, differs in that it uses mixed image sources (i.e., digital images and photographs), whereas many other projects use only photos. The final CNN was trained on the AWS server and achieved F<sub>0.5</sub> = 0.90 for recyclability, and averaged AUC = 0.75 for material classification (with  60/20/20 train-validate-test split). Lastly, the model was deployed into a Dash web app (currently defunct) on AWS Elastic Beanstalk. Presentation of this project can be found [here](https://docs.google.com/presentation/d/e/2PACX-1vRIqHnvxaCXHn-46DpMthLigO3ssJMiKFFpMz0ilDhhPHTWmeRv4fKm8noZtMFaapnuNUGYVorlfSXh/pub?start=true&loop=false&delayms=60000).

https://user-images.githubusercontent.com/39468345/140475768-3d0f3596-1cbf-489f-8710-234f884cfea5.mov

## Dataset

The dataset (in zip files) is now accessible in [a GDrive](https://drive.google.com/drive/folders/1r3EiKldemvRvk2j9Dy68FekbDYfZLmf_?usp=sharing).

Image sources for this project include:

1. Google Image Search, URLs from Google Custom Search API (code in [getting-urls notebook](./code/getting-urls.ipynb))
2. [TrashNet](https://github.com/garythung/trashnet)
3. A subset of [Caltech 256 Image Dataset](https://www.kaggle.com/jessicali9530/caltech256)
4. A subset of [Flickr Material Database (FMD)](https://people.csail.mit.edu/celiu/CVPR2010/FMD/)

Currently, the dataset consists of 11045 images separated into 8 categories:

1. Recyclables: 7543 images
   1. Glass (e.g., jars, bottles): 729 images
   2. Metal (e.g., cans, aluminum foil): 1747 images
   3. Paper (e.g., cardboard, books): 3230 images
   4. Plastic (e.g., soda bottles, food containers): 1837 images
2. Non-recyclables: 3502 images
   1. Glass (e.g., lightbulbs, mirror): 531 images
   2. Plastics (e.g., styrofoam, sports balls): 1850 images
   3. Tanglers (e.g., wire, cable): 290 images
   4. Other (e.g., battery, ceramic): 831 images

This model has two distinct outputs: (1) recyclability (binary output), and (2) material classification (categorical output). Recyclability is trained with F<sub>0.5</sub> as the metric, as F<sub>0.5</sub> weighs precision twice as much as recall (minimize true recyclable contamination). Material classification is trained with AUC to balance separation of one class from others.

## Notes

The code presented here is slightly simplified to be run on a local machine. To train the full dataset (~11000 images), an AWS Deep Learning AMI is recommended. 

Python packages required: pandas, numpy, seaborn, matplotlib, keras, tensowflow, sklearn, PIL, cv2
