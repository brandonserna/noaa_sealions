# NOAA Fisheries Steller Sea Lion Population Count

Source for my solution into detecting and classifying Stellar Sea Lions for the recent Kaggle competition sponsored by NOAA Fisheries. This approach is using python and keras. 

### Files

```bash
├── README.md
├── annotated_wDotted_model.ipynb  # working with dotted photos (modified from Radu kernel) 
├── eda.ipynb  # exploratory data analysis and other testing/verification
├── input  # input data
│   ├── data  # sample store for original file sizes
│   │   ├── 41.jpg
│   │   └── 42.jpg
│   └── data_512  # store for modified images 512x512 (where additional dirs live (validation, etc...))
│       ├── 41.jpg
│       └── 42.jpg
├── model_plot.png  # keras export of model
├── noaa\ sealions.ipynb  # main in-progress notebook
├── requirements.txt  # python requirements 
├── sealions_kfold.py  # implementation with K-Fold validation (very slow)
└── yolo  # in-progress testing of yolo v2
    └── yad2k
        └── Untitled\ Folder
            └── yoloTest.ipynb
```

### Data

Please resize the provided dataset to 512x512. My filepath is usually referencing a directory higher similar to: ```../data_512/```

### Requirements

I used Python 3.6 for my development on an ubuntu system with gpu support, please see the ```requirements.txt``` file for your environment setup. Adjust the tensorflow-gpu package if you will not run on a machine with a graphics card.

### Resources

Competition guidelines: https://www.kaggle.com/c/noaa-fisheries-steller-sea-lion-population-count


http://imagenet.stanford.edu/synset?wnid=n02077923 

http://www.robots.ox.ac.uk/~vgg/research/very_deep/ 

https://www.kaggle.com/c/noaa-fisheries-steller-sea-lion-population-count/ 

https://www.kaggle.com/radustoicescu/use-keras-to-classify-sea-lions-0-91-accuracy 

https://www.kaggle.com/radustoicescu/count-the-sea-lions-in-the-first-image 

https://github.com/mrgloom/Kaggle-Sea-Lions-Solution/blob/master/01-Image%20level%20regression/run_me.py 

http://machinelearningmastery.com/display-deep-learning-model-training-history-in-keras/ 


##### My Resources

Blog: https://brandonserna.github.io/NOAA-Fisheries/ 

Source code: https://github.com/brandonserna/noaa_sealions/ 

Presentation: https://brandonserna.github.io/nota_reveal_pres/

