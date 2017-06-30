# NOAA Fisheries Steller Sea Lion Population Count

Source for my adventure into finding sea lions for the recent Kaggle competition sponsored by NOAA Fisheries.

### Files

"noaa sealions.ipynb" — main (working) notebook

"eda.ipynb" — exploratory analysis

"sealions_kfold.py" — similar approach to working notebook but configured for 5 (n) fold-validation 

### Data

Please resize the provided dataset to 512x512. My filepath is usually referencing a directory higher similar to: ```../data_512/```

### Requirements

I used Python 3.6 for my development on an ubuntu system with gpu support, please see the ```requirements.txt``` file for your environment setup. Adjust the tensorflow-gpu package if you will not run on a machine with a graphics card.

### Contact

bserna@regis.edu

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
