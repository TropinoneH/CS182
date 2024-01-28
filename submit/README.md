# Introduction
This project is a neural network based project for music genre classification. We will accept an audio file and then categorize this audio based on the audio file. In the end, we roughly categorized into 4 broad categories: pop, jazz, classical and rock.

# Preparation
First install the dependencies:

```shell
pip install tqdm numpy pandas matplotlib 
pip install sklearn
pip install tensorflow==2.15.0 librosa
pip install pydot
```

Then, download the dataset: [kaggle](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification/), [fma](https://os.unil.cloud.switch.ch/fma/fma_small.zip), [fma metadata](https://os.unil.cloud.switch.ch/fma/fma_metadata.zip), and then put them in the correct place.

After that, run the `NeuralNetwork/data-preprocess.py` to generate training data.

Finally, open `classifier.ipynb` and run it one by one to get a good training result.

# Results presentation:
In the training dataset, the final accuracy is able to reach 82% accuracy. Replacing another dataset, it can be noticed that the accuracy is around 58%. For common songs that are longer in length, the accuracy improves. But there are too few songs completed, so the test data cannot produce statistical results.