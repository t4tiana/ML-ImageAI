#using ImageAI library
from imageai.Classification import ImageClassification
import os

execution_path = os.getcwd()

prediction = ImageClassification()
# Using Google's InceptionV3
prediction.setModelTypeAsInceptionV3()
prediction.setModelPath(os.path.join(execution_path, "inception_v3_google-1a9a5a14.pth"))
prediction.loadModel()


# result count = how many guesses you want it to make
# probabilities are floats indicating percent of certainty
predictions, probabilities = prediction.classifyImage(os.path.join(execution_path, "dandelion.jpg"), result_count=5 )
for eachPrediction, eachProbability in zip(predictions, probabilities):
    print(eachPrediction , " : " , eachProbability)
