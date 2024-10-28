from imageai.Classification import ImageClassification

prediction = ImageClassification()
prediction.setModelTypeAsMobileNetV2()
prediction.setModelPath('./assets/mobilenet_v2-b0353104.pth')
prediction.loadModel()

predctions, probabilities = prediction.classifyImage('./assets/house.jpg', result_count=5)
for eachPred, eachProb in zip(predctions, probabilities):
    print(f'{eachPred} : {eachProb}')