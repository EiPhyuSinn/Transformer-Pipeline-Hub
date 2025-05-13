from transformers import pipeline

class SentimentAnalysis:
    def __init__(self, text):
        self.text = text 
    def predict(self):
        clf = pipeline('sentiment-analysis')
        return clf(self.text)

class FillMask:
    def __init__(self, text):
        self.text = text 
    def predict(self):
        model = pipeline('fill-mask')
        return model(self.text)
    
class NER:
    def __init__(self, text):
        self.text = text 
    def predict(self):
        model = pipeline('ner')
        return model(self.text)
    
class QuestionAnswering:
    def __init__(self, question, context):
        self.question = question
        self.context = context
    def predict(self):
        model = pipeline('question-answering')
        return model(
            question=self.question,
            context=self.context
        )
    
class Summarization:
    def __init__(self, text):
        self.text = text
    def predict(self):
        model = pipeline('summarization')
        return model(self.text)
        
class TextGenerator:
    def __init__(self, text):
        self.text = text
    def predict(self):
        model = pipeline('text-generation')
        return model(self.text)
           
class Translation:
    def __init__(self, text):
        self.text = text
    def predict(self):
        model = pipeline('translation', model="Helsinki-NLP/opus-mt-fr-en")
        return model(self.text)
    
class ZeroShotClassification:
    def __init__(self, text, candidate_labels):
        self.text = text
        self.candidate_labels = candidate_labels
    def predict(self):
        model = pipeline('zero-shot-classification')
        return model(
            self.text,
            candidate_labels=self.candidate_labels
        )
