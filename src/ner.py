from flair.data import Sentence
from flair.nn import Classifier

# load the NER tagger
tagger = Classifier.load('ner-large')

def extract_entities(text):
    sentence = Sentence(text)
    # run NER over sentence
    tagger.predict(sentence)

    entities= []
    for entity in sentence.get_labels('ner'):
        entities.append({
            "text":entity.data_point.text,
            "type":entity.value,
            "score":entity.score
            })
    return entities

