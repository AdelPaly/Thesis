from ner import extract_entities
# pick a Signal A text
text = "Kaizer Chiefs were absolutely shocking on Saturday"
entities = extract_entities(text)
for e in entities:
    print(e)