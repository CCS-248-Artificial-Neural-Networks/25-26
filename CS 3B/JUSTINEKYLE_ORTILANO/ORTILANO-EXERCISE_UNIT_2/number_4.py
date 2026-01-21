import nltk
from nltk.corpus import webtext
import re


nltk.download('webtext')

pirates = webtext.raw('pirates.txt')

pattern = r"JACK SPARROW:.*"

jack_lines = re.findall(pattern, pirates)   

for line in jack_lines: 
    print(line)