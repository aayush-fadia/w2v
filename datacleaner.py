import os
import re
import pickle


def clean_string(s):
    clean = re.sub(r"""
                   [,.;@#?!&$\t/\-:()=]+  # Accept one or more copies of punctuation
                   \ *           # plus zero or more copies of a space,
                   """,
                   " ",  # and replace it with a single space
                   s.lower().strip(), flags=re.VERBOSE)
    clean = re.sub("[0-9]+", "#", clean)
    clean = re.sub("\s\s+", " ", clean)
    return clean.strip()


reviews = []
dataset_root = 'scratchspace/hotels/data/'
folders = os.listdir(dataset_root)
for d in folders:
    if os.path.isdir(dataset_root + d):
        files = os.listdir(dataset_root + d + '/')
        print(files)
        for file in files:
            with open(dataset_root + d + '/' + file, 'r', encoding='ISO-8859-1') as f:
                lines = f.readlines()
                for line in lines:
                    spl = line.split('200')
                    if len(spl) == 2:
                        clean_str = clean_string(spl[1][1:])
                        if clean_str.count(' ') > 4:
                            print(clean_str)
                            reviews.append(clean_str)
print(len(reviews))
with open('scratchspace/reviews.pkl', 'wb') as f:
    pickle.dump(reviews, f, -1)
