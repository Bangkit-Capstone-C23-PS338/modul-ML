import csv
import random

CATEGORIES = ["Fashion", "Tech", "General", "Game", "Pop Culture", "Sport", "Music"]

def generate(n, name):
    with open(name, 'w', newline='') as file:
        writer = csv.writer(file)
        for i in range(n):
            print(";".join(random.sample(CATEGORIES, random.randint(1, len(CATEGORIES)))))
            writer.writerows()

def generate2(name, out):
    with open(name, 'r') as file:
        with open(out, 'w') as wfile:
            reader = csv.reader(file)
            writer = csv.writer(wfile, lineterminator='\n')
            all = []
            next(reader)
            for r in reader:
                r.append(";".join(random.sample(CATEGORIES, random.randint(1, len(CATEGORIES)))))
                all.append(r)

            writer.writerows(all)

# generate(4924, "influencer.csv")
# generate(1792, "owner.csv")

generate2("data_content_owner.csv", "data_content_owner_categ.csv")
generate2("data_content_influencer.csv", "data_content_influencer_categ.csv")