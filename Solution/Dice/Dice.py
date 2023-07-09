def dice_similarity(text1,text2):
    set1 = set(text1.split())
    set2 = set(text2.split())
    intersection = len(set1.intersection(set2))
    return 2 * intersection / (len(set1) + len(set2)) if (len(set1) + len(set2)) != 0 else 0

text1 = 'I have a dog and i love it'
text2 = 'I have a dog and i love it'
print(dice_similarity(text1, text2))