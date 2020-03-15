def readFile(path):
    tagFile = open(path).readlines()
    # print(tagFile)
    for i in range(len(tagFile)):
        tagFile[i] = tagFile[i].strip()

    return tagFile
