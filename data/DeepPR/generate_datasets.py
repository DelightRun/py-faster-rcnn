import os
import sys
import cv2
import numpy
import random

indices = []
for line in open('labels.txt', 'r').readlines():
    objs = line.strip().split(' * ')
    index = objs[0]
    indices.append(index)

    def getBoundingRect(obj):
        points = numpy.asarray(list(map(int, obj.split()))).reshape((4,2))
        return cv2.boundingRect(points)

    rects = [getBoundingRect(obj) for obj in objs[1:]]
    lines = ['%d %d %d %d\n' % rect for rect in rects]

    with open(os.path.join('.', 'data', 'Annotations', index + '.txt'), 'w') as f:
        f.writelines(lines)

print(indices[:20])
random.shuffle(indices)
print(indices[:20])

trainSet = sorted(indices[:-100])
testSet = sorted(indices[-100:])

with open(os.path.join('.', 'data', 'ImageSets', 'train.txt'), 'w') as f:
    f.write('\n'.join(trainSet))

with open(os.path.join('.', 'data', 'ImageSets', 'test.txt'), 'w') as f:
    f.write('\n'.join(testSet))
