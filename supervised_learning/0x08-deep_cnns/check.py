#!/usr/bin/env python3


linesB = {hash(line) for line in open('intranet.txt')}
count = 0
for line in open('output.txt'):
    if hash(line) not in linesB:
        print(count)
        print('diferente')
    count +=1
