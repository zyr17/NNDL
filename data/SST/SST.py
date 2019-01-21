p_pnum_f = open('dictionary.txt').readlines()[:]
snum_s_f = open('datasetSentences.txt').readlines()[1:]
pnum_label_f = open('sentiment_labels.txt').readlines()[1:]
snum_split_f = open('datasetSplit.txt').readlines()[1:]

p_pnum_f = [x.strip() for x in p_pnum_f]
snum_s_f = [x.strip() for x in snum_s_f]
pnum_label_f = [x.strip() for x in pnum_label_f]
snum_split_f = [x.strip() for x in snum_split_f]

ress = [[], [], []]
resp = [[], [], []]

sentences = ['']

for i in snum_s_f:
    [a, b] = i.split('\t')
    a = int(a)
    assert(a == len(sentences))
    sentences.append(b)

for i in snum_split_f:
    [a, b] = i.split(',')
    a = int(a)
    b = int(b) - 1
    ress[b].append([sentences[a]])

labels = []
    
for i in pnum_label_f:
    [a, b] = i.split('|')
    a = int(a)
    b = float(b)
    assert(a == len(labels))
    labels.append(b)

for count, i in enumerate(p_pnum_f):
    if count % 1000 == 0:
        print(count, len(p_pnum_f))
    [phrase, pnum] = i.split('|')
    pnum = int(pnum)
    for num in range(3):
        flag = False
        for sentence in ress[num]:
            if phrase == sentence[0]:
                sentence.append(labels[pnum])
                flag = True
        if flag:
            continue
        for sentence in ress[num]:
            if sentence[0].find(phrase) != -1:
                resp[num].append([phrase, labels[pnum]])
                break
                
for num in range(3):
    for j in ress[num]:
        if len(j) == 1:
            j.append(-1)
                
with open('train_p.txt', 'w') as f:
    f.write('\n'.join([x[0] + '|' + str(x[1]) for x in resp[0]]))
with open('verify_p.txt', 'w') as f:
    f.write('\n'.join([x[0] + '|' + str(x[1]) for x in resp[1]]))
with open('test_p.txt', 'w') as f:
    f.write('\n'.join([x[0] + '|' + str(x[1]) for x in resp[2]]))
with open('train_s.txt', 'w') as f:
    f.write('\n'.join([x[0] + '|' + str(x[1]) for x in ress[0]]))
with open('verify_s.txt', 'w') as f:
    f.write('\n'.join([x[0] + '|' + str(x[1]) for x in ress[1]]))
with open('test_s.txt', 'w') as f:
    f.write('\n'.join([x[0] + '|' + str(x[1]) for x in ress[2]]))