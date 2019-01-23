import torch
import fastNLP
import pandas
import pickle as pkl
import numpy as np
    
class CNNText(torch.nn.Module):
    """
    Text classification model by character CNN, the implementation of paper
    'Yoon Kim. 2014. Convolution Neural Networks for Sentence
    Classification.'
    """
    
    def addembed(self, embed):
        pass

    def __init__(self, embed_num,
                 embed_dim,
                 num_classes,
                 kernel_nums=(3, 4, 5),
                 kernel_sizes=(3, 4, 5),
                 padding=0,
                 dropout=0.5):
        super(CNNText, self).__init__()

        # no support for pre-trained embedding currently
        self.embed = fastNLP.modules.encoder.Embedding(embed_num, embed_dim)
        self.conv_pool = fastNLP.modules.encoder.ConvMaxpool(
            in_channels=embed_dim,
            out_channels=kernel_nums,
            kernel_sizes=kernel_sizes,
            padding=padding)
        self.dropout = torch.nn.Dropout(dropout)
        self.fc = fastNLP.modules.encoder.Linear(sum(kernel_nums), num_classes)

    def forward(self, word_seq):
        """

        :param word_seq: torch.LongTensor, [batch_size, seq_len]
        :return output: dict of torch.LongTensor, [batch_size, num_classes]
        """
        if word_seq.size(1) < 5:
            #print('word_seq ', word_seq)
            zeros = torch.zeros(word_seq.size(0), 5).long().cuda()
            #print('zeros ', zeros.size())
            zeros[:, :word_seq.size(1)] = word_seq
            word_seq = zeros
            #print('word_seq ', word_seq)
        #print(word_seq, word_seq.size())
        x = self.embed(word_seq)  # [N,L] -> [N,L,C]
        #print(x.size())
        x = self.conv_pool(x)  # [N,L,C] -> [N,C]
        #print(x.size())
        x = self.dropout(x)
        #print(x.size())
        x = self.fc(x)  # [N,C] -> [N, N_class]
        #print(x.size())
        return {'pred': x}

    def predict(self, word_seq):
        """

        :param word_seq: torch.LongTensor, [batch_size, seq_len]
        :return predict: dict of torch.LongTensor, [batch_size, seq_len]
        """
        output = self(word_seq)
        _, predict = output['pred'].max(dim=1)
        return {'pred': predict}
        
class CNNTextEmbFixed(CNNText):
    def __init__(self, embed_num,
                 embed_dim,
                 num_classes,
                 kernel_nums=(3, 4, 5),
                 kernel_sizes=(3, 4, 5),
                 padding=0,
                 dropout=0.5):
        super(CNNTextEmbFixed, self).__init__(embed_num,
                 embed_dim,
                 num_classes,
                 kernel_nums,
                 kernel_sizes,
                 padding,
                 dropout)
    def addembed(self, embed):
        self.embed.embed.weight.data.copy_(torch.from_numpy(embed))
        self.embed.embed.weight.requires_grad = False
        
class CNNTextEmb(CNNText):
    def __init__(self, embed_num,
                 embed_dim,
                 num_classes,
                 kernel_nums=(3, 4, 5),
                 kernel_sizes=(3, 4, 5),
                 padding=0,
                 dropout=0.5):
        super(CNNTextEmb, self).__init__(embed_num,
                 embed_dim,
                 num_classes,
                 kernel_nums,
                 kernel_sizes,
                 padding,
                 dropout)
    def addembed(self, embed):
        self.embed.embed.weight.data.copy_(torch.from_numpy(embed))
    
def RUN(data, label, split, modelfunc, classnum = 2, epochs = 10):
    assert(len(data) == len(label))
    if split == None:
        dataset = fastNLP.DataSet({'raw_sentence': data, 'label_str': label})
    else:
        dataset = fastNLP.DataSet({'raw_sentence': data, 'label_str': label, 'split': split})
    dataset.drop(lambda x: len(x['raw_sentence']) == 0)
    #[dataset.append(fastNLP.DataSet({'raw_sentence': data[x], 'label': label[x]})) for x in range(len(data))]
    dataset.apply(lambda x: int(float(x['label_str'])), new_field_name='label', is_target=True)
    dataset.apply(lambda x: x['raw_sentence'].split(), new_field_name='word_str')
    
    vocab = fastNLP.Vocabulary(min_freq = 1)
    dataset.apply(lambda x: [vocab.add(word) for word in x['word_str']])
    
    if split == None:
        traindata, testdata = dataset.split(0.1)
        #print(len(traindata), len(testdata))
    else:
        traindata = dataset[:]
        testdata = dataset[:]
        traindata.drop(lambda x: x['split'] != 'train')
        testdata.drop(lambda x: x['split'] != 'test')
        
    #print(len(traindata), len(testdata))
    
    traindata.apply(lambda x: [vocab.to_index(word) for word in x['word_str']], new_field_name='word_seq', is_input=True)
    testdata.apply(lambda x: [vocab.to_index(word) for word in x['word_str']], new_field_name='word_seq', is_input=True)
    
    model = modelfunc(embed_num = len(vocab),
                      embed_dim = 100,
                      num_classes = classnum,
                      kernel_nums = (3, 4, 5), 
                      kernel_sizes = (3, 4, 5),
                      padding = 0,
                      dropout = 0)
    model.embed.dropout = torch.nn.Dropout(0.5)
    
    gloveemb = np.random.rand(len(vocab), 100)
    for i in range(len(vocab)):
        word = vocab.to_word(i)
        try:
            #print(word)
            #print(len(glove), word)
            #print(glove[word])
            #input()
            emb = glove[word]
            gloveemb[i, :] = emb
        except:
            pass
            
    model.addembed(gloveemb)

    trainer = fastNLP.Trainer(model=model, 
                      train_data=traindata, 
                      dev_data=testdata,
                      loss=fastNLP.CrossEntropyLoss(),
                      metrics=fastNLP.AccuracyMetric(),
                      use_cuda = True,
                      n_epochs=epochs,
                      check_code_level=-1
                      )
    trainer.train()


pickle = pandas.read_pickle('data/MR.pkl')
MRdata, MRlabel = list(pickle.sentence), list(pickle.label)
pickle = pandas.read_pickle('data/SST1.pkl')
SST1data, SST1label, SST1split = list(pickle.sentence), list(pickle.label), list(pickle.split)
pickle = pandas.read_pickle('data/SST2.pkl')
SST2data, SST2label, SST2split = list(pickle.sentence), list(pickle.label), list(pickle.split)
pickle = pandas.read_pickle('data/TREC.pkl')
TRECdata, TREClabel, TRECsplit = list(pickle.sentence), list(pickle.label), list(pickle.split)
pickle = pandas.read_pickle('data/SUBJ.pkl')
SUBJdata, SUBJlabel = list(pickle.sentence), list(pickle.label)
pickle = pandas.read_pickle('data/TREC.pkl')
TRECdata, TREClabel, TRECsplit = list(pickle.sentence), list(pickle.label), list(pickle.split)
pickle = pandas.read_pickle('data/CR.pkl')
CRdata, CRlabel = list(pickle.sentence), list(pickle.label)
pickle = pandas.read_pickle('data/MPQA.pkl')
MPQAdata, MPQAlabel = list(pickle.sentence), list(pickle.label)

glove = pkl.load(open('data/twitter.pkl', 'rb'))
#print(len(glove))
#glove = {}

#print(len(MPQAdata), len(MPQAlabel))
#print(MPQAdata[10110], MPQAlabel[10110])

#RUN(MRdata, MRlabel, None, CNNText)
#RUN(SST1data, SST1label, SST1split, CNNText, 5)
#RUN(SST2data, SST2label, SST2split, CNNText)
#RUN(SUBJdata, SUBJlabel, None, CNNText)
#RUN(TRECdata, TREClabel, TRECsplit, CNNText, 6, 100)
#RUN(CRdata, CRlabel, CNNText)
#RUN(MPQAdata, MPQAlabel, None, CNNText)
def RUNALL(model):
    print('-----\nMR\n-----\n')
    RUN(MRdata, MRlabel, None, model)
    print('\n-----\nSST1\n-----\n')
    RUN(SST1data, SST1label, SST1split, model, 5)
    print('\n-----\nSST2\n-----\n')
    RUN(SST2data, SST2label, SST2split, model)
    print('\n-----\nSUBJ\n-----\n')
    RUN(SUBJdata, SUBJlabel, None, model)
    print('\n-----\nTREC\n-----\n')
    RUN(TRECdata, TREClabel, TRECsplit, model, 6, 100)
    print('\n-----\nCR\n-----\n')
    RUN(CRdata, CRlabel, None, model)
    print('\n-----\nMPQA\n-----\n')
    RUN(MPQAdata, MPQAlabel, None, model)
    
RUNALL(CNNTextEmbFixed)