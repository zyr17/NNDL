import torch
import fastNLP
import pandas

def cuda(tensor):
    """A cuda wrapper
    """
    if tensor is None:
        return None
    if torch.cuda.is_available():
        return tensor.cuda()
    else:
        return tensor

def sequence_mask(sequence_length, max_length=None):
    """
    e.g., sequence_length = "5,7,8", max_length=None
    it will return
    tensor([[ 1,  1,  1,  1,  1,  0,  0,  0],
            [ 1,  1,  1,  1,  1,  1,  1,  0],
            [ 1,  1,  1,  1,  1,  1,  1,  1]], dtype=torch.float32)

    :param sequence_length: a torch tensor
    :param max_length: if not given, it will be set to the maximum of `sequence_length`
    :return: a tensor with dimension  [*sequence_length.size(), max_length]
    """
    if len(sequence_length.size()) > 1:
        ori_shape = list(sequence_length.size())
        sequence_length = sequence_length.view(-1) # [N, ?]
        reshape_back = True
    else:
        reshape_back = False

    if max_length is None:
        max_length = sequence_length.data.max()
    batch_size = sequence_length.size(0)
    seq_range = torch.arange(0, max_length).long() # [max_length]
    seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_length) # [batch, max_len], repeats on each column
    seq_range_expand = torch.autograd.Variable(seq_range_expand).to(sequence_length.device)
    #if sequence_length.is_cuda:
    #    seq_range_expand = seq_range_expand.cuda()
    seq_length_expand = sequence_length.unsqueeze(1).expand_as(seq_range_expand) # [batch, max_len], repeats on each row

    ret = (seq_range_expand < seq_length_expand).float() # [batch, max_len]

    if reshape_back:
        ret = ret.view(ori_shape + [max_length])

    return ret
    
class CNN(torch.nn.Module):
    def __init__(self, maxlength = 40, cnnkernel = [3, 4, 5], cnndim = [100, 100, 100], label = 2, indim = 1, wordsize = 200000, embdim = 300, embdata = None, mode = 'simple'):
        super(CNN,self).__init__()
        self.maxlength = maxlength
        self.cnnkernel = cnnkernel
        self.cnndim = cnndim
        assert(len(cnnkernel) == len(cnndim))
        self.outputlabel = label
        self.indim = indim
        self.mode = mode
        
        #in:[B, L, wordsize] out: [B, L, embdim]
        self.emb1 = torch.nn.Embedding(wordsize, embdim, wordsize)
        self.emb2 = torch.nn.Embedding(wordsize, embdim, wordsize)
        self.emb2.weight.data.copy(embdata)
        self.emb2.weight.requires_grad = False
        if mode != 'simple':
            self.emb1.weight.data.copy(embdata)
        if mode == 'fixed':
            self.emb1.weight.requires_grad = False
            
        #in: [B, embdim, L] out: len(cnnkernel) * [B, cnndim, 1]
        self.convs = []
        for i in range(len(cnnkernel)):
            k = cnnkernel[i]
            dim = cnndim[i]
            conv = torch.nn.Conv1d(embdim, dim, k)
            maxpool = torch.nn.MaxPool1d(maxlength - k + 1)
            self.convs.append([conv, maxpool])
        
        #in: [B, sum(cnndim)] out: [B, label]
        cnndimsum = 0
        for i in cnndim:
            cnndimsum += i
        self.fc = torch.nn.Linear(i, label)
        
    def forward(self, input): # [B, L, wordsize]
        x = self.emb1(input) # [B, L, embdim]
        
        if mode == 'multi':
            print('multi not implemented.')
            assert(0)
        
        x = x.permute(0, 2, 1) # [B, embdim, L]
        convx = []
        for [conv, maxpool] in self.convs:
            y = conv(x) # [B, cnndim, L - k + 1]
            y = torch.nn.ReLU(y)
            y = maxpool(y) # [B, cnndim, 1]
        x = torch.cat(convx, 1).squeeze(2) # [B, sum(cnndim)]
        x = self.fc(x) # [B, label]
        
        return x
        
def SST1(modelfunc):
    def getdata(filename):
        data = fastNLP.DataSet.read_csv(filename, headers=('raw_sentence', 'label_str'), sep='|')
        data.apply(lambda x: x['raw_sentence'].lower(), new_field_name='raw_sentence')
        data.apply(lambda x: int(float(x['label_str']) * 4.99999), new_field_name='label', is_target=True)
        data.apply(lambda x: x['raw_sentence'].split(), new_field_name='word_str')
        data.drop(lambda x: x['label_str'] == '-1')
        return data

    traindata = getdata("data/SST/train_s.txt")
    [traindata.append(x) for x in getdata("data/SST/train_p.txt")]
    testdata = getdata("data/SST/test_s.txt")

    vocab = fastNLP.Vocabulary(min_freq = 1)
    traindata.apply(lambda x: [vocab.add(word) for word in x['word_str']])
    testdata.apply(lambda x: [vocab.add(word) for word in x['word_str']])
    traindata.apply(lambda x: [vocab.to_index(word) for word in x['word_str']], new_field_name='word_seq', is_input=True)
    testdata.apply(lambda x: [vocab.to_index(word) for word in x['word_str']], new_field_name='word_seq', is_input=True)

    #print(traindata[1111], testdata[111])

    model = modelfunc(embed_num=len(vocab),embed_dim=100,num_classes=5,kernel_nums=(3,4,5), kernel_sizes=(3,4,5),padding=0,dropout=0)
    model.embed.dropout=torch.nn.Dropout(0.5)

    trainer = fastNLP.Trainer(model=model, 
                      train_data=traindata, 
                      dev_data=testdata,
                      loss=fastNLP.CrossEntropyLoss(),
                      metrics=fastNLP.AccuracyMetric(),
                      use_cuda = True,
                      n_epochs=10
                      )
    trainer.train()
    
def SST2(modelfunc):
    def getdata(filename):
        data = fastNLP.DataSet.read_csv(filename, headers=('raw_sentence', 'label_str'), sep='|')
        data.apply(lambda x: x['raw_sentence'].lower(), new_field_name='raw_sentence')
        data.apply(lambda x: int(float(x['label_str']) * 1.99999), new_field_name='label', is_target=True)
        data.apply(lambda x: x['raw_sentence'].split(), new_field_name='word_str')
        data.drop(lambda x: x['label_str'] == '-1')
        data.drop(lambda x: float(x['label_str']) >= 0.4 and float(x['label_str']) < 0.6)
        return data

    traindata = getdata("data/SST/train_s.txt")
    [traindata.append(x) for x in getdata("data/SST/train_p.txt")]
    testdata = getdata("data/SST/test_s.txt")
    
    #print(len(testdata))

    vocab = fastNLP.Vocabulary(min_freq = 1)
    traindata.apply(lambda x: [vocab.add(word) for word in x['word_str']])
    testdata.apply(lambda x: [vocab.add(word) for word in x['word_str']])
    traindata.apply(lambda x: [vocab.to_index(word) for word in x['word_str']], new_field_name='word_seq', is_input=True)
    testdata.apply(lambda x: [vocab.to_index(word) for word in x['word_str']], new_field_name='word_seq', is_input=True)

    #print(traindata[1111], testdata[111])

    model = modelfunc(embed_num=len(vocab),embed_dim=100,num_classes=2,kernel_nums=(3,4,5), kernel_sizes=(3,4,5),padding=0,dropout=0)
    model.embed.dropout=torch.nn.Dropout(0.5)

    trainer = fastNLP.Trainer(model=model, 
                      train_data=traindata, 
                      dev_data=testdata,
                      loss=fastNLP.CrossEntropyLoss(),
                      metrics=fastNLP.AccuracyMetric(),
                      use_cuda = True,
                      n_epochs=10
                      )
    trainer.train()
    
class CNNText(torch.nn.Module):
    """
    Text classification model by character CNN, the implementation of paper
    'Yoon Kim. 2014. Convolution Neural Networks for Sentence
    Classification.'
    """

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
    
    model = modelfunc(embed_num=len(vocab),embed_dim=100,num_classes=classnum,kernel_nums=(3,4,5), kernel_sizes=(3,4,5),padding=0,dropout=0)
    model.embed.dropout=torch.nn.Dropout(0.5)

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

#print(len(MPQAdata), len(MPQAlabel))
#print(MPQAdata[10110], MPQAlabel[10110])

#RUN(MRdata, MRlabel, None, CNNText)
#RUN(SST1data, SST1label, SST1split, CNNText, 5)
#RUN(SST2data, SST2label, SST2split, CNNText)
#RUN(SUBJdata, SUBJlabel, None, CNNText)
#RUN(TRECdata, TREClabel, TRECsplit, CNNText, 6, 100)
#RUN(CRdata, CRlabel, CNNText)
#RUN(MPQAdata, MPQAlabel, None, CNNText)