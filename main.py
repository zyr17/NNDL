import torch
import fastNLP

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
    
def SST2(modelfunc):
    def getdata(filename):
        data = fastNLP.DataSet.read_csv(filename, headers=('raw_sentence', 'label_str'), sep='|')
        data.apply(lambda x: x['raw_sentence'].lower(), new_field_name='raw_sentence')
        data.apply(lambda x: int(float(x['label_str']) * 1.99999), new_field_name='label', is_target=True)
        data.apply(lambda x: x['raw_sentence'].split(), new_field_name='word_str')
        data.drop(lambda x: x['label_str'] == '-1' or x['label_str'] == '0.5')
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
    
SST2(fastNLP.models.CNNText)