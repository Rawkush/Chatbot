from spacy.lang.en import English
nlp = English()
tokenizer = nlp.Defaults.create_tokenizer(nlp)

class voc:
    
    def __init__(self):
        self.num_words= 1  # 0 is reserved for padding 
        self.num_tags=0
        self.tags={}
        self.index2tags={}
        self.questions={}
        self.word2index={}
        self.response={}
  
    def addWord(self,word):
        if word not in self.word2index:
            self.word2index[word] = self.num_words
            self.num_words += 1

    def addTags(self,tag):
        if tag not in self.tags:
            self.tags[tag]=self.num_tags
            self.index2tags[self.num_tags]=tag
            self.num_tags+=1
    def addQuestion(self, question, answer):
        self.questions[question]=answer
        words=self.tokenization(question)
        for  wrd in words:
            self.addWord(wrd)
                 
    def tokenization(self,ques):
        tokens = tokenizer(ques)
        token_list = []
        for token in tokens:
            token_list.append(token.lemma_)
        return token_list
    
    def getIndexOfWord(self,word):
        return self.word2index[word]
    
    def getQuestionInNum(self, ques):
        words=self.tokenization(ques)
        tmp=[ 0 for i in range(self.num_words)]
        for wrds in words:
            tmp[self.getIndexOfWord(wrds)]=1
        return tmp
    
 
    def getTag(self, tag):
        tmp=[0.0 for i in range(self.num_tags)]
        tmp[self.tags[tag]]=1.0
        return tmp
    
    def getVocabSize(self):
        return self.num_words
    
    def getTagSize(self):
        return self.num_tags

    def addResponse(self, tag, responses):
        self.response[tag]=responses
        
      