import re,string
import numpy as np 

class Preproses:
	KATA_DASAR  = []
	DATA_KBBI 	= []

	def __init__(self):
		global KATA_DASAR
		global DATA_KBBI
		KATA_DASAR 	= [line.strip('\n')for line in open('data/rootword.txt')]
		DATA_KBBI	= [kamus.strip('\n').strip('\r') for kamus in open('data/kbba.txt')]

	def tokenize(self, text): 
		# token = nltk.word_tokenize(text)
		token = text.split(' ')
		return token

	def kbbi(self, token): 
		global DATA_KBBI

		dic={}
		for i in DATA_KBBI: 
			(key,val)=i.split('\t')
			dic[str(key)]=val

		final_string = ' '.join(str(dic.get(word, word)) for word in token).split()
		return final_string

	def normalize_token(self, _tokens):
		tokens = self.kbbi(_tokens)
		return tokens

	def preprocess(self, text):

		def hapus_tanda(text): 
			tanda_baca = set(string.punctuation)
			text = ''.join(ch for ch in text if ch not in tanda_baca)
			return text

		def hapus_katadouble(s): 
			#look for 2 or more repetitions of character and replace with the character itself
			pattern = re.compile(r"(.)\1{1,}", re.DOTALL)
			return pattern.sub(r"\1\1", s)

		text=text.lower()
		text = re.sub(r'\\u\w\w\w\w', '', text)
		text=re.sub(r'http\S+','',text)
		#hapus @username
		text=re.sub('@[^\s]+','',text)
		#hapus #tagger 
		text = re.sub(r'#([^\s]+)', r'\1', text)
		#hapus tanda baca
		text=hapus_tanda(text)
		#hapus angka dan angka yang berada dalam string 
		text=re.sub(r'\w*\d\w*', '',text).strip()
		#hapus repetisi karakter 
		text=hapus_katadouble(text)
		return text

	def prep(self, sent):
		return self.normalize_token(self.tokenize(self.preprocess(sent)))