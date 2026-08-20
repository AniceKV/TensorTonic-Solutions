import numpy as np
from typing import List, Dict

class SimpleTokenizer:
    """
    A word-level tokenizer with special tokens.
    """
    
    def __init__(self):
        self.word_to_id: Dict[str, int] = {}
        self.id_to_word: Dict[int, str] = {}
        self.vocab_size = 0
        
        # Special tokens
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.bos_token = "<BOS>"
        self.eos_token = "<EOS>"
    
    def build_vocab(self, texts: List[str]) -> None:
        """
        Build vocabulary from a list of texts.
        Add special tokens first, then unique words.
        """
        unique_words=[]
        for x in texts:
            unique_words.extend(x.lower().strip().split())
        unique_words=sorted(list(set(unique_words)))

        self.vocab_size=len(unique_words)+4

        for i,x in enumerate([self.pad_token,self.unk_token,self.bos_token,self.eos_token]+unique_words):
            self.word_to_id[x]=i
            self.id_to_word[i]=x


    
    def encode(self, text: str) -> List[int]:
        """
        Convert text to list of token IDs.
        Use UNK for unknown words.
        """
        return [self.word_to_id.get(x.lower(),1) for x in text.strip().split()]
        
    
    def decode(self, ids: List[int]) -> str:
        """
        Convert list of token IDs back to text.
        """
        return " ".join([self.id_to_word.get(x,"<UNK>") for x in ids])
