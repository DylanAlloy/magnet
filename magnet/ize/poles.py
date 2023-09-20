import faiss
import pandas as pd
from magnet.utils import Utils, _f
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import numpy as np

class Charge:
    def __init__(self, model: str = 'BAAI/bge-large-en-v1.5'):
        self.model = model
        self.sentences_index = None
        self.utils = Utils()
        
    def index_document_embeddings(self, df: pd.DataFrame = None):
        try:
            model = SentenceTransformer(self.model)
            d = model[1].word_embedding_dimension
            all_embeddings = []
            if self.utils.check_cuda():
                sentences_index = faiss.IndexFlatIP(d)
                co, co.shard, co.useFloat16 = faiss.GpuMultipleClonerOptions(), True, True
                sentences_index = faiss.index_cpu_to_all_gpus(sentences_index, co=co)
            else:
                sentences_index = faiss.IndexFlatL2(d)
            if sentences_index.is_trained:
                pbar = tqdm(range(len(df)))
                for i in pbar:
                    sentences = df['sentences'].iloc[i]
                    embeddings = model.encode(sentences, normalize_embeddings=True)
                    all_embeddings.append([embeddings])
                    pbar.set_description(
                        _f(
                            "success",
                            f"embedded {df['answerId'].iloc[i]}",
                            no_print=True,
                        ),
                        refresh=True,
                    )
            _f('wait', f'indexing {len(all_embeddings)} objects')
            sentences_index.add(np.asarray(all_embeddings, dtype=np.float32))
            self.sentences_index = sentences_index
            _f('success', 'index created')
        except Exception as e:
            _f('fatal', e)
    
    def search_document_embeddings(self, q: str = None, k: int = 64):
        try:
            model = SentenceTransformer(self.model)
            xq = model.encode(q, normalize_embeddings=True)
            D, I  = self.sentences_index.search(xq, k)
            _f('info', f'found {I} indices')
            results = []
            for i in range(len(I)):
                results.append(self.df['sentences'].iloc[i])
            return results
        except Exception as e:
            _f('fatal', e)
    
    def save_embeddings(self, index_path: str = None):
        if self.sentences_index:
            faiss.write_index(self.sentences_index, index_path)
            _f('success', f'embeddings saved to {index_path}')
        else:
            _f('fatal', 'no index in memory')

    def load_embeddings(self, index_path: str = None):
        try:
            f = open(index_path, 'rb')
            reader = faiss.PyCallbackIOReader(f.read)
            index = faiss.read_index(reader)
            if Utils.check_cuda():
                co, co.shard, co.useFloat16 = faiss.GpuMultipleClonerOptions(), True, True
                index = faiss.index_cpu_to_all_gpus(index, co=co)
            self.sentences_index = index
            _f('success', f'index loaded - {index_path}')
        except Exception as e:
            _f('fatal',e)

        