import pandas as pd
import os
from .utils import _f, Utils
from tqdm import tqdm

class Processor:
    def __init__(self):
        self.df = None
        self.utils = Utils()

    def save(self, filename: str = None, raw: pd.DataFrame = None):
        try:
            file_extension = os.path.splitext(filename)[-1]
            file_handlers = {
                ".csv": raw.to_csv,
                ".json": raw.to_json,
                ".xlsx": raw.to_excel,
                ".parquet": raw.to_parquet,
            }
            if file_extension in file_handlers:
                file_handlers[file_extension](filename)
                _f("success", f"saved - {filename}")
            else:
                _f("fatal", "unsupported data")
        except Exception as e:
            _f("fatal", e)
    def load(self, raw: str | pd.DataFrame = None):
        try:
            if isinstance(raw, str):
                raw_data_dir = raw
                file_extension = os.path.splitext(raw)[-1]
                file_handlers = {
                    ".csv": pd.read_csv,
                    ".json": pd.read_json,
                    ".xlsx": pd.read_excel,
                    ".parquet": pd.read_parquet,
                }
                if file_extension in file_handlers:
                    self.df = file_handlers[file_extension](raw_data_dir)
                    _f("success", f"loaded - {raw_data_dir}")
                else:
                    _f("fatal", "unsupported file type")
            elif isinstance(raw, pd.DataFrame):
                self.df = raw
                _f("success", f"loaded - {raw}")
            else:
                _f("fatal", "data type not in [csv, json, xlsx, parquet, pd.DataFrame]")
        except Exception as e:
            _f("fatal", e)

    def export_as_sentences(self, path: str = None, text_column: str = "clean", id_column: str = 'id', splitter: any = None):
        if self.df is not None:
            try:
                _f("wait", f"get coffee or tea - {len(self.df)} processing...")
                sentence_splitter = self.bge_sentence_splitter if splitter is None else splitter
                all_sentences = []
                knowledge_base = pd.DataFrame()
                tqdm.pandas()
                self.df["sentences"] = self.df[text_column].progress_apply(
                    lambda x: [
                        str(s) for s in sentence_splitter(self.utils.normalize_text(x))
                    ]
                )
                for i in range(len(self.df)):
                    for s in self.df['sentences'].iloc[i]:
                        a = self.df[id_column].iloc[i]
                        all_sentences.append((a, s))
                knowledge_base['sentences'] = [x[1] for x in all_sentences]
                knowledge_base['id'] = [x[0] for x in all_sentences]
                self.df = knowledge_base
                self.save(path, self.df)
                return
            except Exception as e:
                _f("fatal", e)
        else:
            return _f("fatal", "no data loaded!")
        
    def bge_sentence_splitter(self, data):
        to_pop = []
        chunk = 768
        self.utils.nlp.max_length = len(data) + 100
        _ = list([str(x) for x in self.utils.nlp(data).sents])
        for sentence in range(len(_)-1):
            if len(_[sentence])>chunk:
                chunked = [_[sentence][i:i+chunk] for i in range(0, len(_[sentence]), chunk)]
                _+=chunked
                to_pop.append(sentence)
        [_.pop(sentence) for sentence in to_pop]
        return _
