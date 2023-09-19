import pandas as pd
import os
from .utils import _f, Utils

class Processor:
    def __init__(
        self, filename: str = None, raw_dir: str = None, cleaned_dir: str = None
    ):
        self.df = None
        self.filename = filename if filename else _f("warn", "no filepath passed!")
        self.raw_dir = (
            os.path.abspath(raw_dir) if raw_dir else _f("warn", "no raw_dir passed!")
        )
        self.cleaned_dir = (
            os.path.abspath(cleaned_dir)
            if cleaned_dir
            else _f("warn", "no cleaned_dir passed!")
        )
        
        self.utils = Utils()
        _f(
            "info", "Processor init"
        ) if self.filename and self.raw_dir and self.cleaned_dir else _f(
            "warn", "Processor partially loaded..."
        )

    def load(self, raw: str | pd.DataFrame = None):
        try:
            if isinstance(raw, str):
                raw_data_dir = os.path.join(self.raw_dir, raw)
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

    def export_with_sentences(self, category: str = "clean", splitter: any = None):
        if self.df is not None:
            try:
                _f("wait", f"get coffee or tea - {len(self.df)} processing...")
                sentence_splitter = self.bge_sentence_splitter if splitter is None else splitter
                self.df["sentences"] = self.df[category].apply(
                    lambda x: [
                        str(s) for s in sentence_splitter(self.utils.normalize_text(x))
                    ]
                )
                final_path = os.path.join(self.cleaned_dir, f"{self.filename}.parquet")
                self.df.to_parquet(final_path)
                return _f("success", f"🗳️  - {final_path}")
            except Exception as e:
                _f("fatal", e)
        else:
            return _f("fatal", "no data loaded!")
        
    def bge_sentence_splitter(self, data):
        self.utils.nlp.max_length = len(data) + 100
        _ = self.utils.nlp(data).sents
        return _
