import pandas as pd
import os
from .utils import _f, Utils


class Processor:
    def __init__(
        self, filename: str = None, raw_dir: str = None, cleaned_dir: str = None
    ):
        """
        The function initializes a Processor object with optional parameters for filename, raw_dir, and
        cleaned_dir.

        :param filename: The filename parameter is used to specify the name of the file that will be
        processed. It is a string that represents the name of the file
        :type filename: str
        :param raw_dir: The `raw_dir` parameter is a string that represents the directory path where the
        raw data files are located. This directory is used to read the raw data files for processing
        :type raw_dir: str
        :param cleaned_dir: The `cleaned_dir` parameter is a string that represents the directory path
        where the cleaned data will be stored. It is an optional parameter, so if no value is provided,
        a warning message will be displayed
        :type cleaned_dir: str
        :param utils: The `utils` parameter is an instance of the Utils class that provides utility functions
        for data cleaning and processing
        :type utils: Utils
        """
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
        """
        The function `load` loads data from a file or a DataFrame into an object, with support for CSV,
        JSON, Excel, Parquet, and SQL file formats.

        :param raw: The `raw` parameter in the `load` method can accept either a string or a pandas
        DataFrame
        :type raw: str | pd.DataFrame
        """
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

    def export_with_sentences(self, category: str = "clean"):
        """
        The function exports a DataFrame to a JSON file, adding a new column called 'sentences' that
        contains the cleaned sentences from a specified category column.

        :param category: The "category" parameter is a string that specifies the column name in the
        dataframe that contains the text data to be processed, defaults to clean
        :type category: str (optional)
        :return: a message indicating that no data is loaded if the `self.df` is `None`.
        """
        if self.df is not None:
            try:
                _f("wait", f"get coffee or tea - {len(self.df)} processing...")
                self.df["sentences"] = self.df[category].apply(
                    lambda x: [
                        self.utils.clean(s) for s in self.utils.sentence_splitter(x)
                    ]
                )
                final_path = os.path.join(self.cleaned_dir, f"{self.filename}.json")
                self.df.to_json(final_path, default_handler=str)
                return _f("success", f"🗳️  - {final_path}")
            except Exception as e:
                _f("fatal", e)
        else:
            return _f("fatal", "no data loaded!")
