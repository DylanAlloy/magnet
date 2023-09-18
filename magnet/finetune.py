import pandas as pd
import random, os, json, multiprocessing
from tqdm import tqdm
from .utils import _f, Utils
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss


def _create_index(embeddings, use_gpu):
    """
    The function `_create_index` creates an index using Faiss library for similarity search on a set of
    embeddings.

    :param embeddings: The "embeddings" parameter is a list of vectors representing the embeddings of
    some data. Each vector should have the same length
    :param use_gpu: The `use_gpu` parameter is a boolean value that indicates whether to use a GPU for
    indexing or not. If `use_gpu` is `True`, the function will use a GPU for indexing. If `use_gpu` is
    `False`, the function will use the CPU for indexing
    :return: the created index.
    """
    index = faiss.IndexFlatIP(len(embeddings[0]))
    embeddings = np.asarray(embeddings, dtype=np.float32)
    if use_gpu:
        co = faiss.GpuMultipleClonerOptions()
        co.shard = True
        co.useFloat16 = True
        index = faiss.index_cpu_to_all_gpus(index, co=co)
    index.add(embeddings)
    return index


def _batch_search(index, query, topk, batch_size: int = 64):
    """
    The function `_batch_search` performs a batch search using an index and a query, returning the
    scores and indices of the top results.

    :param index: The index is a data structure that stores the information needed for efficient
    searching. It could be a search index, a database index, or any other data structure that allows for
    fast retrieval of information
    :param query: The `query` parameter is a list of queries that you want to search for in the index
    :param topk: The parameter "topk" specifies the number of top results to retrieve for each query
    :param batch_size: The `batch_size` parameter determines the number of queries that are processed in
    each batch. It specifies how many queries are processed at a time during the search operation. In
    the given code, the default value of `batch_size` is set to 64, which means that 64 queries will be,
    defaults to 64
    :type batch_size: int (optional)
    :return: The function `_batch_search` returns two lists: `all_scores` and `all_inxs`.
    """
    all_scores, all_inxs = [], []
    for start_index in tqdm(
        range(0, len(query), batch_size),
        desc=_f("wait", "searching", no_print=True),
        disable=len(query) < 256,
    ):
        batch_query = query[start_index : start_index + batch_size]
        batch_scores, batch_inxs = index.search(
            np.asarray(batch_query, dtype=np.float32), k=topk
        )
        all_scores.extend(batch_scores.tolist())
        all_inxs.extend(batch_inxs.tolist())
    return all_scores, all_inxs


def _score_data_job(args):
    """
    The `_score_data_job` function takes in various arguments and performs a scoring task on a chunk of
    data. It iterates over the chunk, randomly selects sentences and contexts from a dataframe, and
    calculates scores based on a given model. The scores are then stored in a new dataframe called
    `training_data`.

    :param args: The `args` parameter is a tuple that contains the following elements:
    """
    (
        group_by,
        plaintext_column,
        chunk_start,
        chunk_end,
        df,
        training_data,
        model,
        prompt,
        task,
    ) = args
    pbar = tqdm(range(chunk_start, chunk_end))
    for i in pbar:
        context_index, sentences_index = random.randint(0, len(df)), random.randint(
            0, len(df)
        )
        q1, q2 = (
            (
                df["sentences"][sentences_index],
                Utils().sentence_splitter(df[plaintext_column][context_index]),
            )
            if task == "similarity"
            else (df["sentences"][sentences_index], df[plaintext_column][context_index])
        )
        _min = min([len(q1), len(q2)])
        if _min > 0:
            q1, q2 = (
                (random.sample(q1, _min), random.sample(q2, _min))
                if task == "similarity"
                else ([prompt + s for s in random.sample(q1, _min)], q2)
            )
            for _s in range(_min):
                emb1 = model.encode(q1, normalize_embeddings=True)
                emb2 = model.encode(q2, normalize_embeddings=True)
                _scores = emb1 @ emb2.T
                for _q, _score in zip(q1, _scores):
                    _df = pd.DataFrame(
                        [
                            {
                                "sentences": _q,
                                "id": int(df[group_by][sentences_index]),
                                "scores": _score,
                                "context_sentences": q2,
                                "context_id": df[group_by][context_index],
                            }
                        ]
                    )
                    training_data = pd.concat([training_data, _df], ignore_index=True)
                pbar.set_description(
                    _f(
                        "success",
                        f"sample {_s} - comparing {int(df[group_by][sentences_index])} 🧮 {df[group_by][context_index]}",
                        no_print=True,
                    ),
                    refresh=True,
                )
    return training_data


class FinePrep:
    def __init__(
        self, filename: str = None, raw_dir: str = None, cleaned_dir: str = None
    ):
        """
        The function initializes an instance of the FinePrep class with optional parameters for filename,
        raw_dir, and cleaned_dir.

        :param filename: The filename parameter is used to specify the name of the file that will be
        processed by the FinePrep class
        :type filename: str
        :param raw_dir: The `raw_dir` parameter is a string that represents the directory path where the raw
        data files are located
        :type raw_dir: str
        :param cleaned_dir: The `cleaned_dir` parameter is a string that represents the directory where the
        cleaned data will be saved.
        :type cleaned_dir: str
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
        _f(
            "info", "FinePrep init"
        ) if self.filename and self.raw_dir and self.cleaned_dir else _f(
            "warn", "FinePrep partially loaded..."
        )

    def load(self, raw: str | pd.DataFrame = None):
        """
        The function `load` loads data from a file or a DataFrame into an object, with support for CSV
        and JSON formats.

        :param raw: The `raw` parameter in the `load` method can be either a string or a pandas
        DataFrame
        :type raw: str | pd.DataFrame
        """
        try:
            if isinstance(raw, str):
                raw_data_dir = os.path.join(self.raw_dir, raw)
                if raw.endswith(".csv"):
                    self.df = pd.read_csv(raw_data_dir)
                elif raw.endswith(".json"):
                    self.df = pd.read_json(raw_data_dir)
                _f("success", f"loaded - {raw_data_dir}")
            elif isinstance(raw, pd.DataFrame):
                self.df = raw
                _f("success", f"loaded - {raw}")
            else:
                _f("fatal", "type(data) not in [str, dict, pd.DataFrame]")
        except Exception as e:
            _f("fatal", e)

    def generate_training_data(self, quant: float = 0.7):
        """
        The function `generate_training_data` takes a dataframe `self.df` and generates training data by
        selecting positive and negative context sentences based on a given quantile value. The function
        writes the training data in JSONL format to a file.

        :param quant: The parameter "quant" is a float value that represents the quantile value used to
        split the scores into positive and negative examples. It determines the threshold for
        classifying a context sentence as positive or negative. The default value is 0.7, which means
        that context sentences with scores above the
        :type quant: float
        """
        data = self.df
        out = os.path.join(self.cleaned_dir, f"{self.filename}.jsonl")
        f = open(out, "w")
        pbar = tqdm(range(len(data)))
        for i in pbar:
            _scores = data.iloc[i]["scores"]
            context_sentences = data.iloc[i]["context_sentences"]
            quantile = np.quantile(data["scores"].iloc[i], quant)
            _pos, _neg = [
                context_sentences[s]
                for s in range(len(context_sentences))
                if _scores[s] > quantile
            ], [
                context_sentences[s]
                for s in range(len(context_sentences))
                if _scores[s] < quantile
            ]
            if min(len(_pos), len(_neg)) > 0:
                json.dump(
                    {
                        "query": data.iloc[i]["sentences"],
                        "pos": [x for x in _pos],
                        "neg": [x for x in _neg],
                    },
                    f,
                )
                f.write("\n")
                pbar.set_description(
                    _f(
                        "info",
                        f'processed  - "{data.iloc[i]["sentences"]}"',
                        no_print=True,
                        luxe=True,
                    ),
                    refresh=True,
                )
        _f("success", f"written - {out}")

    def generate_scored_data(
        self,
        group_by: str = "answerId",
        plaintext_column: str = "clean",
        split: int = 16,
        model: str = "BAAI/bge-large-en-v1.5",
        use_multiprocessing: bool = False,
        prompt: str = None,
        task: str = None,
    ):
        """
        The function `generate_scored_data` generates scored data based on a given task (retrieval or
        similarity) using a sentence transformer model. It splits the data into chunks and processes
        them either sequentially or in parallel using multiprocessing. The scored data is then saved to
        a JSON file.

        :param sentence_column: The `sentence_column` parameter is the name of the column in your
        dataframe that contains the sentences you want to generate scores for, defaults to sentences
        :type sentence_column: str (optional)
        :param group_by: The `group_by` parameter is used to specify the column in the dataframe that
        should be used for grouping the data. This is typically used when you want to generate scores
        for each group separately, defaults to answerId
        :type group_by: str (optional)
        :param plaintext_column: The `plaintext_column` parameter is the name of the column in the
        dataframe that contains the plaintext sentences, defaults to clean
        :type plaintext_column: str (optional)
        :param split: The `split` parameter determines the number of chunks the data will be divided
        into for processing. It is used to control the number of processes or iterations when generating
        scored data, defaults to 16
        :type split: int (optional)
        :param model: The `model` parameter is used to specify the pre-trained model to be used for
        generating sentence representations. In this case, the default value refers to a specific
        pre-trained model for generating sentence representations, BAAI/bge-large-en-v1.5
        :type model: str (optional)
        :param use_multiprocessing: The `use_multiprocessing` parameter determines whether to use
        multiprocessing for parallel processing or not. If set to `True`, the data processing will be
        divided into multiple processes to speed up the computation. If set to `False`, the data
        processing will be done sequentially, defaults to False
        :type use_multiprocessing: bool (optional)
        :param prompt: The `prompt` parameter is used to specify the prompt text that will be used when
        generating a representation for each sentence. If the `prompt` parameter is not provided and the
        `model` is set to 'BAAI/bge-large-en-v1.5', a default prompt will be used
        :type prompt: str
        :param task: The `task` parameter specifies the type of task to perform. It can be either
        "retrieval" or "similarity"
        :type task: str
        :return: a message indicating the success or failure of the operation. If the operation is
        successful, it returns a success message along with the path where the generated data is saved.
        If there is an error or if no data is loaded, it returns a fatal error message.
        """
        if task is None:
            return _f("fatal", 'please pass "retrieval" or "similarity" as `task`')
        if self.df is not None:
            try:
                _f(
                    "wait",
                    f"get coffee or tea - {int(len(self.df)/split)} (1/{split} of your data) processing...",
                )
                _model = SentenceTransformer(model)
                _prompt = (
                    "Generate a representation for this sentence that can be used to retrieve related articles："
                    if prompt is None and model == "BAAI/bge-large-en-v1.5"
                    else prompt
                )

                if use_multiprocessing:
                    num_processes = multiprocessing.cpu_count()
                    chunk_size = int((int(len(self.df) / split) / num_processes))

                    with multiprocessing.Pool(processes=num_processes) as pool:
                        args_list = []
                        for i in range(num_processes):
                            training_data = pd.DataFrame()
                            (
                                training_data["sentences"],
                                training_data["id"],
                                training_data["scores"],
                                training_data["context_sentences"],
                                training_data["context_id"],
                            ) = ("", "", "", "", "")
                            chunk_start = i * chunk_size
                            chunk_end = (
                                (i + 1) * chunk_size
                                if i < num_processes - 1
                                else int(len(self.df) / split)
                            )
                            args_list.append(
                                (
                                    group_by,
                                    plaintext_column,
                                    chunk_start,
                                    chunk_end,
                                    self.df,
                                    training_data,
                                    _model,
                                    _prompt,
                                    task,
                                )
                            )
                            _f(
                                "warn",
                                f"{i+1}/{num_processes} processes started from index {chunk_start} to {chunk_end}/{int(len(self.df) / split)} ({chunk_size})",
                            )

                        results = pool.map(_score_data_job, args_list)
                        training_data = pd.concat(results, ignore_index=True)
                else:
                    training_data = pd.DataFrame()
                    (
                        training_data["sentences"],
                        training_data["id"],
                        training_data["scores"],
                        training_data["context_sentences"],
                        training_data["context_id"],
                    ) = ("", "", "", "", "")
                    for i in range(int(len(self.df) / split)):
                        _score_data_job(
                            group_by,
                            plaintext_column,
                            i,
                            i + 1,
                            self.df,
                            training_data,
                            _model,
                            _prompt,
                            task,
                        )

                final_path = os.path.join(self.cleaned_dir, f"{self.filename}.json")
                training_data.to_json(final_path)
                self.df = training_data
                _f("success", f"saved to - {final_path}")
            except Exception as e:
                _f("fatal", e)
        else:
            return _f("fatal", "no data loaded!")

    def find_knn_neg(
        self,
        model: str = "BAAI/bge-large-en-v1.5",
        input_file: str = None,
        output_file: str = None,
        sample_range: list | str = [0 - 200],
        num_hard_negatives: int = 15,
        use_gpu: bool = False,
        prompt=None,
    ):
        """
        The function `find_knn_neg` takes in a model, input file, output file, sample range, number of
        hard negatives, use_gpu flag, and prompt as parameters. It loads data from the input file,
        encodes the corpus and queries using a pre-trained model, creates an index, performs a search,
        filters and selects hard negatives, and writes the updated data to the output file.

        :param model: The `model` parameter is a string that specifies the pre-trained model to use for
        sentence embedding. In this case, the default model is 'BAAI/bge-large-en-v1.5', defaults to
        BAAI/bge-large-en-v1.5
        :type model: str (optional)
        :param input_file: The `input_file` parameter is the name of the file that contains the input
        data for the model. It should be a JSONL file format
        :type input_file: str
        :param output_file: The `output_file` parameter is the name of the file where the updated
        training data will be saved. It should be a string representing the file name without the file
        extension
        :type output_file: str
        :param sample_range: The `sample_range` parameter is used to specify the range of indices to
        retrieve from the search results. It can be either a list or a string. If it is a list, it
        should contain two integers representing the start and end indices (inclusive) of the range. If
        it is a string
        :type sample_range: list|str
        :param num_hard_negatives: The parameter `num_hard_negatives` specifies the number of hard
        negative examples to be selected for each query, defaults to 15
        :type num_hard_negatives: int (optional)
        :param use_gpu: The `use_gpu` parameter is a boolean flag that indicates whether to use the GPU
        for encoding the embeddings. If set to `True`, the code will utilize the GPU for faster
        computation. If set to `False`, the code will use the CPU for encoding the embeddings, defaults
        to False
        :type use_gpu: bool (optional)
        :param prompt: The `prompt` parameter is a string that represents the prompt or query for which
        you want to find the nearest neighbors. It is used to generate a representation for the prompt
        that can be used to retrieve related articles
        :return: a success message along with the path of the output file where the modified train data
        is written.
        """
        try:
            _model = SentenceTransformer(model)
            corpus = []
            queries = []
            train_data = []
            for line in open(
                os.path.join(os.path.abspath(self.cleaned_dir), input_file)
            ):
                line = json.loads(line.strip())
                train_data.append(line)
                corpus.extend(line["neg"])
                queries.append(line["query"])

            corpus = list(set(corpus))

            _f(
                "wait",
                f"inferencing massive embedding for corpus index - {len(corpus)}",
            )
            p_vecs = _model.encode(corpus, batch_size=256)
            _f(
                "wait",
                f"inferencing massive embedding for search queries - {len(queries)}",
            )
            prompt = (
                "Generate a representation for this sentence that can be used to retrieve related articles："
                if prompt == None
                else prompt
            )
            q_vecs = _model.encode([prompt + q for q in queries], batch_size=256)

            _f("success", "create index and search")
            index = _create_index(p_vecs, use_gpu=use_gpu)
            _, all_inxs = _batch_search(index, q_vecs, topk=sample_range[-1])
            assert len(all_inxs) == len(train_data)

            for i, data in enumerate(train_data):
                query = data["query"]
                inxs = all_inxs[i][sample_range[0] : sample_range[1]]
                filtered_inx = []
                for inx in inxs:
                    if inx == -1:
                        break
                    if corpus[inx] not in data["pos"] and corpus[inx] != query:
                        filtered_inx.append(inx)

                if len(filtered_inx) > num_hard_negatives:
                    filtered_inx = random.sample(filtered_inx, num_hard_negatives)
                data["neg"] = [corpus[inx] for inx in filtered_inx]
            final_path = os.path.join(
                os.path.abspath(self.cleaned_dir), f"{output_file}.jsonl"
            )
            with open(final_path, "w") as f:
                for data in train_data:
                    if len(data["neg"]) < num_hard_negatives:
                        data["neg"].extend(
                            random.sample(corpus, num_hard_negatives - len(data["neg"]))
                        )
                    f.write(json.dumps(data) + "\n")
            return _f("success", f"written - {final_path}")
        except Exception as e:
            _f("fatal", e)
