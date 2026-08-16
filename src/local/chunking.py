
import os
import pickle
import numpy as np
import time
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer

from constants import FIXED, SENTENCE, SEMANTIC

class Chunking:
    def __init__(self, dataset, dataset_size, device, model_name):
        self.model_name = model_name
        self.dataset = dataset
        self.dataset_size = dataset_size
        self.device = device

        self.model = SentenceTransformer(self.model_name, device=self.device)
        self.tokenizer = self.model.tokenizer

        self.fixed_path = f"chunks/{dataset_size}_{FIXED}/"
        self.sentence_path = f"chunks/{dataset_size}_{SENTENCE}/"
        self.semantic_path = f"chunks/{dataset_size}_{SEMANTIC}/"


    def CHUNK(self, doc_id, title, chunk_id, chunk_text, chunk_size, chunking_type):
        return {
            'doc_id': doc_id,
            'title': title,
            'chunk_id': f"{doc_id}_{chunk_id}",
            'chunk_text': chunk_text,
            'chunk_size' : chunk_size,
            'chunking_type': chunking_type
        }

    def save(self, chunks, out_path):
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

        with open(out_path, 'wb') as f:
            pickle.dump(chunks, f)

    @staticmethod
    def _validate_overlap(max_chunk_size, fix_chunk_overlap):
        if fix_chunk_overlap >= max_chunk_size:
            raise ValueError(
                f"fix_chunk_overlap ({fix_chunk_overlap}) must be smaller than "
                f"max_chunk_size ({max_chunk_size}); otherwise the sliding window "
                "never advances."
            )

    # -------------- Shared fixed-window splitter --------------
    # Used directly by fixed_chunking, and as the fallback for any single sentence
    # that is longer than max_chunk_size on its own (sentence-aware/semantic chunking).
    def _split_fixed_window(self, doc_id, title, tokens, max_chunk_size, fix_chunk_overlap, chunking_type, start_chunk_id):
        self._validate_overlap(max_chunk_size, fix_chunk_overlap)

        chunks = {}
        chunk_sizes = []
        chunk_id = start_chunk_id
        start = 0

        while start < len(tokens):
            end = start + max_chunk_size
            # Get the tokens between start and end, chunk boundary
            chunk_tokens = tokens[start:end]
            # Get text from tokens. Decode
            chunk_text = self.tokenizer.decode(chunk_tokens, skip_special_tokens=True)

            # Make CHUNK
            ch = self.CHUNK(doc_id, title, chunk_id, chunk_text, len(chunk_tokens), chunking_type)
            chunks[ch['chunk_id']] = ch

            chunk_sizes.append(len(chunk_tokens))

            # To ensure overlap
            start += max_chunk_size - fix_chunk_overlap
            chunk_id += 1

        return chunks, chunk_sizes, chunk_id

    # -------------- 1. Fixed
    def fixed_chunking(self, doc_id, title, text, max_chunk_size, fix_chunk_overlap):
        s = time.time()

        # Get the tokens from the text
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        chunks, chunk_sizes, _ = self._split_fixed_window(
            doc_id, title, tokens, max_chunk_size, fix_chunk_overlap, FIXED, start_chunk_id=0
        )

        elapsed_time = time.time() - s
        return chunks, chunk_sizes, elapsed_time

    # -------------- Shared sentence-based chunker --------------
    # Backs both sentence-aware and semantic chunking. Sentences accumulate into a
    # chunk until either the token limit is hit, or should_split(prev_idx, curr_idx)
    # says the current sentence belongs in a new chunk (semantic chunking only).
    def _chunk_by_sentences(self, doc_id, title, sentences, sentence_tokens, max_chunk_size, fix_chunk_overlap, chunking_type, should_split=None):
        self._validate_overlap(max_chunk_size, fix_chunk_overlap)

        chunks = {}
        chunk_sizes = []
        current_chunk = []
        current_chunk_tokens = 0
        chunk_id = 0
        prev_idx = None

        def flush():
            nonlocal chunk_id
            chunk_text = ' '.join(current_chunk)
            # Re-tokenize the joined text so chunk_size reflects the chunk that was
            # actually produced, rather than the sum of its sentences' token counts.
            chunk_size = len(self.tokenizer.encode(chunk_text, add_special_tokens=False))
            ch = self.CHUNK(doc_id, title, chunk_id, chunk_text, chunk_size, chunking_type)
            chunks[ch['chunk_id']] = ch
            chunk_sizes.append(chunk_size)
            chunk_id += 1

        for i, sentence in enumerate(sentences):
            tokens = sentence_tokens[i]

            # 1. a sentence longer than max_chunk_size on its own: flush what we have,
            # then split just this sentence with the fixed-window fallback.
            if len(tokens) > max_chunk_size:
                if current_chunk:
                    flush()
                    current_chunk = []
                    current_chunk_tokens = 0

                sub_chunks, sub_sizes, chunk_id = self._split_fixed_window(
                    doc_id, title, tokens, max_chunk_size, fix_chunk_overlap, chunking_type, chunk_id
                )
                chunks.update(sub_chunks)
                chunk_sizes.extend(sub_sizes)

                # Move to the next sentence after splitting the long sentence into chunks
                prev_idx = None
                continue

            # 2. adding this sentence would exceed the token limit, or (semantic only)
            # it's too dissimilar from the previous sentence: finalize the current chunk.
            over_limit = current_chunk_tokens + len(tokens) > max_chunk_size
            extra_split = prev_idx is not None and should_split is not None and should_split(prev_idx, i)

            if current_chunk and (over_limit or extra_split):
                flush()
                current_chunk = []
                current_chunk_tokens = 0

            # 3. proceed normally: append the sentence to the current chunk
            current_chunk.append(sentence)
            current_chunk_tokens += len(tokens)
            prev_idx = i

        # Add the last chunk if it has content
        if current_chunk:
            flush()

        return chunks, chunk_sizes

    # ----- 2. Sentence aware chunking
    def sentence_aware_chunking(self, doc_id, title, sentences, sentence_tokens, max_chunk_size, fix_chunk_overlap):
        s = time.time()

        chunks, chunk_sizes = self._chunk_by_sentences(
            doc_id, title, sentences, sentence_tokens, max_chunk_size, fix_chunk_overlap, SENTENCE, should_split=None
        )

        elapsed_time = time.time() - s
        return chunks, chunk_sizes, elapsed_time

    # --------------- 3. Semantic chunking
    def semantic_chunking(self, doc_id, title, sentences, sentence_tokens, sentence_embeddings, max_chunk_size, fix_chunk_overlap, semantic_threshold):
        s = time.time()

        def should_split(prev_idx, curr_idx):
            embedding_1 = sentence_embeddings[prev_idx]
            embedding_2 = sentence_embeddings[curr_idx]
            denom = np.linalg.norm(embedding_1) * np.linalg.norm(embedding_2)
            if denom == 0:
                return False
            cosine_similarity = np.dot(embedding_1, embedding_2) / denom
            return cosine_similarity < semantic_threshold

        chunks, chunk_sizes = self._chunk_by_sentences(
            doc_id, title, sentences, sentence_tokens, max_chunk_size, fix_chunk_overlap, SEMANTIC, should_split=should_split
        )

        elapsed_time = time.time() - s
        return chunks, chunk_sizes, elapsed_time

    # -----------------------
    def compute_chunks(self, max_chunk_size, fix_chunk_overlap, semantic_threshold):
        self._validate_overlap(max_chunk_size, fix_chunk_overlap)

        path1 = self.fixed_path + f"size_{max_chunk_size}_overlap_{fix_chunk_overlap}.pkl"
        path2 = self.sentence_path + f"size_{max_chunk_size}_overlap_{fix_chunk_overlap}.pkl"
        path3 = self.semantic_path + f"size_{max_chunk_size}_overlap_{fix_chunk_overlap}_threashold_{semantic_threshold}.pkl"

        # If already exists, return that
        if os.path.exists(path1) and os.path.exists(path2) and os.path.exists(path3):
            print("Read chunks from disk")
            return path1, path2, path3

        try:
            # ---- Pass 1: sentence-tokenize each doc once (shared by sentence-aware
            # and semantic chunking), and collect one flat list of sentences across
            # the whole dataset so they can be embedded in a single batched call
            # instead of one small call per document. ----
            s0 = time.time()

            docs_meta = []
            per_doc_sentences = []
            per_doc_sentence_tokens = []
            sentence_doc_offsets = []
            all_sentences = []

            for doc in self.dataset:
                doc_id, title, text = doc['doc_id'], doc['title'], doc['text']
                try:
                    sentences = sent_tokenize(text)
                    sentence_tokens = [
                        self.tokenizer.encode(sentence, add_special_tokens=False) for sentence in sentences
                    ]
                except Exception as e:
                    print(f"WARNING: sentence tokenization failed for doc_id={doc_id}, skipping: {e}")
                    continue

                start = len(all_sentences)
                all_sentences.extend(sentences)

                docs_meta.append((doc_id, title, text))
                per_doc_sentences.append(sentences)
                per_doc_sentence_tokens.append(sentence_tokens)
                sentence_doc_offsets.append((start, len(all_sentences)))

            if all_sentences:
                all_embeddings = np.asarray(
                    self.model.encode(all_sentences, batch_size=256, show_progress_bar=False)
                )
            else:
                all_embeddings = np.empty((0, 0))

            time0 = time.time() - s0

            # ---- Pass 2: build all three chunkings per document ----
            fixed_chunks, sentence_chunks, semantic_chunks = {}, {}, {}
            fixed_chunk_sizes, sentence_chunk_sizes, semantic_chunk_sizes = [], [], []
            time1 = time2 = time3 = 0.0
            skipped_docs = 0

            for idx, (doc_id, title, text) in enumerate(docs_meta):
                try:
                    c11, c12, t1 = self.fixed_chunking(doc_id, title, text, max_chunk_size, fix_chunk_overlap)

                    sentences = per_doc_sentences[idx]
                    sentence_tokens = per_doc_sentence_tokens[idx]
                    start, end = sentence_doc_offsets[idx]
                    doc_embeddings = all_embeddings[start:end]

                    c21, c22, t2 = self.sentence_aware_chunking(
                        doc_id, title, sentences, sentence_tokens, max_chunk_size, fix_chunk_overlap
                    )
                    c31, c32, t3 = self.semantic_chunking(
                        doc_id, title, sentences, sentence_tokens, doc_embeddings,
                        max_chunk_size, fix_chunk_overlap, semantic_threshold
                    )
                except Exception as e:
                    print(f"WARNING: chunking failed for doc_id={doc_id}, skipping: {e}")
                    skipped_docs += 1
                    continue

                fixed_chunks.update(c11)
                fixed_chunk_sizes.extend(c12)
                time1 += t1

                sentence_chunks.update(c21)
                sentence_chunk_sizes.extend(c22)
                time2 += t2

                semantic_chunks.update(c31)
                semantic_chunk_sizes.extend(c32)
                time3 += t3

            if skipped_docs:
                print(f"Skipped {skipped_docs} document(s) that failed chunking")

            # Save
            self.save(fixed_chunks, path1)
            self.save(sentence_chunks, path2)
            self.save(semantic_chunks, path3)

            # Size of the files
            fixed_size = os.path.getsize(path1) / (1024**2)
            sentence_size = os.path.getsize(path2) / (1024**2)
            semantic_size = os.path.getsize(path3) / (1024**2)

            print("Avg chunk sizes : fixed, sentence, semantic : ", np.mean(fixed_chunk_sizes), np.mean(sentence_chunk_sizes), np.mean(semantic_chunk_sizes))
            print("Sentence tokenization + batch embedding time : ", time0)
            print("Total time for chunks : fixed, sentence, semantic : ", time1, time2, time3)
            print("File sizes : fixed, sentence, semantic : ", fixed_size, sentence_size, semantic_size)

            return path1, path2, path3

        except Exception as e:
            print(f"Chunking failed: {e}")
            # remove file if it was written before the failure
            for path in (path1, path2, path3):
                if os.path.exists(path):
                    os.remove(path)

            return None, None, None
