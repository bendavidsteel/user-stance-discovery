import glob
import logging
import os

import hydra
import polars as pl
import transformers

from stancemining import StanceMining

# The classifier head was tuned at 2048 tokens, of which the prompt template and the
# target take up to ~150.
MAX_MODEL_LEN = 2048
MAX_DOCUMENT_TOKENS = 1400
MAX_PARENT_TOKENS = 400
TOKENIZE_CHUNK = 50_000

SENTENCE_END = r'(?s)^(.*[.!?…。！？])(?:\s|$)'
LINE_END = r'(?s)^(.*)\n'
WORD_END = r'(?s)^(.*)\s'
# a boundary further back than this discards more than the ragged edge is worth
MIN_KEPT_FRACTION = 0.5


def stance_key(target_column: str = 'Target') -> pl.Expr:
    """Key a classification by everything the model sees: text, parent and target."""
    return pl.concat_str([
        pl.col('Document').fill_null(''),
        pl.col('ParentDocument').fill_null(''),
        pl.col(target_column).fill_null(''),
    ], separator='\x00').hash(seed=0)


def blank_parent_to_null() -> pl.Expr:
    """A blank parent is no parent; left as '' it renders a chain of empty quotes."""
    return pl.when(pl.col('ParentDocument').str.strip_chars().str.len_chars() > 0)\
        .then(pl.col('ParentDocument')).otherwise(None).alias('ParentDocument')


def truncate_to_sentence(df: pl.DataFrame, column: str, max_tokens: int, tokenizer, logger) -> pl.DataFrame:
    """Cut texts over the token budget back to the last sentence boundary that fits.

    A byte level BPE token always covers at least one UTF-8 byte, so a text shorter
    than the budget in bytes is already inside it and never has to be tokenized.
    """
    candidates = df.select(pl.col(column).alias('text')).drop_nulls()\
        .filter(pl.col('text').str.len_bytes() > max_tokens)\
        .unique('text')
    if candidates.is_empty():
        logger.info(f'No {column} values exceed {max_tokens} tokens.')
        return df

    texts = candidates['text'].to_list()
    over_texts, cut_texts = [], []
    for start in range(0, len(texts), TOKENIZE_CHUNK):
        chunk = texts[start:start + TOKENIZE_CHUNK]
        encoded = tokenizer(chunk, add_special_tokens=False)['input_ids']
        over = [i for i, ids in enumerate(encoded) if len(ids) > max_tokens]
        if not over:
            continue
        over_texts.extend(chunk[i] for i in over)
        cut_texts.extend(tokenizer.batch_decode([encoded[i][:max_tokens] for i in over]))

    logger.info(f'Truncating {len(over_texts)} of {len(texts)} distinct long {column} values to {max_tokens} tokens.')
    if not over_texts:
        return df

    # decoding a cut token run can end part way through a character
    cut = pl.col('cut').str.strip_chars_end('�').str.strip_chars()
    boundary = pl.coalesce([
        cut.str.extract(SENTENCE_END, 1),
        cut.str.extract(LINE_END, 1),
        cut.str.extract(WORD_END, 1),
    ]).str.strip_chars()
    mapper = pl.DataFrame({'text': over_texts, 'cut': cut_texts})\
        .with_columns(pl.when(boundary.str.len_chars() >= MIN_KEPT_FRACTION * cut.str.len_chars())
                        .then(boundary).otherwise(cut).alias('cut'))

    return df.join(mapper.rename({'text': column}), on=column, how='left')\
        .with_columns(pl.coalesce(['cut', column]).alias(column))\
        .drop('cut')


def load_previous_stances(previous_path: str, logger) -> pl.DataFrame:
    """Classifications from an earlier run, keyed on the text they were made from."""
    empty = pl.DataFrame(schema={'key': pl.UInt64, 'Stance': pl.String})
    files = sorted(glob.glob(os.path.join(previous_path, '*.parquet.zstd'))) if previous_path else []
    if not files:
        logger.info(f'No previous stance files to reuse at {previous_path}.')
        return empty

    cache_df = pl.scan_parquet(files).select(['Document', 'ParentDocument', 'Targets', 'Stances'])\
        .explode(['Targets', 'Stances']).drop_nulls(['Targets', 'Stances'])\
        .filter(pl.col('ParentDocument').is_null()
                | (pl.col('ParentDocument').str.strip_chars().str.len_chars() > 0))\
        .select(stance_key('Targets').alias('key'), pl.col('Stances').alias('Stance'))\
        .unique('key').collect()
    logger.info(f'Loaded {cache_df.height} reusable classifications from {len(files)} files in {previous_path}.')
    return cache_df


def write_week_stance(week_df: pl.DataFrame, pair_df: pl.DataFrame, miner: StanceMining,
                      week_batch_path: str, week_probs_path: str, logger) -> None:
    todo_df = pair_df.filter(pl.col('Stance').is_null())
    if not todo_df.is_empty():
        # one classification per distinct (text, parent, target), not per document
        work_df = todo_df.unique('key')\
            .join(week_df.select(['row_id', 'Document', 'ParentDocument']), on='row_id', how='left')\
            .group_by(['Document', 'ParentDocument'])\
            .agg(pl.col('Target').alias('Targets'))
        logger.info(f'Classifying {todo_df.height} pairs as {work_df.height} documents, '
                    f'reusing {pair_df.height - todo_df.height}.')
        stance_df = miner.get_stance(work_df, text_column='Document',
                                     parent_text_column='ParentDocument', return_probs=True)
        computed_df = stance_df.explode(['Targets', 'Stances', 'Probs']).drop_nulls('Targets')\
            .rename({'Targets': 'Target'})\
            .select(stance_key().alias('key'), pl.col('Stances').alias('NewStance'),
                    pl.col('Probs').alias('NewProbs'))\
            .unique('key')
        pair_df = pair_df.join(computed_df, on='key', how='left')\
            .with_columns(pl.coalesce(['Stance', 'NewStance']).alias('Stance'),
                          pl.col('NewProbs').alias('Probs'))\
            .drop(['NewStance', 'NewProbs'])

    missing = pair_df['Stance'].null_count()
    if missing:
        logger.warning(f'{missing} of {pair_df.height} pairs came back without a stance.')

    agg_columns = [pl.col('Target').alias('Targets'), pl.col('Stance').alias('Stances')]
    if 'Probs' in pair_df.columns:
        agg_columns.append(pl.col('Probs').alias('Probs'))
    stance_df = pair_df.group_by('row_id').agg(agg_columns)
    week_stance_df = week_df.drop('Targets').join(stance_df, on='row_id', how='left')\
        .with_columns(pl.col('Targets').fill_null([]), pl.col('Stances').fill_null([]))

    # kept beside the stance files rather than in them: adding a column partway through a
    # run leaves the directory with two schemas, which every reader would have to handle
    if 'Probs' in week_stance_df.columns:
        os.makedirs(os.path.dirname(week_probs_path), exist_ok=True)
        week_stance_df.select(['id', 'platform', 'Targets', pl.col('Probs').fill_null([])])\
            .write_parquet(week_probs_path, compression='zstd')
        week_stance_df = week_stance_df.drop('Probs')

    # written last, so its presence still means the whole week is done
    week_stance_df.drop(['row_id', 'year', 'week'])\
        .write_parquet(week_batch_path, compression='zstd')


@hydra.main(version_base=None, config_path="../../config", config_name="config")
def main(config):
    logger = logging.getLogger('get_stance')
    pl.set_random_seed(42)

    period = '2022-01-01-onwards'
    document_df = pl.read_parquet(
        f'./data/stance_targets/{period}_{config.stance_target_type}_doc_targets.parquet.zstd',
        columns=['id', 'Document', 'ParentDocument', 'createtime', 'seed', 'Targets', 'finetune_kwargs', 'platform'],
    )
    logger.info(f'Loaded {document_df.height} documents.')

    # TODO train gemma3 sequence classification when this is in transformers release that vllm supports: https://github.com/huggingface/transformers/pull/39465

    if config.stance_target_type == 'claims':
        stance_detection_finetune_kwargs = {
            'model_path': '/home/ndg/users/bsteel2/repos/stancemining/models/stancemining/Qwen-Qwen3-0.6B-claim-entailment-7way-stanceosaurus-head-merged',
            'classification_method': 'head',
        }
    elif config.stance_target_type == 'noun-phrases':
        stance_detection_finetune_kwargs = {
            'model_path': '/home/ndg/users/bsteel2/repos/stancemining/models/stancemining/Qwen-Qwen3.5-4B-stance-classification-vast-ezstance-pstance-semeval-mtcsd-ctsdt-catalonia-french-election-head-merged',
            'classification_method': 'head',
        }
    else:
        raise ValueError(f"Unknown stance_target_type: {config.stance_target_type}")

    # classification is prefill only and already saturates the GPU at vLLM's default
    # batch limits; raising max_num_batched_tokens or max_num_seqs only slows it down
    stance_detection_model_kwargs = {
        'max_model_len': MAX_MODEL_LEN,
        'gpu_memory_utilization': float(config.get('stance_gpu_memory_utilization', 0.85)),
        # experimental for this model's linear attention layers, and it silently
        # returns content belonging to other prompts
        'enable_prefix_caching': False,
    }

    tokenizer = transformers.AutoTokenizer.from_pretrained(stance_detection_finetune_kwargs['model_path'])

    document_df = document_df.with_columns(pl.col('createtime').dt.iso_year().alias('year'),
                                           pl.col('createtime').dt.week().alias('week'))
    week_df = document_df.select(['year', 'week']).unique().sort(['year', 'week'], descending=True)

    # one process per GPU, each taking every nth week, so neither touches the other's
    # output file and each only has to hold its own share of the corpus
    num_shards = int(config.get('stance_num_shards', 1))
    shard = int(config.get('stance_shard', 0))
    if num_shards > 1:
        week_df = week_df.with_row_index('week_index')\
            .filter(pl.col('week_index') % num_shards == shard).drop('week_index')
        document_df = document_df.join(week_df, on=['year', 'week'], how='semi')
        logger.info(f'Shard {shard} of {num_shards}: {len(week_df)} weeks, {document_df.height} documents.')

    # trim to the prompt budget rather than dropping long documents outright
    slim_df = document_df.select(['Document', 'ParentDocument']).with_row_index('row_id')\
        .with_columns(blank_parent_to_null())
    slim_df = truncate_to_sentence(slim_df, 'Document', MAX_DOCUMENT_TOKENS, tokenizer, logger)
    slim_df = truncate_to_sentence(slim_df, 'ParentDocument', MAX_PARENT_TOKENS, tokenizer, logger)
    document_df = document_df.drop(['Document', 'ParentDocument']).with_row_index('row_id')\
        .join(slim_df, on='row_id', how='left')
    del slim_df

    pair_df = document_df.select(['row_id', 'Document', 'ParentDocument', 'Targets'])\
        .explode('Targets').drop_nulls('Targets').rename({'Targets': 'Target'})\
        .select(['row_id', 'Target', stance_key().alias('key')])\
        .join(load_previous_stances(config.get('previous_stance_path'), logger), on='key', how='left')\
        .join(document_df.select(['row_id', 'year', 'week']), on='row_id', how='left')
    logger.info(f'{pair_df.height - pair_df["Stance"].null_count()} of {pair_df.height} '
                f'document-target pairs already classified by a previous run.')

    miner = StanceMining(
        verbose=True,
        stance_target_type=config.stance_target_type,
        stance_detection_finetune_kwargs=stance_detection_finetune_kwargs,
        stance_detection_model_kwargs=stance_detection_model_kwargs,
    )

    # batch out calls
    os.makedirs(config.base_stance_path, exist_ok=True)
    doc_parts = document_df.partition_by(['year', 'week'], as_dict=True)
    pair_parts = pair_df.partition_by(['year', 'week'], as_dict=True)
    # a week whose documents all had empty target lists has no pairs at all
    no_pairs = pair_df.clear()
    del document_df, pair_df

    probs_path = f'{config.base_stance_path}_probs'
    for i, week in enumerate(week_df.to_dicts()):
        week_batch_path = f'{config.base_stance_path}/{week["year"]}_{week["week"]}_doc_targets_with_stance.parquet.zstd'
        week_probs_path = f'{probs_path}/{week["year"]}_{week["week"]}_doc_targets_stance_probs.parquet.zstd'
        if os.path.exists(week_batch_path):
            continue
        logger.info(f'Processing week {week["week"]} of year {week["year"]}, {i + 1} of {len(week_df)} weeks')
        part_key = (week['year'], week['week'])
        write_week_stance(doc_parts[part_key], pair_parts.get(part_key, no_pairs),
                          miner, week_batch_path, week_probs_path, logger)


if __name__ == "__main__":
    main()
