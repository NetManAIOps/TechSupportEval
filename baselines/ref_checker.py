import warnings

warnings.filterwarnings("ignore")

def init_config(model, args):
    disable_tqdm()

    from refchecker import LLMExtractor, LLMChecker

    global extractor, checker
    extractor = LLMExtractor(model=model, batch_size=8)
    checker = LLMChecker(model=model, batch_size=8)

def evaluate(question, ground_truth, answer):
    from refchecker.aggregator import soft_agg

    extraction_results = extractor.extract(
        batch_responses=[ground_truth],
        batch_questions=[question],
        max_new_tokens=1000,
    )
    batch_claims = [[c.content for c in res.claims] for res in extraction_results]

    batch_labels = checker.check(
        batch_claims=batch_claims,
        batch_references=[answer],
        max_reference_segment_length=0,
        is_joint=False,
    )
    
    # Entailment Ratio
    labels = [item[0] for item in batch_labels[0]]
    res = soft_agg(labels)
    score = res["Entailment"]
    return score, {'detail': res}

def disable_tqdm():
    import tqdm

    class NoOpTqdm:
        def __init__(self, iterable=None, *args, **kwargs):
            self.iterable = iterable if iterable is not None else []
        
        def __iter__(self):
            return iter(self.iterable)
        
        def update(self, *args, **kwargs):
            pass
        
        def close(self):
            pass

        def __enter__(self):
            return self
        
        def __exit__(self, exc_type, exc_value, traceback):
            pass

    tqdm.tqdm = NoOpTqdm
