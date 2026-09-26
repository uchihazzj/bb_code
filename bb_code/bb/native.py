"""Native T5 state capture, bound specifically to Transformers 5.9.0."""
import copy
from types import MethodType
import numpy as np
import torch
import transformers
from .catalog import CatalogMask, ClosureObserver, pack
W, B, V = 40, 256, 1025
SETTINGS = dict(max_new_tokens=4, num_beams=W, num_return_sequences=W,
                do_sample=False, early_stopping=False, use_cache=True,
                renormalize_logits=False, length_penalty=1., repetition_penalty=1.,
                no_repeat_ngram_size=0)

class Captured(Exception):
    pass


class Capture(ClosureObserver):
    def install(self):
        names = ("_get_running_beams_for_next_iteration", "_update_model_kwargs_for_generation")
        if any(name in self.model.__dict__ for name in names):
            raise RuntimeError("native hooks already present")
        running_original = self.model._get_running_beams_for_next_iteration
        update_original = self.model._update_model_kwargs_for_generation

        def running(module, *args, **kwargs):
            result = running_original(*args, **kwargs)
            self.live = result[1].detach().clone()
            self.running = result
            return result

        def update(module, *args, **kwargs):
            result = update_original(*args, **kwargs)
            self.kwargs = result
            return result

        self.model._get_running_beams_for_next_iteration = MethodType(running, self.model)
        self.model._update_model_kwargs_for_generation = MethodType(update, self.model)

    def remove(self):
        for name in ("_get_running_beams_for_next_iteration", "_update_model_kwargs_for_generation"):
            if name in self.model.__dict__:
                delattr(self.model, name)

    def __call__(self, input_ids, scores):
        masked = super().__call__(input_ids, scores)
        if input_ids.shape[1] == 3:
            self.parent_ids = input_ids.detach().clone()
            self.raw_scores = scores.detach().clone()
            raise Captured()
        return masked


def capture_parent(model, input_ids, old_paths):
    if transformers.__version__ != "5.9.0":
        raise RuntimeError("native hook binding requires Transformers 5.9.0")
    mask = CatalogMask(old_paths, full=True, device=model.device)
    observer = Capture(model, mask, old_paths, len(input_ids), W, model.device)
    observer.install()
    try:
        model.generate(input_ids=input_ids, attention_mask=input_ids.ne(0),
                       logits_processor=[observer], **SETTINGS)
    except Captured:
        return observer
    finally:
        observer.remove()
    raise RuntimeError("did not capture the native third-position state")


def third_candidates(capture, local_query, old_paths):
    parent = capture.parent_ids.reshape(capture.batch, W, 3)[local_query, :, 1:].cpu().numpy()
    parent_codes = pack(parent)
    if not np.isin(parent_codes, np.unique(pack(old_paths[:, :2]))).all():
        raise RuntimeError("target parent beam is not all old")
    values = (capture.raw_scores.reshape(capture.batch, W, V)[local_query, :, 513:769]
              + capture.live[local_query, :, None]).cpu().numpy()
    children = (parent_codes[:, None] * B + np.arange(B)).reshape(-1)
    mandatory = np.isin(children, np.unique(pack(old_paths[:, :3])))
    if not np.isfinite(values).all():
        raise RuntimeError("nonfinite penultimate scores")
    return parent, values, mandatory, children


def transition(model, capture, local_query=None, critical=(), subset=0):
    """Use native vocabulary layout and both native top-k selections."""
    masked = torch.full_like(capture.raw_scores, -torch.inf).reshape(capture.batch, W, V)
    masked[:, :, 513:769] = capture.raw_scores.reshape(capture.batch, W, V)[:, :, 513:769]
    if local_query is not None:
        excluded = [c for j, c in enumerate(critical) if not subset & (1 << j)]
        if excluded:
            index = torch.tensor(excluded, device=model.device)
            masked[local_query, index // B, 513 + index % B] = -torch.inf
    accumulated = (masked + capture.live[:, :, None]).reshape(capture.batch, W * V)
    scores, sequences, history = model._get_top_k_continuations(
        accumulated_log_probs=accumulated, running_sequences=capture.running[0],
        running_beam_indices=capture.running[2], cur_len=3, decoder_prompt_len=1,
        do_sample=False, beams_to_keep=2 * W, num_beams=W, vocab_size=V,
        batch_size=capture.batch)
    # SID3 is never EOS and there are at least W finite live candidates.
    stopped = sequences[:, :, 3].eq(model.config.eos_token_id)
    result = model._get_running_beams_for_next_iteration(
        topk_log_probs=scores, topk_running_sequences=sequences,
        topk_running_beam_indices=history, next_token_hits_stopping_criteria=stopped,
        num_beams=W)
    if not torch.isfinite(result[1]).all():
        raise RuntimeError("invalid live beam after native transition")
    return result


def final_step(model, capture, state):
    kwargs = dict(capture.kwargs)
    # The native loop reorders this complete cache before feeding SID3. Each branch
    # gets its own clone so the shared pre-branch cache cannot be mutated by a sibling.
    kwargs["past_key_values"] = copy.deepcopy(capture.kwargs["past_key_values"])
    parent_indices = model._flatten_beam_dim(state[2][..., 2])
    if hasattr(model, "_reorder_cache"):
        kwargs["past_key_values"] = model._reorder_cache(kwargs["past_key_values"], parent_indices)
    else:
        kwargs["past_key_values"].reorder_cache(parent_indices)
    flat_ids = model._flatten_beam_dim(state[0][:, :, :4])
    inputs = model.prepare_inputs_for_generation(flat_ids, next_sequence_length=1, **kwargs)
    outputs = model(**inputs, return_dict=True)
    logits = outputs.logits[:, -1, :].to(copy=True, dtype=torch.float32, device=model.device)
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    cumulative = (log_probs.reshape(capture.batch, W, V)[:, :, 769:1025]
                  + state[1][:, :, None])
    return cumulative
