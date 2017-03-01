from teafacto.core.base import tensorops as T

# from glample/tagger


def log_sum_exp(x, axis=None):
    """
    Sum probabilities in the log-space.
    """
    xmax = x.max(axis=axis, keepdims=True)
    xmax_ = x.max(axis=axis)
    return xmax_ + T.log(T.exp(x - xmax).sum(axis=axis))


def forward(observations, transitions, startsymbol=0, viterbi=False,
            return_alpha=False, return_best_sequence=False):
    """
    Takes as input:
        - observations  (batsize, n_steps, n_classes), masked
        - transitions   (n_classes, n_classes)
    Probabilities must be given in the log space.
    Compute alpha, (batsize, n_steps, n_classes), such that
    alpha[?, i, j] represents one of these 2 values:
        - the probability that the real path at node i ends in j
        - the maximum probability of a path finishing in j at node i (Viterbi)
    Returns one of these 2 values:
        - alpha
        - the final probability, which can be:
            - the sum of the probabilities of all paths
            - the probability of the best path (Viterbi)
    """
    assert not return_best_sequence or (viterbi and not return_alpha)
    small = -1e3

    def recurrence(obs,             # (batsize, n_classes)
                   previous,        # (batsize, n_classes)
                   trans,           # (n_classes, n_classes)
                   ):
        previous = previous.dimshuffle(0, 1, 'x')   # (batsize, n_classes, 1)
        obs = obs.dimshuffle(0, 'x', 1)             # (batsize, 1, n_classes)
        trans = trans.dimshuffle('x', 0, 1)         # (1, n_classes, n_classes)
        if viterbi:
            scores = previous + obs + trans
            out = scores.max(axis=1)
            if return_best_sequence:
                out2 = scores.argmax(axis=1)
                return out, out2
            else:
                return out
        else:
            scores = previous + obs + trans
            return log_sum_exp(scores, axis=1)

    def maskedrecurrence(obs,       # (batsize, n_classes)
                         mask,      # (batsize,)
                         previous,  # (batsize, n_classes)
                         trans,     # (n_classes, n_classes)
                         ):
        prev = previous.dimshuffle(0, 1, 'x')   # (batsize, n_classes, 1)
        obs = obs.dimshuffle(0, 'x', 1)             # (batsize, 1, n_classes)
        trans = trans.dimshuffle('x', 0, 1)         # (1, n_classes, n_classes)
        mask = mask.dimadd(1)
        scores = prev + obs + trans             # (batsize, n_classes, n_classes)
        ret2 = None
        if viterbi:
            ret = scores.max(axis=1)
            if return_best_sequence:
                ret2 = scores.argmax(axis=1)    # (batsize, n_classes)
                if mask is not None:
                    ret2 = T.switch(mask, ret2, T.zeros_like(ret2, dtype="int32"))
        else:
            ret = log_sum_exp(scores, axis=1)
        ret = ret * mask + (1 - mask) * previous
        if ret2 is None:
            return ret
        else:
            return ret, ret2

    mask = observations.mask

    initial = T.concatenate([
        T.ones((observations.shape[0], startsymbol)) * small,
        T.ones((observations.shape[0], 1)) * 0.0,
        T.ones((observations.shape[0], observations.shape[2] - startsymbol - 1)) * small
        ], axis=1)

    f = recurrence if mask is None else maskedrecurrence
    seqs = [observations.dimswap(0, 1)] if mask is None \
                    else [observations.dimswap(0, 1), mask.dimswap(0, 1)]

    alpha = T.scan(
        fn=f,
        outputs_info=[initial, None] if return_best_sequence else [initial],
        sequences=seqs,
        non_sequences=transitions
    )
    if return_best_sequence:
        alpha, bestseq = alpha

    alpha = alpha.dimswap(0, 1)     # (batsize, n_steps, n_classes)
    alpha.mask = mask

    if return_alpha:
        return alpha
    elif return_best_sequence:

        def bestrec(beta_i, prev):
            ret = beta_i[T.arange(0, beta_i.shape[0]), prev]
            return ret

        def maskedbestrec(beta_i, mask, prev):
            ret = beta_i[T.arange(0, beta_i.shape[0]), prev]
            ret = T.switch(mask, ret, prev)
            return ret

        f = maskedbestrec if mask is not None else bestrec
        outs = [T.cast(T.argmax(alpha[:, -1], axis=1), "int32")]
        seqs = [T.cast(bestseq[::-1], "int32")] if mask is None else \
            [T.cast(bestseq[::-1], "int32"), mask.dimswap(0, 1)[::-1]]
        #return outs[0]
        sequence = T.scan(
            fn=f,
            outputs_info=outs,
            sequences=seqs
        )
        sequence = T.concatenate([sequence[::-1],
                                  [T.argmax(alpha[:, -1], axis=1)]])
        seqmask = T.concatenate([T.ones((1, mask.shape[0]), dtype="int8"), mask], axis=1)
        sequence = sequence.dimswap(0, 1)
        #seqmask = seqmask.dimswap(0, 1)
        sequence.mask = seqmask
        return sequence     # (batsize, seqlen)
    else:
        if viterbi:
            return alpha[:, -1].max(axis=1)
        else:
            return log_sum_exp(alpha[:, -1, :], axis=1)

