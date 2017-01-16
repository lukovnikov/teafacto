import pandas as pd
import seaborn as sb
import numpy as np
import matplotlib.pyplot as plt
from IPython import embed


##################### ATTENTION WEIGHTS PLOTTING #######################
class AttentionPlotter(object):

    @classmethod
    def plot(cls, weights, srcseq=None, dstseq=None, cmap="Greys", scale=1.):
        assert(weights.ndim == 2)
        if srcseq is not None:
            assert(weights.shape[0] == len(srcseq))
        else:
            srcseq = range(weights.shape[0])
        if dstseq is not None:
            assert(weights.shape[1] == len(dstseq))
        else:
            dstseq = range(weights.shape[1])

        yticks = srcseq
        xticks = dstseq

        sb.set_style(style="white")
        sb.set(font_scale=1.5 * scale)

        height = weights.shape[0] * scale * 0.5
        width = weights.shape[1] * scale * 0.5
        margin = 0.2

        #f, ax = plt.subplots(figsize=(width, height),
        #                     gridspec_kw={"top": 1 - margin, "bottom": margin})

        ax = sb.heatmap(weights, cmap=cmap, square=True, linewidths=1.*scale,
                   yticklabels=yticks, xticklabels=xticks, vmax=1., vmin=0.,
                   cbar=False)


        plt.yticks(rotation=0)
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    s = "the quick brown fox jumped over the lazy dog".split()
    d = "de snelle bruine vos sprong over de luie hond".split()
    w = np.random.random((len(s), len(d)))
    AttentionPlotter.plot(w, s, d, scale=1.)



