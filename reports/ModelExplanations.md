# Model Architectures: EfficientNet-B0 + Temporal Attention and ResNet18 + BiGRU

---

## Model001 — EfficientNet-B0 + Temporal Attention

### Core Idea

How do humans identify a music genre? They don't just look at the "overall tone" — they focus on specific moments. A guitar riff, a drum break, a bass line. `model001` mathematically formalizes exactly this intuition.

The model consists of two major components: a visual feature extractor called **EfficientNet-B0**, and a **temporal attention** layer placed on top of it.

---

### EfficientNet-B0: Why This Backbone?

Spectrograms are photographs of audio signals — time on the horizontal axis, frequency on the vertical axis, energy encoded in pixel intensity. This means we're essentially dealing with an image classification problem, which makes it sensible to use visual models pretrained on ImageNet as a starting point.

EfficientNet-B0 is one of the standout models in the efficiency/performance trade-off. The code ships with `freeze_until=0` as the default, meaning all layers are open for fine-tuning. Alternatively, setting something like `freeze_until=4` freezes the early layers and lets only the deeper layers learn — useful for preventing overfitting when data is scarce.

The backbone output is a feature map of shape `(B, 1280, H', W')`. Here 1280 is the number of learned channels; H' and W' are the spatial dimensions (roughly 7×7 for a 224×224 input).

---

### Transitioning to the Temporal Dimension

The width of a spectrogram corresponds to time. If we interpret the width dimension of the feature map as "time steps," we can process the features extracted by the CNN as a sequence.

That's exactly what `model001` does:

```
(B, 1280, H', W')
    → AdaptiveAvgPool2d((7, 1))   # collapse width to 1
    → (B, 1280, 7, 1)             # 7 "time slices" remain
    → squeeze + permute
    → (B, 7, 1280)                # a sequence of 7 steps
    → Linear projection
    → (B, 7, 256)                 # reduced to a smaller dimension
```

We now have a sequence that divides the spectrogram into 7 segments. Each segment is a feature vector extracted by the CNN for that time slice.

---

### Multi-Head Self-Attention

Two `TemporalAttention` layers are applied. Each layer does the following: every time step "talks" to every other time step, and learns how relevant each one is to itself. As a result, some time slices receive more weight, others less.

This mechanism uses multiple heads (`num_heads=4`); each head can learn a different "relational pattern." One head might capture rhythmic repetitions, another melodic structure, and another dynamic changes.

In the final step, the 7 time steps are averaged (`mean(dim=1)`) and passed to a classifier head.

---

### Why Choose This Model?

- **Transfer learning advantage:** Instead of starting from scratch, it leverages rich features learned from millions of images.
- **Long-range dependencies:** Pure CNNs capture local patterns; attention can model relationships between the beginning and end of a spectrogram.
- **Interpretability:** By examining the attention weights, one can ask "which part of this piece did the model focus on?"
- **Efficient dimensionality:** The projection from 1280 to 256 significantly reduces the computational cost of attention.

---

## Model002 — ResNet18 + Bidirectional GRU + Attention Pooling

### Core Idea

`model002` focuses on the *directional* and *selective* interpretation of the musical time series. Two components serve this purpose: a **BiGRU** (bidirectional recurrent network) and **attention pooling**.

---

### ResNet18: Why a Different Backbone?

While `model001` uses EfficientNet, `model002` opts for ResNet18. ResNet18 is a simpler and older architecture, but it comes with a few advantages:

- **Fewer parameters** — faster training and lower overfitting risk
- **More controlled feature map size** — the final layer produces 512 channels, which is ideal to feed into a GRU
- **Well-understood behavior** — its effectiveness on spectrograms is widely validated

The code strips the last two layers (avgpool and fc) with `backbone = nn.Sequential(*list(backbone.children())[:-2])`, keeping only the feature extractor. Output shape: `(B, 512, H', W')`.

---

### Collapsing the Frequency Dimension

In a spectrogram, the vertical axis is frequency and the horizontal axis is time. `model002` collapses the frequency dimension into a single vector using `AdaptiveAvgPool2d((1, None))`. This yields:

```
(B, 512, H', W')
    → (B, 512, 1, W')    # frequency averaged out
    → squeeze → permute
    → (B, W', 512)       # W' time steps, each 512-dimensional
```

We now have a sequence of 512-dimensional feature vectors, one for each time slice of the spectrogram.

---

### Bidirectional GRU: Understanding in Two Directions

GRU (Gated Recurrent Unit) is the lighter counterpart of LSTM. It carries a "hidden state" as it passes over the sequence, selectively retaining or discarding the influence of previous steps at each point.

Being **bidirectional** is critical: one GRU runs from the beginning of the sequence to the end, the other runs from the end to the beginning. Their outputs are concatenated (`gru_hidden * 2`). This allows the model to evaluate "the current time slice" in light of both past and future context. For instance, in a jazz piece, what comes *before* and *after* a saxophone solo can serve as context clues for that moment.

The code uses two GRU layers (`gru_layers=2`); the output of the first layer feeds into the second, enabling more complex temporal patterns to be learned.

---

### Attention Pooling: Which Moments Matter?

The BiGRU produces vectors for all time steps with shape `(B, W', hidden*2)`. To reduce these to a single representation, there are two options:

1. **Take the mean** — combine all with equal weight (what model001 does)
2. **Attention pooling** — assign an importance score to each time step and compute a weighted sum (what model002 does)

The `AttentionPooling` module uses a `Linear(hidden, 1)` layer: it maps each time step to a scalar, then normalizes with softmax. Time steps where this scalar is high receive more weight. The model *learns* which "moments" are decisive for the genre decision.

---

### Comparison with model001

| Feature | model001 | model002 |
|---|---|---|
| Backbone | EfficientNet-B0 (1280 ch) | ResNet18 (512 ch) |
| Temporal processing | Multi-head self-attention | Bidirectional GRU |
| Aggregation | Simple mean | Attention pooling |
| Context direction | Every step attends to all others | Bidirectional sequential flow |
| Parameter count | Higher | Lower |
| Strongest at | Global patterns, long-range dependencies | Sequential development, rhythm, phase transitions |

---

### Design Philosophy

Both models solve the same problem with different "cognitive strategies":

**model001** works like looking at a photograph: it sees the entire spectrogram at once and uses self-attention to compute which regions relate to which others. This is a global perspective — "what is the overall tonal identity of this piece?"

**model002** listens like a musician: it follows the audio through time, evaluating what came before and what comes after simultaneously. The BiGRU's bidirectional structure means "the context of this moment" is fed by both past and future. Then attention pooling answers the question: "which moments should I focus on?"

In practice, running both models and comparing their results — or ensembling them — is a strong strategy, and `run_all.py` already supports this by accepting both in the `models_list` config field.
