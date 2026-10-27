# Configuration

Every run is controlled by one YAML file. Back to the [README](../README.md).

See [`sample/config.yml`](../sample/config.yml) for a fully annotated example.

Section and setting names are matched without regard to case, so `transformer` and
`Transformer` both work. A name the loader does not recognise is an error rather
than a silent default:

```
ValueError: config.yml contains settings this version does not recognise, so they
would have been silently ignored:
  model.common.embeding_dim was never read. Did you mean 'embedding_dim'?
```

`sample/config.yml` ships with `token_file_path` pointing at the bundled
vocabulary, so it loads as-is from the repository root. The data paths under
`training.data` are blank and must be filled in before training; an unset path
is reported by name rather than failing later inside the loader.

## Top-level sections

| Section    | Description                                                                           |
| ---------- | ------------------------------------------------------------------------------------- |
| `system`   | Environment (`normal`, or `hpc` to read data files from `$LOCALDIR`), seed, workers, bf16 |
| `logging`  | WandB settings, progress bars, log interval                                           |
| `model`    | Architecture: VAE flag, embedding dim, Transformer layers/heads, RoPE, latent pooling |
| `training` | Strategy, optimizer, ZClip, checkpointing, validation, evaluation, data               |

## Training strategies

| `strategy.type` | Description                                                                                                                                               |
| --------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `normal`        | Standard single-phase training                                                                                                                            |
| `phased`        | Multi-phase curriculum — each phase adds a new data split; earlier data is replayed at `replay_ratio`                                                     |
| `dynamic`       | Bucket-based curriculum — sampling probabilities per bucket interpolate from `start_sampling_probabilities` to `end_sampling_probabilities` over training |

## Key model parameters (`model` section)

Values below are those of [`sample/config.yml`](../sample/config.yml) (the paper settings). A key
left out of the YAML falls back to the code default, which differs for three of them:
`loss.beta` 0.01, `latent.cross_attn` none, `latent.latent_conditioning` `add_once`.

| Parameter                                | sample value | Description                                                       |
| ---------------------------------------- | ------------ | ----------------------------------------------------------------- |
| `use_vae`                                | `false`      | Enable VAE mode (adds KL loss with `beta` weighting)              |
| `loss.beta`                              | `0.1`        | KL weight, in nats of KL per token of reconstruction              |
| `loss.beta_warmup_steps`                 | `0`          | Steps over which the KL weight rises linearly from 0 to `beta`    |
| `common.embedding_dim`                   | `512`        | Hidden/latent dimension                                           |
| `common.dropout`                         | `0.2`        | Dropout probability                                               |
| `Transformer.n_layer`                    | `8`          | Number of encoder/decoder layers                                  |
| `Transformer.n_head`                     | `8`          | Number of attention heads                                         |
| `Transformer.n_positions`                | `2048`       | Maximum sequence length                                           |
| `Transformer.latent.pooling_mode`        | `MHA`        | Latent pooling: `MHA` or `concat`                                 |
| `Transformer.latent.cross_attn`          | `L-T`        | Cross-attention mode: `L-T`, `S-T`, or blank                      |
| `Transformer.latent.latent_conditioning` | `adaln-zero` | Decoder conditioning: `add_once`, `adaln`, `adaln-zero`, or blank |

## VAE mode

`use_vae: true` adds a KL term against a standard normal prior. Two things are worth
knowing before starting a run.

**`beta` is per token.** The KL is divided by the molecule's token count, so the
objective is the per-molecule ELBO divided by the number of target tokens and `beta`
means the same thing at every sequence length. Without that division the effective
weight would be `beta x L / embedding_dim`, which under the bucketed curriculum
(one optimizer step draws from a single length bucket) swings by an order of
magnitude between consecutive steps.

**Watch for posterior collapse.** A Transformer decoder with teacher forcing can
drive the reconstruction loss down without using the latent at all, after which the
KL term pushes the posterior onto the prior and the latent carries nothing. Two
things help:

- `beta_warmup_steps` raises the KL weight from 0, giving the decoder time to learn
  to use the latent before the prior pressure arrives.
- `latent_conditioning: adaln-zero` initializes its gates to zero, so at step 1 the
  latent has *no* influence on the decoder output at all — measured as an exactly
  zero change in logits under a 10-sigma perturbation of the latent. That is the
  intended behaviour for a deterministic autoencoder, but for a VAE it starts
  training in the collapsed state. `adaln` and `add_once` both leave the latent
  live from the first step.

Evaluation reports `eval_klloss` alongside `eval_celoss`, plus
`eval_kl_active_units` (latent dimensions whose KL exceeds 0.01 nats) and
`eval_kl_per_dim_max`. The summed KL alone cannot distinguish a posterior that uses
every dimension from one where five dimensions carry everything, so watch the active
unit count. The same three are logged per step with a `train_` prefix, together with
`train_kl_beta_scale` (the warm-up position).

`use_vae: true` with `cross_attn: S-T` is rejected at config load: S-T feeds the
decoder the encoder hidden states directly, so no latent is pooled.
