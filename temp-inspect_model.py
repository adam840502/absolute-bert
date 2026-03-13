import logging
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import wandb
from transformers import AutoTokenizer
from umap import UMAP

from absolute_bert.base_types import LanguageModel
# from absolute_bert.model.absolute_bert.models import AbsoluteBertLM as LM
from absolute_bert.model.roformer.models import RoformerLM as LM
from absolute_bert.sweep import setup
from absolute_bert.utils import init_logging

init_logging()
logger = logging.getLogger(__name__)


config_unresolved = setup.get_config(config_file="configs/default.yaml")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
config = config_unresolved.resolve(tokenizer.vocab_size)

wandb.init()

artifact = wandb.use_artifact("ghuang-nlp/absolute-bert/model:v53")
artifact_dir = artifact.download()
artifact_path = Path(artifact_dir)

states = torch.load(artifact_path / "model.pt", weights_only=True, map_location="cpu")
model: LanguageModel = LM(config.model)
model.load_state_dict(states)


# metric = "euclidean"
metric = "cosine"

logger.info("start umap learning")
learner = UMAP(metric=metric)
transed = learner.fit_transform(model.word_embeddings.data.detach().numpy())

# import numpy as np

# np.save("transed.npy", transed)

# transed = np.load("transed.npy")

# plt.figure()
# plt.scatter(*transed.T, s=1)
# plt.savefig("plt.png")


import numpy as np
import plotly.graph_objects as go
from jaxtyping import Float


special_token_list = tokenizer.all_special_tokens
special_token_ids = tokenizer.convert_tokens_to_ids(special_token_list)

ordered_token_list = tokenizer.convert_ids_to_tokens(torch.arange(tokenizer.vocab_size))


def plotly_2d(
    transed: Float[np.ndarray, "N2"],
    norms: Float[np.ndarray, "N"] | None = None,
    opacity: Float[np.ndarray, ""] | None = None,
) -> go.Figure:

    fig = go.Figure(layout=dict(width=1200, height=1200))
    # fig.update_layout(
    #     autosize=False,
    #     width=500,
    #     height=500,
    # )

    trace = go.Scatter(
        mode="markers",
        x=transed[0],
        y=transed[1],
        marker=dict(
            color=norms,
            size=3,
            opacity=opacity,
        ),
        # custom_data=b,
        # hovertemplate=f""
        # hovertext=[f"{name}: {bias}" for name, bias in zip(ordered_token_list, b)],
        hovertext=ordered_token_list,
        # showlegend=False
    )
    fig.add_trace(trace)

    trace = go.Scatter(
        mode="markers",
        x=transed[0, special_token_ids],
        y=transed[1, special_token_ids],
        marker=dict(color="black", symbol="diamond"),
        hovertext=special_token_list,
    )
    fig.add_trace(trace)

    return fig


fig = plotly_2d(transed.T)
fig.write_html(f"v53-{metric}.html")
