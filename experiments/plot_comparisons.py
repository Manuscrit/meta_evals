import logging
from pathlib import Path
from typing import Any, Sequence, cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import scipy.cluster.hierarchy
import scipy.spatial.distance
from dotenv import load_dotenv
from mppr import MContext
from tqdm import tqdm

from meta_evals.activations.probe_preparations import ActivationArrayDataset
from meta_evals.datasets.elk.types import DatasetId, Split
from meta_evals.datasets.elk.utils.collections import (
    DatasetCollectionId,
    resolve_dataset_ids,
)
from meta_evals.datasets.elk.utils.filters import DatasetFilter, DatasetIdFilter
from meta_evals.evals_utils.probes import (
    eval_probe_by_question,
    eval_probe_by_row,
)
from meta_evals.evaluations.predict_from_next_token import BaseEval
from meta_evals.models.llms import LlmId
from meta_evals.models.points import get_points
from meta_evals.evaluations.probes.base import BaseProbe
from meta_evals.evaluations.probes.collections import (
    GROUPED_PROBES,
    SUPERVISED_PROBES,
    EvalMethod,
    train_probe,
    ALL_BEHAVIORAL_PROBES,
    get_evals_to_work_with,
    train_evaluation,
)
from meta_evals.utils.compare_strucs import (
    TrainSpec,
    EvalSpec,
    EvalResult,
    PipelineResultRow,
)
from meta_evals.utils.constants import (
    ACTIVATION_DIR,
    DEBUG,
    PLOT_DIR,
    get_layer_filter,
    get_model_ids,
    get_ds_collection_ids,
    inital_layer,
    WORK_WITH_BEHAVIORAL_PROBES,
    DEBUG_VERSION,
    layer_skip,
)
from meta_evals.utils.utils import get_dataset_ids


def compute_and_plot_all_comparisons(activation_ds_paths):
    assert load_dotenv("../.env")
    token_idxs: list[int] = [-1]

    if len(LLM_IDS) != 1:
        raise ValueError("Only one LLM_ID is supported")
    points = get_points(LLM_IDS[0])[inital_layer() :: layer_skip()]
    dataset_ids = [DatasetIdFilter(dataset) for dataset in DATASETS_TO_EVAL]
    to_rm = DatasetIdFilter("truthful_qa")
    if to_rm in dataset_ids:
        dataset_ids.remove(to_rm)

    (
        dataset,
        mcontext,
        output_path,
    ) = load_activation_dataset(activation_ds_paths)

    output_dir = output_path / STAGE_VERSION
    output_dir.mkdir(parents=True, exist_ok=True)

    assert len(dataset_ids) > 0

    df = compute_metrics(
        dataset_ids,
        dataset,
        mcontext,
        points,
        token_idxs,
    )
    df = get_ranking_by_generalization_performance(df)
    ds_order, algorithm_order = get_ds_and_algo_order(df)
    plot_cumulative_distribution_function(df, output_path, "recovered_accuracy")
    plot_cumulative_distribution_function(df, output_path, "accuracy")
    compute_percent_above_80(df)
    examine_best_probe(df, ds_order, algorithm_order, output_path)
    plot_algo_generalization_meta_eval(df, output_path, algorithm_order)
    plot_dataset_generalization_meta_eval(df, output_path, ds_order)
    plot_algo_bias_meta_eval(df, output_path, algorithm_order, ds_order)
    plot_dataset_bias_meta_eval(df, output_path, ds_order)

    plot_truthful_qa_meta_eval(
        dataset,
        mcontext,
        output_path,
        points,
        token_idxs,
        dataset_ids,
        df,
    )

    # plot_persona_meta_eval(
    #     dataset,
    #     mcontext,
    #     output_path,
    #     points,
    #     token_idxs,
    #     dataset_ids,
    #     df,
    # )
    if not DEBUG:
        sanity_check_results(
            points, token_idxs, output_path, dataset, mcontext, df
        )


def load_activation_dataset(activation_ds_paths: list[str] = None):
    output_path = Path(PLOT_DIR / "comparison")
    mcontext = MContext(output_path)
    activation_results = None
    for ds_idx, path in enumerate(activation_ds_paths):
        activation_result = mcontext.download_cached(
            f"{STAGE_VERSION}_activations_results_idx_{ds_idx}",
            path=path,
            to="pickle",
        ).get()
        if activation_results is None:
            activation_results = activation_result
        else:
            activation_results += activation_result
    print(set(row.llm_id for row in activation_results))
    print(set(row.dataset_id for row in activation_results))
    print(set(row.split for row in activation_results))
    dataset = ActivationArrayDataset(activation_results)
    return dataset, mcontext, output_path


def run_pipeline(
    dataset: ActivationArrayDataset,
    mcontext: MContext,
    points,
    token_idxs,
    llm_ids: list[LlmId],
    train_datasets: Sequence[DatasetFilter],
    eval_datasets: Sequence[DatasetFilter],
    probe_methods: list[EvalMethod],
    eval_splits: list[Split],
) -> list[PipelineResultRow]:
    train_specs = mcontext.create(
        {
            "-".join(
                [
                    llm_id,
                    str(train_dataset),
                    probe_method,
                    point.name,
                    str(token_idx),
                ]
            ): TrainSpec(
                llm_id=llm_id,
                dataset=train_dataset,
                probe_method=probe_method,
                point_name=point.name,
                token_idx=token_idx,
            )
            for llm_id in llm_ids
            for train_dataset in train_datasets
            for probe_method in probe_methods
            for point in points
            for token_idx in token_idxs
        }
    )
    probes = train_specs.map_cached(
        f"{STAGE_VERSION}_probe_train",
        lambda _, spec: train_evaluation(
            spec.probe_method,
            dataset.get(
                llm_id=spec.llm_id,
                dataset_filter=spec.dataset,
                split="train",
                point_name=spec.point_name,
                token_idx=spec.token_idx,
                limit=None,
            ),
        ),
        to="pickle",
    ).filter(lambda _, probe: probe is not None)
    return (
        probes.join(
            train_specs,
            lambda _, probe, spec: (probe, spec),
        )
        .flat_map(
            lambda key, probe_and_spec: {
                f"{key}-{eval_dataset}": EvalSpec(
                    train_spec=probe_and_spec[1],
                    probe=cast(BaseProbe, probe_and_spec[0]),
                    dataset=eval_dataset,
                )
                for eval_dataset in eval_datasets
            }
        )
        .map_cached(
            f"{STAGE_VERSION}_probe_evaluate",
            lambda _, spec: _evaluate_probe(dataset, spec, eval_splits),
            to=PipelineResultRow,
        )
        .get()
    )


def _evaluate_probe(
    dataset: ActivationArrayDataset, spec: EvalSpec, splits: list[Split]
) -> PipelineResultRow:
    result_hparams = None
    result_validation = None
    if "train-hparams" in splits:
        result_hparams = _evaluate_probe_on_split(
            dataset, spec.probe, spec.train_spec, spec.dataset, "train-hparams"
        )
    if "validation" in splits:
        result_validation = _evaluate_probe_on_split(
            dataset, spec.probe, spec.train_spec, spec.dataset, "validation"
        )
    assert "train" not in splits, splits
    return PipelineResultRow(
        llm_id=spec.train_spec.llm_id,
        train_dataset=spec.train_spec.dataset.get_name(),
        eval_dataset=spec.dataset.get_name(),
        probe_method=spec.train_spec.probe_method,
        point_name=spec.train_spec.point_name,
        token_idx=spec.train_spec.token_idx,
        accuracy=result_validation.accuracy if result_validation else 0,
        accuracy_n=result_validation.n if result_validation else 0,
        accuracy_hparams=result_hparams.accuracy if result_hparams else 0,
        accuracy_hparams_n=result_hparams.n if result_hparams else 0,
    )


def _evaluate_probe_on_split(
    dataset: ActivationArrayDataset,
    evaluation: BaseProbe | BaseEval,
    train_spec: TrainSpec,
    eval_dataset: DatasetFilter,
    split: Split,
) -> EvalResult:
    arrays = dataset.get(
        llm_id=train_spec.llm_id,
        dataset_filter=eval_dataset,
        split=split,
        point_name=train_spec.point_name,
        token_idx=train_spec.token_idx,
        limit=None,
    )
    if arrays.groups is not None:
        question_result = eval_probe_by_question(
            evaluation,
            activations=arrays.activations,
            labels=arrays.labels,
            groups=arrays.groups,
        )
        return EvalResult(
            accuracy=question_result.accuracy, n=question_result.n
        )
    else:
        row_result = eval_probe_by_row(
            evaluation, activations=arrays.activations, labels=arrays.labels
        )
        return EvalResult(accuracy=row_result.accuracy, n=row_result.n)


def to_dataframe(results: Sequence[PipelineResultRow]) -> pd.DataFrame:
    df = pd.DataFrame([row.model_dump() for row in results])
    df = df.rename(
        columns={
            "train_dataset": "train",
            "eval_dataset": "eval",
            "probe_method": "algorithm",
        }
    )
    df["eval"] = df["eval"].replace(
        {"dlk-val": "dlk", "repe-qa-val": "repe", "got-val": "got"}
    )
    df["supervised"] = np.where(
        df["algorithm"].isin(SUPERVISED_PROBES), "sup", "unsup"
    )
    df["grouped"] = np.where(
        df["algorithm"].isin(GROUPED_PROBES), "grouped", "ungrouped"
    )
    df["layer"] = df["point_name"].apply(lambda p: int(p.lstrip("h")))
    df["train_group"] = df["train"].apply(
        lambda d: (
            "dlk"
            if d in DLK_DATASETS
            else "repe"
            if d in REPE_DATASETS
            else "got"
        )
    )
    df["algorithm"] = df["algorithm"].str.upper()
    df = df.drop(columns=["point_name", "token_idx"])
    return df


def select_best(
    df: pd.DataFrame,
    column: str,
    metric: str,
    extra_dims: list[str] | None = None,
) -> pd.DataFrame:
    extra_dims = extra_dims or []
    return (
        df.sort_values(metric, ascending=False)
        .groupby(list(DIMS - {column} | set(extra_dims)))
        .first()
        .reset_index()
    )


def compute_metrics(
    dataset_ids: list[DatasetId],
    dataset: ActivationArrayDataset,
    mcontext: MContext,
    points: list[str],
    token_idxs: list[int],
):

    results = run_pipeline(
        dataset=dataset,
        mcontext=mcontext,
        points=points,
        token_idxs=token_idxs,
        llm_ids=LLM_IDS,
        train_datasets=dataset_ids,
        eval_datasets=dataset_ids,
        probe_methods=get_evals_to_work_with(),
        eval_splits=["validation", "train-hparams"],
    )

    # %%
    df = to_dataframe(results)
    thresholds_idx = (
        df.query("train == eval").groupby("eval")["accuracy_hparams"].idxmax()
    )
    thresholds = (
        df.loc[thresholds_idx][["eval", "accuracy", "accuracy_n"]]
        .rename(
            columns={
                "accuracy": "threshold",
                "accuracy_n": "threshold_n",
            }
        )
        .set_index("eval")
    )
    df = df.join(thresholds, on="eval")
    df["recovered_accuracy"] = (
        df["accuracy"].to_numpy() / df["threshold"].to_numpy()
    ).clip(0, 1)
    df["recovered_accuracy_hparams"] = (
        df["accuracy_hparams"].to_numpy() / df["threshold"].to_numpy()
    ).clip(0, 1)
    return df


def get_ds_and_algo_order(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """
    Fix the order of train datasets & algorithms in plots.
    """
    train_order = (
        df.groupby(["train"])["recovered_accuracy"]
        .mean()
        .sort_values(ascending=False)  # type: ignore
        .index.to_list()
    )
    algorithm_order = (
        df.groupby(["algorithm"])["recovered_accuracy"]
        .mean()
        .sort_values(ascending=False)  # type: ignore
        .index.to_list()
    )
    return train_order, algorithm_order


def get_ranking_by_generalization_performance(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rank probes by generalization performance.

    Use a pair-wise comparison score, where a probe gets a point if it generalizes better
    on all datasets that *neither* probe has been trained on.
    """
    probes = (
        df.sort_values("eval")
        .groupby(["train", "algorithm", "layer"])[
            ["eval", "recovered_accuracy_hparams"]
        ]
        .agg(list)
        .reset_index()
    )
    results = []
    probe1: Any
    probe2: Any
    for probe1 in tqdm(probes.itertuples()):
        for probe2 in probes.itertuples():
            probe1_recovered_wins = 0
            probe2_recovered_wins = 0
            probe1_perfect_wins = 0
            probe2_perfect_wins = 0
            for eval1, generalizes1, eval2, generalizes2 in zip(
                probe1.eval,
                probe1.recovered_accuracy_hparams,
                probe2.eval,
                probe2.recovered_accuracy_hparams,
            ):
                assert eval1 == eval2, (eval1, eval2)
                if eval1 == probe1.train or eval1 == probe2.train:
                    continue
                if generalizes1 > generalizes2:
                    probe1_recovered_wins += 1
                if generalizes1 < generalizes2:
                    probe2_recovered_wins += 1
                if generalizes1 == 1 and generalizes2 < 1:
                    probe1_perfect_wins += 1
                if generalizes1 < 1 and generalizes2 == 1:
                    probe2_perfect_wins += 1
            results.append(
                dict(
                    train=probe1.train,
                    algorithm=probe1.algorithm,
                    layer=probe1.layer,
                    recovered_score=probe1_recovered_wins
                    > probe2_recovered_wins,
                    perfect_score=probe1_perfect_wins > probe2_perfect_wins,
                )
            )

    # %%
    (
        df.groupby(["train", "algorithm", "layer"])[
            ["recovered_accuracy", "recovered_accuracy_hparams"]
        ]
        .mean()
        .reset_index()
        .join(
            pd.DataFrame(results)
            .groupby(["train", "algorithm", "layer"])[["recovered_score"]]
            .mean(),
            on=["train", "algorithm", "layer"],
        )
        .sort_values("recovered_score", ascending=False)
        .head(20)
    )
    return df


def plot_cumulative_distribution_function(
    df: pd.DataFrame, output_path: str | Path, x: str = "recovered_accuracy"
):
    # %%
    fig = px.ecdf(
        df.groupby(["algorithm", "train", "layer", "supervised"])[x]
        .mean()
        .reset_index(),
        x=x,
        color="layer",
        width=800,
        height=400,
        color_discrete_sequence=px.colors.sequential.deep,
    )
    fig.update_layout(xaxis_tickformat=".0%", yaxis_tickformat=".0%")
    fig.write_image(
        output_path / f"{STAGE_VERSION}/r0_{x}_by_layer.png", scale=3
    )
    plt.close()


def compute_percent_above_80(df: pd.DataFrame):
    perc_above_80 = np.mean(
        df.query(get_layer_filter())
        .groupby(["algorithm", "train", "layer", "supervised"])[
            "recovered_accuracy"
        ]
        .mean()
        > 0.8
    )
    print(
        f"Percent of layer 13+ probes with >80% recovered accuracy: {perc_above_80:.1%}"
    )


def examine_best_probe(df, train_order, algorithm_order, output_path):
    best_train = "dbpedia_14"
    best_algorithm = "DIM"
    best_layer = 5 if DEBUG else 21
    examine_probe(
        df,
        train=best_train,
        algo=best_algorithm,
        layer=best_layer,
        train_order=train_order,
        algorithm_order=algorithm_order,
        output_path=output_path,
    )


def examine_probe(
    df, train, algo, layer, train_order, algorithm_order, output_path
):
    df_best_probes = (
        df.copy()
        .query("train != eval")
        .query(f"train == '{train}'")
        .query(f"layer == {layer}")
        .groupby(["algorithm", "eval"])["recovered_accuracy"]
        .mean()
        .reset_index()
    )
    if len(df_best_probes) == 0:
        logging.warning(f"No probes for {train} at layer {layer}")
        return
    fig = px.imshow(
        df_best_probes.pivot(
            index="algorithm", columns="eval", values="recovered_accuracy"
        )
        .reindex(algorithm_order, axis=0)
        .reindex([d for d in train_order if d != train], axis=1)
        .rename(index={algo: f"<b>{algo} (best)</b>"}),
        color_continuous_scale=COLORS,
        range_color=[0, 1],
        text_auto=".0%",  # type: ignore
        width=800,
        height=500,
    )
    fig.update_layout(coloraxis_showscale=False)
    fig.write_image(
        output_path / f"{STAGE_VERSION}/r1a_by_algorithms.png", scale=3
    )
    plt.close()

    df_best_train = (
        df.copy()
        .query(f"algorithm == '{algo}'")
        .query(f"layer == {layer}")
        .groupby(["train", "eval"])["recovered_accuracy"]
        .mean()
        .reset_index()
    )
    fig = px.imshow(
        df_best_train.pivot(
            index="train", columns="eval", values="recovered_accuracy"
        )
        .reindex(train_order, axis=0)
        .reindex([d for d in train_order if d != train], axis=1)
        .rename(index={train: f"<b>{train} (best)</b>"})
        .dropna(),
        color_continuous_scale=COLORS,
        range_color=[0, 1],
        text_auto=".0%",  # type: ignore
        width=800,
        height=800,
    )
    fig.update_layout(coloraxis_showscale=False)
    fig.write_image(output_path / f"{STAGE_VERSION}/r1b_by_train.png", scale=3)
    plt.close()

    df_best_layer = (
        df.copy()
        .query(f"train == '{train}'")
        .query(f"algorithm == '{algo}'")
        .groupby(["layer", "eval"])["recovered_accuracy"]
        .mean()
        .reset_index()
    )
    df_best_layer = df_best_layer.pivot(
        index="layer", columns="eval", values="recovered_accuracy"
    )
    df_best_layer.index = df_best_layer.index.map(str)
    fig = px.imshow(
        df_best_layer.reindex(
            [d for d in train_order if d != train], axis=1
        ).rename(index={str(layer): f"<b>{layer} (best)</b>"}),
        color_continuous_scale=COLORS,
        range_color=[0, 1],
        text_auto=".0%",  # type: ignore
        width=800,
        height=600,
    )
    fig.update_layout(coloraxis_showscale=False)
    fig.write_image(output_path / f"{STAGE_VERSION}/r1c_by_layer.png", scale=3)
    plt.close()


def plot_algo_generalization_meta_eval(df, output_path, algorithm_order):
    """
    2. Examining algorithm performance
    """
    id_vars = ["algorithm", "supervised", "grouped"]
    fig = px.bar(
        df.copy().groupby(id_vars)["recovered_accuracy"].mean().reset_index(),
        x="algorithm",
        y="recovered_accuracy",
        color="supervised",
        pattern_shape="grouped",
        category_orders={"algorithm": algorithm_order},
        width=800,
        height=400,
        text_auto=".0%",  # type: ignore
    )
    fig.update_layout(yaxis_tickformat=".0%")
    fig.write_image(
        output_path / f"{STAGE_VERSION}/algo_generalization.png", scale=3
    )
    plt.close()

    # %%


def plot_algo_bias_meta_eval(df, output_path, algorithm_order, ds_order):
    """
    2. Examining algorithm performance
    """
    id_vars = ["algorithm", "supervised", "grouped"]
    fig = px.bar(
        df.copy().groupby(id_vars)["accuracy"].mean().reset_index(),
        x="algorithm",
        y="accuracy",
        color="supervised",
        pattern_shape="grouped",
        category_orders={"algorithm": algorithm_order},
        width=800,
        height=400,
        text_auto=".0%",  # type: ignore
    )
    fig.update_layout(yaxis_tickformat=".0%")
    fig.write_image(output_path / f"{STAGE_VERSION}/algo_bias_all.png", scale=3)
    plt.close()

    id_vars = ["algorithm", "supervised", "grouped"]
    fig = px.bar(
        df.copy()
        .query("train == eval")
        .groupby(id_vars)["accuracy"]
        .mean()
        .reset_index(),
        x="algorithm",
        y="accuracy",
        color="supervised",
        pattern_shape="grouped",
        category_orders={"algorithm": algorithm_order},
        width=800,
        height=400,
        text_auto=".0%",  # type: ignore
    )
    fig.update_layout(yaxis_tickformat=".0%")
    fig.write_image(
        output_path / f"{STAGE_VERSION}/algo_bias_train.png", scale=3
    )
    plt.close()

    id_vars = ["algorithm", "supervised", "grouped"]
    fig = px.bar(
        df.copy()
        .query("train != eval")
        .groupby(id_vars)["accuracy"]
        .mean()
        .reset_index(),
        x="algorithm",
        y="accuracy",
        color="supervised",
        pattern_shape="grouped",
        category_orders={"algorithm": algorithm_order},
        width=800,
        height=400,
        text_auto=".0%",  # type: ignore
    )
    fig.update_layout(yaxis_tickformat=".0%")
    fig.write_image(
        output_path / f"{STAGE_VERSION}/algo_bias_test.png", scale=3
    )
    plt.close()

    df_train = df.copy()
    df_train = (
        df_train.query(get_layer_filter())
        .groupby(["train", "algorithm"])["accuracy"]  # type: ignore
        .mean()
        .reset_index()
    )
    fig = px.imshow(
        df_train.pivot(index="train", columns="algorithm", values="accuracy")
        .reindex(ds_order, axis=0)
        .reindex(algorithm_order, axis=1),
        text_auto=".0%",  # type: ignore
        color_continuous_scale=COLORS,
        width=800,
        height=800,
    )
    fig.update_layout(coloraxis_showscale=False)
    fig.write_image(
        output_path / f"{STAGE_VERSION}/algo_generalization_matrix_acc.png",
        scale=3,
    )
    plt.close()


def plot_dataset_generalization_meta_eval(df, output_path, ds_order):
    """
    2. Examining dataset performance
    """
    id_vars = ["train", "train_group"]
    fig = px.bar(
        df.copy().groupby(id_vars)["recovered_accuracy"].mean().reset_index(),
        x="train",
        y="recovered_accuracy",
        color="train_group",
        category_orders={"train": ds_order},
        width=800,
        height=400,
        text_auto=".0%",  # type: ignore
    )
    fig.update_layout(yaxis_tickformat=".0%")
    fig.write_image(
        output_path / f"{STAGE_VERSION}/datasets_generalization.png", scale=3
    )
    plt.close()

    # %%
    """
    2. Examining dataset performance: matrix
    """
    df_train = df.copy()
    df_train = (
        df_train.query(get_layer_filter())
        .groupby(["train", "eval"])["recovered_accuracy"]  # type: ignore
        .mean()
        .reset_index()
    )
    fig = px.imshow(
        df_train.pivot(
            index="train", columns="eval", values="recovered_accuracy"
        )
        .reindex(ds_order, axis=0)
        .reindex(ds_order, axis=1),
        text_auto=".0%",  # type: ignore
        color_continuous_scale=COLORS,
        width=800,
        height=800,
    )
    fig.update_layout(coloraxis_showscale=False)
    fig.write_image(
        output_path / f"{STAGE_VERSION}/datasets_generalization_matrix.png",
        scale=3,
    )
    plt.close()

    df_train = df.copy()
    df_train = (
        df_train.query(get_layer_filter())
        .groupby(["train", "eval"])["accuracy"]  # type: ignore
        .mean()
        .reset_index()
    )
    fig = px.imshow(
        df_train.pivot(index="train", columns="eval", values="accuracy")
        .reindex(ds_order, axis=0)
        .reindex(ds_order, axis=1),
        text_auto=".0%",  # type: ignore
        color_continuous_scale=COLORS,
        width=800,
        height=800,
    )
    fig.update_layout(coloraxis_showscale=False)
    fig.write_image(
        output_path / f"{STAGE_VERSION}/datasets_generalization_matrix_acc.png",
        scale=3,
    )
    plt.close()

    # %%
    """
    2. Examining dataset performance: from and to
    """
    df_sym = (
        pd.concat(
            [
                df.groupby("train")["recovered_accuracy"]
                .mean()
                .rename("generalizes_from"),
                df.groupby("eval")["recovered_accuracy"]
                .mean()
                .rename("generalizes_to"),
            ],
            axis=1,
        )
        .reset_index()
        .rename({"index": "dataset"}, axis=1)
    )
    df_sym["group"] = df_sym["dataset"].apply(
        lambda d: (
            "dlk"
            if d in DLK_DATASETS
            else "repe"
            if d in REPE_DATASETS
            else "got"
        )
    )
    fig = px.scatter(
        df_sym,
        x="generalizes_from",
        y="generalizes_to",
        color="group",
        text="dataset",
        range_x=[0.66, 0.76],
        height=600,
        width=800,
    )
    fig.update_traces(textposition="bottom center")
    fig.write_image(
        output_path
        / f"{STAGE_VERSION}/datasets_generalization_to_and_from.png",
        scale=3,
    )
    plt.close()

    # %%
    # """
    # 2. Examining dataset performance: clustering
    # """
    # df_train = df.copy()
    # df_train = (
    #     (
    #         df_train.query(get_layer_filter())
    #         .groupby(["train", "eval"])["recovered_accuracy"]  # type: ignore
    #         .mean()
    #         .reset_index()
    #     )
    #     .pivot(index="train", columns="eval", values="recovered_accuracy")
    #     .reindex(ds_order, axis=0)
    #     .reindex(ds_order, axis=1)
    # )
    # adjacency_matrix = 1 - df_train.to_numpy()
    # adjacency_matrix = (adjacency_matrix.T + adjacency_matrix) / 2
    # condensed_dist_matrix = scipy.spatial.distance.squareform(
    #     adjacency_matrix, checks=False
    # )
    # linkage = scipy.cluster.hierarchy.linkage(condensed_dist_matrix, "average")
    #
    # fig, ax = plt.subplots(figsize=(10, 7))
    # scipy.cluster.hierarchy.dendrogram(linkage, ax=ax)
    # ax.set_ylabel("1 - recovered accuracy")
    # ax.set_xticklabels(
    #     [
    #         df_train.index[int(label.get_text())]
    #         for label in ax.get_xticklabels()
    #     ],
    #     rotation=90,
    # )
    # fig.savefig(
    #     output_path / f"{STAGE_VERSION}/datasets_generalization_clustering.png",
    #     dpi=300,
    #     bbox_inches="tight",
    # )
    # fig.tight_layout()
    # plt.close()


def plot_dataset_bias_meta_eval(df, output_path, ds_order):
    """
    2. Examining dataset performance
    """
    id_vars = ["train", "train_group"]
    fig = px.bar(
        df.copy().groupby(id_vars)["accuracy"].mean().reset_index(),
        x="train",
        y="accuracy",
        color="train_group",
        category_orders={"train": ds_order},
        width=800,
        height=400,
        text_auto=".0%",  # type: ignore
    )
    fig.update_layout(yaxis_tickformat=".0%")
    fig.write_image(output_path / f"{STAGE_VERSION}/datasets_bias.png", scale=3)
    plt.close()

    # %%
    """
    2. Examining dataset performance: matrix
    """
    df_train = df.copy()
    df_train = (
        df_train.query(get_layer_filter())
        .groupby(["train", "eval"])["accuracy"]  # type: ignore
        .mean()
        .reset_index()
    )
    fig = px.imshow(
        df_train.pivot(index="train", columns="eval", values="accuracy")
        .reindex(ds_order, axis=0)
        .reindex(ds_order, axis=1),
        text_auto=".0%",  # type: ignore
        color_continuous_scale=COLORS,
        width=800,
        height=800,
    )
    fig.update_layout(coloraxis_showscale=False)
    fig.write_image(
        output_path / f"{STAGE_VERSION}/datasets_bias_matrix.png", scale=3
    )
    plt.close()

    # %%
    """
    2. Examining dataset performance: from and to
    """
    df_sym = (
        pd.concat(
            [
                df.groupby("train")["accuracy"]
                .mean()
                .rename("generalizes_from"),
                df.groupby("eval")["accuracy"].mean().rename("generalizes_to"),
            ],
            axis=1,
        )
        .reset_index()
        .rename({"index": "dataset"}, axis=1)
    )
    df_sym["group"] = df_sym["dataset"].apply(
        lambda d: (
            "dlk"
            if d in DLK_DATASETS
            else "repe"
            if d in REPE_DATASETS
            else "got"
        )
    )
    fig = px.scatter(
        df_sym,
        x="generalizes_from",
        y="generalizes_to",
        color="group",
        text="dataset",
        range_x=[0.66, 0.76],
        height=600,
        width=800,
    )
    fig.update_traces(textposition="bottom center")
    fig.write_image(
        output_path / f"{STAGE_VERSION}/datasets_bias_to_and_from.png", scale=3
    )
    plt.close()

    # %%
    # """
    # 2. Examining dataset performance: clustering
    # """
    # df_train = df.copy()
    # df_train = (
    #     (
    #         df_train.query(get_layer_filter())
    #         .groupby(["train", "eval"])["accuracy"]  # type: ignore
    #         .mean()
    #         .reset_index()
    #     )
    #     .pivot(index="train", columns="eval", values="accuracy")
    #     .reindex(ds_order, axis=0)
    #     .reindex(ds_order, axis=1)
    # )
    # adjacency_matrix = 1 - df_train.to_numpy()
    # adjacency_matrix = (adjacency_matrix.T + adjacency_matrix) / 2
    # condensed_dist_matrix = scipy.spatial.distance.squareform(
    #     adjacency_matrix, checks=False
    # )
    # linkage = scipy.cluster.hierarchy.linkage(condensed_dist_matrix, "average")
    #
    # fig, ax = plt.subplots(figsize=(10, 7))
    # scipy.cluster.hierarchy.dendrogram(linkage, ax=ax)
    # ax.set_ylabel("1 - accuracy")
    # ax.set_xticklabels(
    #     [
    #         df_train.index[int(label.get_text())]
    #         for label in ax.get_xticklabels()
    #     ],
    #     rotation=90,
    # )
    # fig.savefig(
    #     output_path / f"{STAGE_VERSION}/datasets_bias_clustering.png",
    #     dpi=300,
    #     bbox_inches="tight",
    # )
    # fig.tight_layout()
    # plt.close()


def plot_truthful_qa_generalization_meta_eval(
    dataset,
    mcontext,
    output_path,
    points,
    token_idxs,
    datasets,
    df,
):
    """
    4. TruthfulQA
    """
    results_truthful_qa = run_pipeline(
        dataset=dataset,
        mcontext=mcontext,
        points=points,
        token_idxs=token_idxs,
        llm_ids=LLM_IDS,
        train_datasets=datasets,
        eval_datasets=[DatasetIdFilter("truthful_qa")],
        probe_methods=get_evals_to_work_with(),
        eval_splits=["validation"],
    )
    truthful_qa = (
        to_dataframe(results_truthful_qa)
        .set_index(["train", "algorithm", "layer"])["accuracy"]
        .rename("truthful_qa")  # type: ignore
    )
    df_truthful_qa = (
        df.query("train != eval")
        .groupby(["train", "algorithm", "layer"])["recovered_accuracy"]
        .mean()
        .reset_index()
        .join(
            truthful_qa,
            on=["train", "algorithm", "layer"],
        )
    )
    fig = px.scatter(
        df_truthful_qa,
        x="recovered_accuracy",
        y="truthful_qa",
        width=800,
    )
    fig.update_layout(xaxis_tickformat=".0%", yaxis_tickformat=".0%")
    # https://arxiv.org/abs/2310.01405, table 1
    fig.add_hline(y=0.359, line_dash="dot", line_color="green")
    fig.add_hline(y=0.503, line_dash="dot", line_color="gray")
    fig.write_image(
        output_path / f"{STAGE_VERSION}/truthful_qa_generalization.png", scale=3
    )
    plt.close()

    perc_measuring_truth = (
        (df_truthful_qa["recovered_accuracy"] > 0.8)
        & (df_truthful_qa["truthful_qa"] > 0.359)
    ).sum() / (df_truthful_qa["recovered_accuracy"] > 0.8).sum()
    print(
        f"Percent of probes with >80% recovered accuracy: {perc_measuring_truth:.1%}"
    )


def plot_persona_meta_eval(
    dataset,
    mcontext,
    output_path,
    points,
    token_idxs,
    datasets,
    df,
):
    """
    4. TruthfulQA
    """
    results_truthful_qa = run_pipeline(
        dataset=dataset,
        mcontext=mcontext,
        points=points,
        token_idxs=token_idxs,
        llm_ids=LLM_IDS,
        train_datasets=datasets,
        eval_datasets=[DatasetIdFilter("truthful_qa")],
        probe_methods=get_evals_to_work_with(),
        eval_splits=["validation"],
    )
    truthful_qa = (
        to_dataframe(results_truthful_qa)
        .set_index(["train", "algorithm", "layer"])["accuracy"]
        .rename("truthful_qa")  # type: ignore
    )
    df_truthful_qa = (
        df.query("train != eval")
        .groupby(["train", "algorithm", "layer"])["accuracy"]
        .mean()
        .reset_index()
        .join(
            truthful_qa,
            on=["train", "algorithm", "layer"],
        )
    )
    fig = px.scatter(
        df_truthful_qa,
        x="accuracy",
        y="truthful_qa",
        width=800,
    )
    fig.update_layout(xaxis_tickformat=".0%", yaxis_tickformat=".0%")
    # https://arxiv.org/abs/2310.01405, table 1
    fig.add_hline(y=0.359, line_dash="dot", line_color="green")
    fig.add_hline(y=0.503, line_dash="dot", line_color="gray")
    fig.write_image(
        output_path / f"{STAGE_VERSION}/truthful_qa_bias.png", scale=3
    )
    plt.close()

    perc_measuring_truth = (
        (df_truthful_qa["accuracy"] > 0.8)
        & (df_truthful_qa["truthful_qa"] > 0.359)
    ).sum() / (df_truthful_qa["accuracy"] > 0.8).sum()
    print(
        f"Percent of probes with >80% accuracy after filtering: {perc_measuring_truth:.1%}"
    )


def plot_truthful_qa_meta_eval(
    dataset,
    mcontext,
    output_path,
    points,
    token_idxs,
    datasets,
    df,
):
    """
    4. TruthfulQA
    """
    results_truthful_qa = run_pipeline(
        dataset=dataset,
        mcontext=mcontext,
        points=points,
        token_idxs=token_idxs,
        llm_ids=LLM_IDS,
        train_datasets=datasets,
        eval_datasets=[DatasetIdFilter("truthful_qa")],
        probe_methods=get_evals_to_work_with(),
        eval_splits=["validation"],
    )
    truthful_qa = (
        to_dataframe(results_truthful_qa)
        .set_index(["train", "algorithm", "layer"])["accuracy"]
        .rename("truthful_qa")  # type: ignore
    )
    df_truthful_qa = (
        df.query("train != eval")
        .groupby(["train", "algorithm", "layer"])["accuracy"]
        .mean()
        .reset_index()
        .join(
            truthful_qa,
            on=["train", "algorithm", "layer"],
        )
    )
    fig = px.scatter(
        df_truthful_qa,
        x="accuracy",
        y="truthful_qa",
        width=800,
    )
    fig.update_layout(xaxis_tickformat=".0%", yaxis_tickformat=".0%")
    # https://arxiv.org/abs/2310.01405, table 1
    fig.add_hline(y=0.359, line_dash="dot", line_color="green")
    fig.add_hline(y=0.503, line_dash="dot", line_color="gray")
    fig.write_image(
        output_path / f"{STAGE_VERSION}/truthful_qa_bias.png", scale=3
    )
    plt.close()

    perc_measuring_truth = (
        (df_truthful_qa["accuracy"] > 0.8)
        & (df_truthful_qa["truthful_qa"] > 0.359)
    ).sum() / (df_truthful_qa["accuracy"] > 0.8).sum()
    print(
        f"Percent of probes with >80% accuracy after filtering: {perc_measuring_truth:.1%}"
    )


def sanity_check_results(
    points, token_idxs, output_path, dataset, mcontext, df
):
    if "arc_easy" in df["train"].unique():
        """
        Validating results 1: Simple test. Do we get 80% accuracy on arc_easy?
        """
        results_arc_easy = run_pipeline(
            dataset=dataset,
            mcontext=mcontext,
            points=points,
            token_idxs=token_idxs,
            llm_ids=LLM_IDS,
            train_datasets=[DatasetIdFilter("arc_easy")],
            eval_datasets=[DatasetIdFilter("arc_easy")],
            probe_methods=[
                "ccs",
                "lat",
                "dim",
                "lda",
                "lr",
                "lr-g",
                "pca",
                "pca-g",
            ],
            eval_splits=["validation", "train-hparams"],
        )
        dfv = to_dataframe(results_arc_easy).sort_values(
            ["supervised", "algorithm", "layer"]
        )
        fig = px.line(
            dfv,
            x="layer",
            y="accuracy",
            color="algorithm",
            line_dash="supervised",
            width=800,
        )
        fig.write_image(
            output_path / f"{STAGE_VERSION}/sanity_check_arc_easy.png", scale=3
        )
        plt.close()

    if "got_cities" in df["train"].unique():
        """
        Validating results 2: Do we get the same accuracy as the GoT paper?
        """
        results_got = run_pipeline(
            dataset=dataset,
            mcontext=mcontext,
            points=points,
            token_idxs=token_idxs,
            llm_ids=LLM_IDS,
            train_datasets=[
                DatasetIdFilter("got_cities"),
                DatasetIdFilter("got_larger_than"),
                DatasetIdFilter("got_sp_en_trans"),
            ],
            eval_datasets=[
                DatasetIdFilter("got_cities"),
                DatasetIdFilter("got_larger_than"),
                DatasetIdFilter("got_sp_en_trans"),
            ],
            probe_methods=["dim", "lda"],
            eval_splits=["validation", "train-hparams"],
        )

        dfv = to_dataframe(results_got)
        fig = px.line(
            dfv.query("eval == train").sort_values(["layer", "supervised"]),
            x="layer",
            y="accuracy",
            facet_col="algorithm",
            facet_row="train",
            width=800,
            height=600,
        )
        fig.write_image(
            output_path / f"{STAGE_VERSION}/sanity_check_got.png", scale=3
        )
        plt.close()

    if "race" in df["train"].unique():
        """
        Validating results 3: Do we get the same accuracy as the RepE paper?
        """
        train_datasets: list[DatasetId] = [
            "race",
            "open_book_qa",
            "arc_easy",
            "arc_challenge",
        ]
        eval_datasets: list[DatasetId] = [
            "race/qa",
            "open_book_qa/qa",
            "arc_easy/qa",
            "arc_challenge/qa",
        ]
        eval_datasets = train_datasets
        results_RepE = run_pipeline(
            dataset=dataset,
            mcontext=mcontext,
            points=points,
            token_idxs=token_idxs,
            llm_ids=LLM_IDS,
            train_datasets=list(map(DatasetIdFilter, train_datasets)),
            eval_datasets=list(map(DatasetIdFilter, eval_datasets)),
            probe_methods=["lat"],
            eval_splits=["validation", "train-hparams"],
        )

        dfv = to_dataframe(results_RepE)
        dfv = dfv[dfv["train"] == dfv["eval"]]  # .str.rstrip("/qa")]
        fig = px.line(
            dfv.sort_values("layer"),  # type: ignore
            x="layer",
            y="accuracy",
            color="train",
            color_discrete_map={
                "race": "red",
                "open_book_qa": "blue",
                "arc_easy": "green",
                "arc_challenge": "orange",
            },
            range_y=[0, 1],
            width=800,
        )
        fig.add_hline(y=0.459, line_dash="dot", line_color="red")
        fig.add_hline(y=0.547, line_dash="dot", line_color="blue")
        fig.add_hline(y=0.803, line_dash="dot", line_color="green")
        fig.add_hline(y=0.532, line_dash="dot", line_color="orange")
        fig.write_image(
            output_path / f"{STAGE_VERSION}/sanity_check_repe.png", scale=3
        )
        plt.close()

    if "LDA" in df["algorithm"].unique():
        """
        Validating results 4: Investigating LDA performance
        """
        fig = px.bar(
            pd.concat(
                [
                    df.query("train == 'boolq'").assign(type="train_boolq"),
                    df.query("train == eval").assign(type="train_eval"),
                ]
            )
            .query("algorithm == 'LDA' or algorithm == 'DIM'")
            .query("layer == 5" if DEBUG else "layer == 21"),
            y="accuracy",
            x="eval",
            color="algorithm",
            facet_row="type",
            barmode="group",
            width=800,
        )
        fig.write_image(
            output_path / f"{STAGE_VERSION}/sanity_check_lda.png", scale=3
        )
        plt.close()


DIMS = {
    "llm_id",
    "train",
    "eval",
    "algorithm",
    "layer",
}
DLK_DATASETS = resolve_dataset_ids("dlk")
REPE_DATASETS = resolve_dataset_ids("repe")
GOT_DATASETS = resolve_dataset_ids("got")
COLORS = list(reversed(px.colors.sequential.YlOrRd))
LLM_IDS = get_model_ids()
COLLECTIONS = get_ds_collection_ids()
STAGE_VERSION = DEBUG_VERSION if DEBUG else "v1.10"  # To reset caching...
DATASETS_TO_EVAL = get_dataset_ids(include_validation_ds_only=False)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    if DEBUG:
        activation_ds_paths = [ACTIVATION_DIR / f"debug_{STAGE_VERSION}.pickle"]
    else:
        activation_ds_paths = [
            ACTIVATION_DIR
            # / "datasets_2024-03-05T08:37:13.825450.pickle"
            # / "datasets_2024-03-05T10:23:50.961381.pickle"
            / "datasets_2024-03-05T22:08:23.337252.pickle"
            # "s3://repeng/datasets/activations/datasets_2024-02-14_v1.pickle",
            # "s3://repeng/datasets/activations/datasets_2024-02-23_truthfulqa_v1.pickle",
        ]

    compute_and_plot_all_comparisons(
        activation_ds_paths,
    )
