import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import yaml
import os
import argparse


plt.style.use("matplotlibrc")


# names for columns and labels
N_WEDGE = "\\ensuremath{N_{\\wedge}}"
DATASET_SIZE = "\\ensuremath{|\\mathcal{D}|}"
EXPECTED_RETURN = "Expected Return"
METRIC = "Metric"
ALGORITHM = "Algorithm"
BEHAVIOR_K = "\\ensuremath{k}"
TARGET_K = "\\ensuremath{k'}'"


def main(conf):
    with open(f'plotting_settings.yaml', 'r') as f:
        plotting_settings = yaml.safe_load(f)
    # import pprint
    # pprint.pprint(conf)
    assert conf["experiment_id"] in plotting_settings.keys()
    conf.update(plotting_settings[conf['experiment_id']])
    conf["out_dir"] = os.path.join("plots", conf['experiment_id'])

    with open(conf["pib"], "r") as f:
        data = yaml.safe_load(f)
    conf.update(data)

    dfs = [
        pd.read_csv(csv_file, index_col=0)
        for csv_file in glob.glob(conf["results"]+"*.csv")
    ]
    df = pd.concat(dfs).reset_index(drop=True)

    df.rename(inplace=True, columns={
        "algorithm": ALGORITHM,
        "expected_return": EXPECTED_RETURN,
        "dataset_size": DATASET_SIZE,
        "n_wedge": N_WEDGE,
        "behavior_k": BEHAVIOR_K,
        "target_k": TARGET_K
    })
    conf["max_performance"] = df[EXPECTED_RETURN].max()

    # FIXME: here we adjust the labels to make the experiments consistent with the notation of the paper
    # once the code if fixed, these lines can be removed
    df[TARGET_K] += 1
    df[BEHAVIOR_K] += 1
    conf[BEHAVIOR_K] = data["k"] + 1
    conf["out_dir"] = conf["out_dir"].replace(f"k-{data['k']}", f"k-{conf[BEHAVIOR_K]}")

    print(conf['experiment_id'])

    os.makedirs(conf["out_dir"], exist_ok=True)

    conf['n_seeds'] = len(df["seed"].unique())
    print(f"Plotting {conf['experiment_id']}. Found {conf['n_seeds']}.")
    df[METRIC] = "Mean"
    cols = [ALGORITHM, DATASET_SIZE, N_WEDGE, TARGET_K]
    cvar10_df = df.sort_values(['Expected Return'], ascending=False).groupby(cols).tail(tail_size(10, conf['n_seeds']))
    cvar10_df[METRIC] = "10\\%-CVaR"
    cvar1_df = df.sort_values(['Expected Return'], ascending=False).groupby(cols).tail(tail_size(1, conf['n_seeds']))
    cvar1_df[METRIC] = "1\\%-CVaR"

    combined_df = pd.concat([df, cvar1_df, cvar10_df]).reset_index(drop=True)

    combined_df[N_WEDGE].unique()
    combined_df[TARGET_K].unique()
    combined_df[METRIC].unique()
    legend_saved = False
    for k in df[TARGET_K].unique():
        for n in df[N_WEDGE].unique():
            if n == 0:
                continue
            selection = (combined_df[TARGET_K] == k) & ((combined_df[N_WEDGE] == n) | (combined_df[N_WEDGE] == 0))

            ax = sns.lineplot(
                data=combined_df[selection], x=DATASET_SIZE, y=EXPECTED_RETURN,
                hue=ALGORITHM,
                markers=True,
                style=METRIC,
                style_order=['Mean', '10\\%-CVaR', '1\\%-CVaR'],
                ci=None,
            )
            ax = sns.lineplot(
                data=combined_df[selection],
                x=DATASET_SIZE,
                y=conf["baseline_performance"],
                ci=None,
                ax=ax,
                linestyle="-.",
                label="Behavior Policy",
                color='green',
            )
            ax = sns.lineplot(
                data=combined_df[selection],
                x=DATASET_SIZE,
                y=conf["pouct-perf"],
                ci=None,
                ax=ax,
                linestyle=':',
                label="PO-UCT",
                color='purple',
            )
            ax.set(xscale="log")

            ax.set_title(f"{conf['env']}(${N_WEDGE} = {n},k = {conf[BEHAVIOR_K]}, k' = {k}$)")
            ax.set_ylim(*conf["y_lim"])

            sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))

            if not legend_saved:
                if conf['pdf']:
                    export_legend(ax, os.path.join(conf["out_dir"], "curve_legend.pdf"))
                if conf['png']:
                    export_legend(ax, os.path.join(conf["out_dir"], "curve_legend.png"))
                legend_saved = True
            if not conf['legend']:
                ax.get_legend().remove()
            if conf['png']:
                ax.figure.savefig(os.path.join(conf["out_dir"], f"curve_k_{k}_N_{n:0>3}.png"), bbox_inches='tight')
            if conf['pdf']:
                ax.figure.savefig(os.path.join(conf["out_dir"], f"curve_k_{k}_N_{n:0>3}.pdf"), bbox_inches='tight')

            if conf['show']:
                plt.show()
            plt.clf()

    # g = sns.catplot(x=dataset_size, y="Expected Return", hue="Algorithm", data=df, row=target_k, col=n_wedge)

    plot_heatmap(conf, df, k_prime=conf[BEHAVIOR_K])
    plot_heatmap(conf, df, k_prime=conf[BEHAVIOR_K] + 1)
    plot_heatmap(conf, df, k_prime=conf[BEHAVIOR_K], cvar=1)
    plot_heatmap(conf, df, k_prime=conf[BEHAVIOR_K] + 1, cvar=1)
    plot_heatmap(conf, df, k_prime=conf[BEHAVIOR_K], cvar=10)
    plot_heatmap(conf, df, k_prime=conf[BEHAVIOR_K] + 1, cvar=10)


def plot_heatmap(conf, df, k_prime, cvar=100):
    df = df[df[TARGET_K] == k_prime]
    cols = [ALGORITHM, DATASET_SIZE, N_WEDGE]
    if cvar < 100:
        df = df.sort_values([EXPECTED_RETURN], ascending=False).groupby(cols).tail(tail_size(cvar, conf['n_seeds']))
    df = df.groupby(cols)[EXPECTED_RETURN].mean().reset_index()
    df["Normalized Performance"] = (df[EXPECTED_RETURN] - conf["baseline_performance"]) / (
            conf["max_performance"] - conf["baseline_performance"])

    p = df.pivot(N_WEDGE, DATASET_SIZE, 'Normalized Performance')
    p.sort_values(N_WEDGE, ascending=False, inplace=True)

    if not conf['legend']:
        main_fig, ax = plt.subplots()
        cbar_fig, cbar_ax = plt.subplots(figsize=(0.6, 2.49))
    else:
        fig, (ax, cbar_ax) = plt.subplots(1, 2, gridspec_kw={"width_ratios": (10, 1), "wspace": 0.4})

    sns.heatmap(
        p,
        annot=False,
        linewidths=2,
        ax=ax,
        cbar_ax=cbar_ax,
        center=0,
        cmap="coolwarm",
        vmin=-3.3,
        vmax=1.1
    )

    ticks, labels = ax.get_xticks(), ax.get_xticklabels()
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels, rotation=-45, ha='left', rotation_mode='anchor')
    ax.tick_params(axis='y', labelrotation=0)
    ax.set_ylabel(ax.get_ylabel(), rotation=0)

    title = f"{conf['env']} (${BEHAVIOR_K} = {conf[BEHAVIOR_K]}, {TARGET_K} = {k_prime}$)"
    if cvar < 100:
        ax.set_title(title + "~" + f" {cvar}\\%-CVaR")
        name = f"heatmap_k_{k_prime}_cvar_{cvar}"
    else:
        ax.set_title(title + "~Mean")
        name = f"heatmap_k_{k_prime}_mean"
    cbar_ax.set_title("$\\bar{\\rho}(\\pi_I)$")
    if not conf['legend']:
        if conf['png']:
            cbar_ax.figure.savefig(os.path.join(conf["out_dir"], f"heatmap_legend.png"), bbox_inches='tight')
        if conf['pdf']:
            cbar_ax.figure.savefig(os.path.join(conf["out_dir"], f"heatmap_legend.pdf"), bbox_inches='tight')
    if conf['png']:
        ax.figure.savefig(os.path.join(conf["out_dir"], f"{name}.png"), bbox_inches='tight')
    if conf['pdf']:
        ax.figure.savefig(os.path.join(conf["out_dir"], f"{name}.pdf"), bbox_inches='tight')
    if conf['show']:
        plt.show()


def tail_size(cvar, n):
    assert isinstance(cvar, int)
    return int(max(cvar * n / 100, 1))


def export_legend(ax, filename):
    fig2 = plt.figure()
    ax2 = fig2.add_subplot()
    ax2.axis('off')
    legend = ax2.legend(*ax.get_legend_handles_labels(), frameon=False, loc='lower center')
    fig = legend.figure
    fig.canvas.draw()
    bbox = legend.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(filename, dpi="figure", bbox_inches=bbox)


VALID_EXPERIMENTS = [
    'CheeseMaze-v0_k-0',
    'CheeseMaze-v0_k-1',
    'CheeseMaze-v0_k-2',
    'Tiger-v0_k-0',
    'Tiger-v0_k-1',
    'Tiger-v0_k-2',
    'Voicemail-v0_k-0',
    'Voicemail-v0_k-1',
    'Voicemail-v0_k-2'
]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment_id", type=str, choices=VALID_EXPERIMENTS)
    parser.add_argument("--show", default=False, action='store_true')
    parser.add_argument("--pdf", default=False, action='store_true')
    parser.add_argument("--png", default=False, action='store_true')
    parser.add_argument("--legend", default=False, action='store_true')
    main(vars(parser.parse_args()))
