import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class Visualizer:

    def __init__(self, result_directory, do_export = False, do_show = False):

        self.dir = result_directory
        self.log_filepath = os.path.join(self.dir, "log.csv")
        self.figures_dir = os.path.join(self.dir, "figures")

        os.makedirs(self.figures_dir, exist_ok=True)

        self.do_export = do_export
        self.do_show = do_show

    def generate_graphs(self, plot_name=None):

        df = pd.read_csv(self.log_filepath, sep=',', header=0, encoding='utf-8')

        # Project to reduce compute resources
        # df = df[['Name', 'Trial', 'Round', 'Reward', 'Regret']]

        # Sort the dataframe for convenience
        df.sort_values(['Name', 'Trial', 'Round'])

        # Compute the cumulative reward
        df['cum_reward'] = df.groupby(['Name', 'Trial'])['Reward'].cumsum()
        df['cum_regret'] = df.groupby(['Name', 'Trial'])['Regret'].cumsum()
        df['cum_min_regret'] = df.groupby(['Name', 'Trial'])['Regret'].cummin()
        df['cum_avg_regret'] = df['cum_regret'] / df['Round']

        data = (
            df
            .groupby(['Name', 'Round'])[['cum_reward', 'cum_regret', 'cum_min_regret', 'cum_avg_regret']]
            .agg(
                avg_cum_reward=('cum_reward', 'mean'),
                std_cum_reward=('cum_reward', 'std'),
                avg_cum_regret=('cum_regret', 'mean'),
                std_cum_regret=('cum_regret', 'std'),
                avg_cum_min_regret=('cum_min_regret', 'mean'),
                std_cum_min_regret=('cum_min_regret', 'std'),
                avg_cum_avg_regret=('cum_avg_regret', 'mean'),
                std_cum_avg_regret=('cum_avg_regret', 'std'),
            )
            .reset_index()
        )

        # Generate The graphs
        # self._generate_reward_graph(data)
        # self._generate_arms_hist(df)
        # self._generate_reward_per_arm_graph(df)
        self._generate_regret_graphs(data, plot_name)

    def _generate_arms_hist(self, df):
        names = df['Name'].to_list()

        actions = df[["Name", "Action_index"]]
        for name in set(names):
            # time = data.loc[data['Name'] == name, 'Round'].to_numpy()
            # reward = data.loc[data['Name'] == name,'avg_cum_reward'].to_numpy()
            # std_reward = data.loc[data['Name'] == name,'std_cum_reward'].to_numpy()
            vals = actions.loc[actions["Name"] == name]["Action_index"]
            if vals.isna().all():
                continue

            plt.figure()
            plt.hist(vals.to_numpy(), bins=np.arange(vals.min(), vals.max() + 2) - 0.5, align='mid')
            plt.xlabel('Action index')
            plt.ylabel('Number of pulls')
            plt.title(f"Frequency of arm pulls for {name}")

            self.do_export and plt.savefig(os.path.join(self.figures_dir, f"action_distr_{name}.png"), dpi=300, bbox_inches='tight', format='png')
            self.do_show and (plt.show())

    def _generate_reward_per_arm_graph(self, df):
        names = df['Name'].to_list()
        df = df[["Name", "Action_index", "Reward"]]
        vals = df.groupby(['Name', 'Action_index'])['Reward'].sum()
        for name in set(names):
            data = vals[name]

            plt.figure()
            plt.bar(np.arange(1, data.shape[0] + 1), data.to_numpy())
            plt.xlabel('Action')
            plt.ylabel('Reward')
            plt.title(f"Reward per arm for {name}")
            # plt.grid()
            # plt.show()

        self.do_export and plt.savefig(os.path.join(self.figures_dir, "reward_per_action.png"), dpi=300,
                                       bbox_inches='tight', format='png')
        self.do_show and plt.show()

    def _generate_reward_graph(self, data):

        names = data['Name'].to_list()

        plt.figure()

        for name in set(names):

            time = data.loc[data['Name'] == name, 'Round'].to_numpy()
            reward = data.loc[data['Name'] == name,'avg_cum_reward'].to_numpy()
            std_reward = data.loc[data['Name'] == name,'std_cum_reward'].to_numpy()

            plt.plot(time, reward, label=f"{name}")
            plt.fill_between(time, reward - std_reward, reward + std_reward, alpha=0.1)

        plt.xlabel('Round t', fontsize=16)
        plt.ylabel('Cumulative Reward', fontsize=16)
        plt.title("Cumulative Reward across Rounds", fontsize=18)
        plt.legend()
        plt.tight_layout()

        self.do_export and plt.savefig(os.path.join(self.figures_dir, "cumulative_reward.png"), dpi=300, bbox_inches='tight', format='png')
        self.do_show and (plt.show())

    def _generate_regret_graphs(self, data, plot_name=None):

        names = data['Name'].to_numpy()

        plt.figure()

        for name in set(names):
            time = data.loc[data['Name'] == name,'Round'].to_numpy()
            cum_regret = data.loc[data['Name'] == name,'avg_cum_regret'].to_numpy()
            std_cum_regret = data.loc[data['Name'] == name,'std_cum_regret'].to_numpy()

            plt.plot(time, cum_regret, label=f"{name}")
            plt.fill_between(time, cum_regret - std_cum_regret, cum_regret + std_cum_regret, alpha=0.1)

        # plt.xlabel('Round', fontsize=16)
        # plt.ylabel('Cumulative Regret', fontsize=16)
        # plt.title("Cumulative Regret across Rounds", fontsize=18)
        plt.xlabel('Round')
        plt.ylabel('Cumulative Regret')
        if plot_name is None:
            plt.title("Cumulative Regret across Rounds")
        else:
            plt.title(plot_name)

        plt.legend()
        plt.tight_layout()

        self.do_export and plt.savefig(os.path.join(self.figures_dir, "cumulative_regret.png"), dpi=300, bbox_inches='tight', format='png')
        self.do_show and plt.show()

        # plt.figure()
        #
        # for name in set(names):
        #
        #     time = data.loc[data['Name'] == name, 'Round'].to_numpy()
        #     simp_regret = data.loc[data['Name'] == name,'avg_cum_min_regret'].to_numpy()
        #     std_simp_regret = data.loc[data['Name'] == name,'std_cum_min_regret'].to_numpy()
        #
        #     plt.plot(time, simp_regret, label=f"{name}")
        #     plt.fill_between(time, simp_regret - std_simp_regret, simp_regret + std_simp_regret, alpha=0.1)
        #
        # plt.xlabel('Round t', fontsize=16)
        # plt.ylabel('Simple Regret', fontsize=16)
        # plt.title("Simple Regret across Rounds", fontsize=18)
        # plt.legend()
        # plt.tight_layout()
        #
        # self.do_export and plt.savefig(os.path.join(self.figures_dir, "simple_regret.png"), dpi=300, bbox_inches='tight', format='png')
        # self.do_show and plt.show()
        #
        # plt.figure()
        #
        # for name in set(names):
        #
        #     time = data.loc[data['Name'] == name, 'Round'].to_numpy()
        #     avg_regret = data.loc[data['Name'] == name,'avg_cum_avg_regret'].to_numpy()
        #     std_avg_regret = data.loc[data['Name'] == name,'std_cum_avg_regret'].to_numpy()
        #
        #     plt.plot(time, avg_regret, label=f"{name}")
        #     plt.fill_between(time, avg_regret - std_avg_regret, avg_regret + std_avg_regret, alpha=0.1)
        #
        # plt.xlabel('Round t', fontsize=16)
        # plt.ylabel('Average Regret', fontsize=16)
        # plt.title("Average Regret across Rounds", fontsize=18)
        # plt.legend()
        # plt.tight_layout()

        # self.do_export and plt.savefig(os.path.join(self.figures_dir, "average_regret.png"), dpi=300, bbox_inches='tight', format='png')
        # self.do_show and plt.show()
