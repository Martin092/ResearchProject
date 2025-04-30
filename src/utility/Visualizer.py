import os.path

import matplotlib.pyplot as plt
import pandas as pd


class Visualizer:

    def __init__(self, result_directory, do_export = False, do_show = False):

        self.dir = result_directory
        self.log_filepath = os.path.join(self.dir, "log.csv")
        self.figures_dir = os.path.join(self.dir, "figures")

        os.makedirs(self.figures_dir, exist_ok=True)

        self.do_export = do_export
        self.do_show = do_show


    def generate_graphs(self):

        df = pd.read_csv(self.log_filepath, sep=',', header=0, encoding='utf-8')

        # Project to reduce compute resources
        df = df[['Name', 'Trial', 'Round', 'Reward', 'Regret']]

        # Sort the dataframe for convenience
        df.sort_values(['Name', 'Trial', 'Round'])

        # Compute the cumulative reward
        df['cum_reward'] = df.groupby(['Name', 'Trial'])['Reward'].cumsum()
        df['cum_regret'] = df.groupby(['Name', 'Trial'])['Regret'].cumsum()
        df['cum_min_regret'] = df.groupby(['Name', 'Trial'])['Regret'].cummin()
        df['cum_avg_regret'] = df['cum_regret']  / df['Round']

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
        self._generate_reward_graph(data)
        self._generate_regret_graphs(data)





    def _generate_reward_graph(self, data):


        names = data['Name'].to_numpy()
        time = data['Round'].to_numpy()
        reward = data['avg_cum_reward'].to_numpy()
        std_reward = data['std_cum_reward'].to_numpy()


        plt.figure()

        plt.plot(time, reward, label='Mean')
        plt.fill_between(time, reward - std_reward, reward + std_reward, alpha=0.3, label='±1 std. dev.')

        plt.xlabel('Round t')
        plt.ylabel('Cumulative Reward')
        plt.title(f"Cumulative Reward for {names[-1]}")
        plt.legend()
        plt.tight_layout()


        self.do_export and plt.savefig(os.path.join(self.figures_dir, "cumulative_reward.png"), dpi=300, bbox_inches='tight', format='png')
        self.do_show and (plt.show())
        plt.close('all')

        pass

    def _generate_regret_graphs(self, data):


        names = data['Name'].to_numpy()
        time = data['Round'].to_numpy()

        cum_regret = data['avg_cum_regret'].to_numpy()
        std_cum_regret = data['std_cum_regret'].to_numpy()

        simp_regret  = data['avg_cum_min_regret'].to_numpy()
        std_simp_regret = data['std_cum_min_regret'].to_numpy()

        avg_regret = data['avg_cum_avg_regret'].to_numpy()
        std_avg_regret = data['std_cum_avg_regret'].to_numpy()

        plt.figure()

        plt.plot(time, cum_regret, label='Mean')
        plt.fill_between(time, cum_regret - std_cum_regret, cum_regret + std_cum_regret, alpha=0.3, label='±1 std. dev.')

        plt.xlabel('Round t')
        plt.ylabel('Cumulative Regret')
        plt.title(f"Cumulative Regret for {names[-1]}")
        plt.legend()
        plt.tight_layout()

        self.do_export and plt.savefig(os.path.join(self.figures_dir, "cumulative_regret.png"), dpi=300, bbox_inches='tight', format='png')
        self.do_show and plt.show()

        plt.figure()

        plt.plot(time, simp_regret, label='Mean')
        plt.fill_between(time, simp_regret - std_simp_regret, simp_regret + std_simp_regret, alpha=0.3, label='±1 std. dev.')

        plt.xlabel('Round t')
        plt.ylabel('Simple Regret')
        plt.title(f"Simple Regret for {names[-1]}")
        plt.legend()
        plt.tight_layout()

        self.do_export and plt.savefig(os.path.join(self.figures_dir, "simple_regret.png"), dpi=300, bbox_inches='tight', format='png')
        self.do_show and plt.show()

        plt.figure()

        plt.plot(time, avg_regret, label='Mean')
        plt.fill_between(time, avg_regret - std_avg_regret, avg_regret + std_avg_regret, alpha=0.3, label='±1 std. dev.')

        plt.xlabel('Round t')
        plt.ylabel('Average Regret')
        plt.title(f"Average Regret for {names[-1]}")
        plt.legend()
        plt.tight_layout()


        self.do_export and plt.savefig(os.path.join(self.figures_dir, "average_regret.png"), dpi=300, bbox_inches='tight', format='png')
        self.do_show and plt.show()
