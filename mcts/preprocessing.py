import polars as pl


class Preprocess:
    def __init__(self):
        self.target = "utility_agent1"
        self.cols_to_drop = set()
        self.cat_features = set()
        self.label_encoders = {}

    def set_cols_to_drop(self, df: pl.DataFrame):
        cols_to_drop = set()
        for col in df.columns:
            unique_vals = df.select(col).unique()
            if unique_vals.shape[0] == 1:
                cols_to_drop.add(col)

        cols_to_drop.update(
            [
                "Id",
                "EnglishRules",
                "LudRules",
                "GameRulesetName",
                "num_wins_agent1",
                "num_draws_agent1",
                "num_losses_agent1",
            ]
        )
        self.cols_to_drop.update(cols_to_drop)

    def set_cat_features(self, df: pl.DataFrame):
        cat_features = df.select(pl.col(pl.String)).columns
        cat_features = set(cat_features).difference(self.cols_to_drop)
        self.cat_features.update(cat_features)

    def create_agent_features(self, df: pl.DataFrame):
        df = (
            df.with_columns(
                agent1_split=pl.col("agent1").str.split(by="-"),
                agent2_split=pl.col("agent2").str.split(by="-"),
            )
            .with_columns(
                pl.col("agent1_split").list.to_struct(
                    fields=[
                        "MCTS_agent1",
                        "selection_agent1",
                        "exploration_const_agent1",
                        "playout_agent1",
                        "score_bounds_agent1",
                    ]
                )
            )
            .with_columns(
                pl.col("agent2_split").list.to_struct(
                    fields=[
                        "MCTS_agent2",
                        "selection_agent2",
                        "exploration_const_agent2",
                        "playout_agent2",
                        "score_bounds_agent2",
                    ]
                )
            )
            .unnest("agent1_split", "agent2_split")
        )

        self.cols_to_drop.update({"MCTS_agent1", "MCTS_agent2", "agent1", "agent2"})

        self.cat_features.update(
            {
                "selection_agent1",
                "exploration_const_agent1",
                "playout_agent1",
                "score_bounds_agent1",
                "selection_agent2",
                "exploration_const_agent2",
                "playout_agent2",
                "score_bounds_agent2",
            }
        )

        self.cat_features = self.cat_features.difference(self.cols_to_drop)

        return df

    def fit_label_encoders(self):
        pass

    def encode_categorical_features(self, df: pl.DataFrame):
        pass

    def drop_columns(self, df: pl.DataFrame):
        return df.drop(*self.cols_to_drop, strict=False)

    def fit_transform(self, df: pl.DataFrame):
        self.set_cols_to_drop(df)
        self.set_cat_features(df)
        df = self.transform(df)
        return df

    def transform(self, df: pl.DataFrame):
        df = self.create_agent_features(df)
        df = self.drop_columns(df)
        return df
