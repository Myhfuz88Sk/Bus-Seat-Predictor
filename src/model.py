from sklearn.ensemble import RandomForestRegressor

def get_model(n_estimators=100, random_state=42):
    """
    Returns an initialized RandomForestRegressor model.

    Args:
        n_estimators (int): The number of trees in the forest.
        random_state (int): Controls the randomness of the bootstrapping of the samples
                            when building trees.

    Returns:
        sklearn.ensemble.RandomForestRegressor: An initialized RandomForestRegressor model.
    """
    model = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state, n_jobs=-1)
    return model