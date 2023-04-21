def fetch_fish(data_train,data_test,target):

    X_train=data_train.drop(target,axis=1)
    X_test=data_test.drop(target,axis=1)

    Y_train=data_train[[target]]
    Y_test=data_test[[target]]

    X_train = X_train.to_numpy()
    Y_train = Y_train.to_numpy().reshape(-1,)

    X_test = X_test.to_numpy()
    Y_test = Y_test.to_numpy().reshape(-1,)

    return X_train, Y_train, X_test, Y_test

def train_xgb(n_trees,learning_rate, max_depth,subsample,dev,oot,target):

    x_train, y_train, x_test, y_test = fetch_fish(data_train=dev,data_test=oot,target=target)
    import xgboost as xgb
    from sklearn.metrics import accuracy_score

    xgb_cl = xgb.XGBClassifier(x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test,subsample=subsample,
                n_trees=n_trees,learning_rate=learning_rate, max_depth=max_depth)
    model=xgb_cl.fit(x_train,y_train)
    y_pred=model.predict(x_test)
    loss = 1-(accuracy_score(y_test, y_pred))
    return loss


