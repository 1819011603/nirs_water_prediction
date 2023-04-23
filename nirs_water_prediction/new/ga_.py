import numpy as np
from genetic_selection import GeneticSelectionCV
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import cross_val_score


from utils import save2excel
from main import train_n_times, para

p = []

def func():
    from nirs.parameters import X_train, y_train, X_test, y_test
    estimator = PLSRegression(n_components=2)

    selector = GeneticSelectionCV(estimator,
                                  cv=5,
                                  verbose=1,
                                  scoring="neg_root_mean_squared_error",
                                  n_population=100,
                                  n_generations=10,
                                  mutation_independent_proba=0.05,
                                  max_features=None,

                                  caching=True,

                                  n_jobs=1)

    # Train and select the features
    selector.fit(X_train, y_train)
    # print(selector.support_.shape)
    # print(selector.support_)
    features_sel = np.arange(len(X_train[0]))[selector.support_]
    print(features_sel)
    return  features_sel

def main(func,filename,feature_):
    for i in range(50):
        features_sel = func()
        p.append(features_sel)
        para.paint = False
        train_n_times(features_sel, features=feature_)
    para.train_n = np.array(para.train_n)
    c = para.train_n
    mean = np.mean(c, axis=0)
    std = np.std(c, axis=0)
    d = np.argmax(c, axis=0)
    colomn = ['R', 'r', 'R2', 'RMSECV', "r2", 'RMSEP', "RPD", 'MAE', "CPU时间", "流程时间", "总时间"]

    mean = [f"{x:.4f}" for x in mean]

    std = [f"{x:.4f}" for x in std]
    a = []
    b = []
    for i, j in enumerate(colomn):
        print(f"{j} = {mean[i]}±{std[i]} ")
        a.append(j)
        b.append(f'{mean[i]}±{std[i]}')
    a.extend(['R_max', 'r_max', 'R_max_index', 'r_max_index'])
    b.append(f'{c[d[0]][0]:.4f}')
    b.append(f'{c[d[1]][1]:.4f}')
    b.append(f"[{','.join(np.array(p[d[0]],dtype=str))}]")
    b.append(f"[{','.join(np.array(p[d[1]],dtype=str))}]")
    b = [f"{str(x)}" for x in b]
    save2excel(b, a, filename)

if __name__ == '__main__':

    main(func,"ga_time.xlsx",'ga')



