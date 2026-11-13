import pandas as pd
import numpy as np
import torch
import lightgbm
import scipy.sparse as sp

from pathlib import Path
from transformers import AutoTokenizer
from adapters import AutoAdapterModel
from sklearn.neighbors import NearestNeighbors, KNeighborsRegressor
from sklearn.preprocessing import normalize
from collections import Counter
from sklearn.model_selection import KFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.preprocessing import OneHotEncoder
from sklearn.cluster import KMeans
from scipy.stats import rankdata
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
train_df = pd.read_csv(DATA_DIR / "train.csv")
test_df = pd.read_csv(DATA_DIR / "test_fixed_clean.csv")


model_name = "allenai/specter2_base"
tokenizer=AutoTokenizer.from_pretrained(model_name)
model=AutoAdapterModel.from_pretrained(model_name)
adapter_name = model.load_adapter(
    "allenai/specter2",
    source="hf",
    set_active=True
)
model.eval()
model.set_active_adapters(adapter_name)
def get_embedding(df):
    token=tokenizer(
        df["title"].fillna("").tolist(),
        padding=True,
        truncation=True,
        return_tensors='pt'
    )
    with torch.no_grad():
        output=model(**token)
    hidden=output.last_hidden_state
    emb_cls=hidden[:,0,:]
    return emb_cls.numpy()

def get_metadata_features(reference_df, target_df):
    reference_df = reference_df.copy()
    target_df = target_df.copy()
    reference_year = pd.to_numeric(
        reference_df["year"], errors="coerce"
    )
    target_year = pd.to_numeric(
        target_df["year"], errors="coerce"
    )
    year_max = reference_year.max()
    venue_freq = (
        reference_df["venue"]
        .fillna("unknown")
        .astype(str)
        .str.strip()
        .str.lower()
        .value_counts(normalize=True)
    )
    target_venue = (
        target_df["venue"]
        .fillna("unknown")
        .astype(str)
        .str.strip()
        .str.lower()
    )
    venue_feature = target_venue.map(venue_freq).fillna(0)
    author_counter = Counter()
    for authors in reference_df["authors"].fillna(""):
        author_list = [
            a.strip().lower()
            for a in str(authors).split(",")
            if a.strip()
        ]
        author_counter.update(set(author_list))
    author_count = []
    author_freq_mean = []
    author_freq_max = []
    for authors in target_df["authors"].fillna(""):
        author_list = [
            a.strip().lower()
            for a in str(authors).split(",")
            if a.strip()
        ]
        author_list = list(set(author_list))
        freqs = [author_counter[a] for a in author_list]
        author_count.append(len(author_list))
        if len(freqs) > 0:
            author_freq_mean.append(np.mean(freqs))
            author_freq_max.append(max(freqs))
        else:
            author_freq_mean.append(0)
            author_freq_max.append(0)
    features = pd.DataFrame({
        "year": target_year,
        "paper_age": year_max - target_year,
        "venue_freq": venue_feature,
        "author_count": author_count,
        "author_freq_mean": author_freq_mean,
        "author_freq_max": author_freq_max
    })
    return features.fillna(0).to_numpy(dtype=float)
def lgbm(X_train, y_train ):
    clf=lightgbm.LGBMClassifier(
        objective='multiclass',
    n_estimators=200,        
    learning_rate=0.05,
    num_leaves=15,
    min_child_samples=30,      
    colsample_bytree=0.5,       
    reg_lambda=5,             
    random_state=42,
    verbose=-1                  
    )
    clf.fit(X_train, y_train)
    return clf


def embed(df, batch_size=32):
    return np.vstack([
        get_embedding(df.iloc[i:i + batch_size])
        for i in range(0, len(df), batch_size)
    ])


def expected_gain(clf, X):
    proba = clf.predict_proba(X)
    return proba @ ((clf.classes_ - 1) / 4)


def compute_q_r(labels_in_order, labels_all):
    labels_in_order = np.asarray(labels_in_order, dtype=float)
    m = len(labels_in_order)
    n_good = (labels_all > 1).sum()
    if m == 0 or n_good == 0:
        return 0.0, 0.0
    gain = (labels_in_order - 1) / 4
    w = 1 / np.log2(np.arange(2, m + 2))
    Q = (w * gain).sum() / w.sum()
    R = (labels_in_order > 1).sum() / n_good
    return Q, R


def compute_score(Q, R):
    return 0.0 if Q + R == 0 else 2 * Q * R / (Q + R)


def evaluate(pred, labels, keep_frac=1.0, shuffle=False, n_sub=200, size=298, seed=0):
    rng = np.random.default_rng(seed)
    qs, rs, ss = [], [], []
    for _ in range(n_sub):
        s = rng.choice(len(labels), size, replace=False)
        order = rng.permutation(size) if shuffle else np.argsort(-pred[s], kind="stable")
        n_keep = max(1, int(round(keep_frac * size)))
        Q, R = compute_q_r(labels[s][order[:n_keep]], labels[s])
        qs.append(Q); rs.append(R); ss.append(compute_score(Q, R))
    return np.mean(qs), np.mean(rs), np.mean(ss)


def make_priority(pred, keep_frac):
    order = np.argsort(-pred, kind="stable")
    n_keep = int(round(keep_frac * len(pred)))
    priority = np.zeros(len(pred))
    priority[order[:n_keep]] = 10 * np.arange(n_keep, 0, -1) / n_keep
    return priority

def get_kmean_rating(emb,y,alpha=10):
    z=normalize(emb,norm='l2')
    fold=KFold(5,shuffle=True,random_state=42)
    K_list=[20,50,100]
    results={}
    for k in K_list:
        km=KMeans(n_clusters=k,random_state=42).fit(z)
        labels=km.labels_
        oof_mean_rating = np.zeros(len(y))
        for tr_idx, va_idx in fold.split(z):
            c_train=labels[tr_idx]
            y_train=y[tr_idx]
            c_val=labels[va_idx]
            global_mean=y_train.mean()
            sum_rating=np.bincount(c_train,weights=y_train,minlength=k)
            count=np.bincount(c_train,minlength=k)
            cluster_mean=(sum_rating+alpha*global_mean)/(count+alpha)
            oof_mean_rating[va_idx]=cluster_mean[c_val]
        results[k] = oof_mean_rating
    return results

            
y = train_df["Label"].values.astype(int)
n_train = len(train_df)

emb = embed(train_df)
emb_test = embed(test_df)

Z_all = (np.vstack([emb, emb_test]))

all_df = pd.concat([train_df, test_df], ignore_index=True)
M_train = get_metadata_features(all_df, train_df)
M_test = get_metadata_features(all_df, test_df)

X_full = np.hstack([Z_all[:n_train], M_train])
X_test_full = np.hstack([Z_all[n_train:], M_test])

oof = np.zeros(n_train)
for tr_idx, va_idx in KFold(5, shuffle=True, random_state=42).split(X_full):
    oof[va_idx] = expected_gain(lgbm(X_full[tr_idx], y[tr_idx]), X_full[va_idx])


def ridge_meta(df):
    f = pd.DataFrame(index=df.index)
    f["year"] = (df["year"] - 2020) / 3
    f["s2"] = df["doi"].astype(str).str.contains("semanticscholar").astype(int)
    f["noauth"] = df["authors"].isna().astype(int)
    f["n_auth"] = (df["authors"].fillna("").str.count(",") + df["authors"].notna()) / 5
    f["tlen"] = df["title"].str.split().str.len() / 10
    return f


tf_word = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, stop_words="english").fit(all_df["title"])
tf_char = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3, sublinear_tf=True).fit(all_df["title"])
venue_oh = OneHotEncoder(handle_unknown="ignore").fit(all_df[["venue"]])


def ridge_design(df):
    return sp.hstack([
        tf_word.transform(df["title"]),
        tf_char.transform(df["title"]),
        venue_oh.transform(df[["venue"]]) * 2,
        sp.csr_matrix(ridge_meta(df).values.astype(float)),
    ]).tocsr()


T_train = ridge_design(train_df)
T_test = ridge_design(test_df)

oof_ridge = np.zeros(n_train)
for tr_idx, va_idx in KFold(5, shuffle=True, random_state=42).split(T_train):
    oof_ridge[va_idx] = Ridge(alpha=5).fit(T_train[tr_idx], y[tr_idx]).predict(T_train[va_idx])
test_ridge = Ridge(alpha=5).fit(T_train, y).predict(T_test)


clf = lgbm(X_full, y)
test_lgbm = expected_gain(clf, X_test_full)


def make_knn():
    return KNeighborsRegressor(n_neighbors=20, metric="cosine", weights="distance")


E_train = normalize(emb)
E_test = normalize(emb_test)

oof_knn = np.zeros(n_train)
for tr_idx, va_idx in KFold(5, shuffle=True, random_state=42).split(E_train):
    oof_knn[va_idx] = make_knn().fit(E_train[tr_idx], y[tr_idx]).predict(E_train[va_idx])
test_knn = make_knn().fit(E_train, y).predict(E_test)


def to_rank(p):
    return rankdata(p) / len(p)


def blend(p_ridge, p_lgbm, p_knn, w_ridge, w_lgbm, w_knn):
    return w_ridge * to_rank(p_ridge) + w_lgbm * to_rank(p_lgbm) + w_knn * to_rank(p_knn)


# ================= FEATURE CỤM KMEANS CHO LIGHTGBM =================
K_LIST = [50, 100, 200]
ALPHA = 10
folds = list(KFold(5, shuffle=True, random_state=42).split(X_full))


def make_cluster_info(emb_train, emb_test, k_list=K_LIST):
    E = normalize(np.vstack([emb_train, emb_test]))
    info = []
    for k in k_list:
        km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(E)
        lab = km.labels_
        size = np.bincount(lab, minlength=k)
        centers = normalize(km.cluster_centers_)
        cos_center = np.sum(E * centers[lab], axis=1)
        info.append((k, lab, np.log1p(size[lab]), cos_center))
    return info


def cluster_label_stats(lab_fit, y_fit, lab_apply, k, alpha=ALPHA):
    cnt = np.bincount(lab_fit, minlength=k)
    out = []
    for target in (y_fit.astype(float), (y_fit == 1).astype(float), (y_fit >= 4).astype(float)):
        s = np.bincount(lab_fit, weights=target, minlength=k)
        out.append(((s + alpha * target.mean()) / (cnt + alpha))[lab_apply])
    return np.column_stack(out)


def cluster_features(info, fit_idx, apply_idx, nested):
    cols = []
    for k, lab, log_size, cos_c in info:
        if nested:
            st = np.zeros((len(fit_idx), 3))
            for a, b in KFold(5, shuffle=True, random_state=0).split(fit_idx):
                st[b] = cluster_label_stats(lab[fit_idx[a]], y[fit_idx[a]], lab[fit_idx[b]], k)
        else:
            st = cluster_label_stats(lab[fit_idx], y[fit_idx], lab[apply_idx], k)
        cols += [st, log_size[apply_idx, None], cos_c[apply_idx, None]]
    return np.hstack(cols)


cluster_info = make_cluster_info(emb, emb_test)

oof_c = np.zeros(n_train)
for tr_idx, va_idx in folds:
    F_fit = cluster_features(cluster_info, tr_idx, tr_idx, nested=True)
    F_val = cluster_features(cluster_info, tr_idx, va_idx, nested=False)
    clf_c = lgbm(np.hstack([X_full[tr_idx], F_fit]), y[tr_idx])
    oof_c[va_idx] = expected_gain(clf_c, np.hstack([X_full[va_idx], F_val]))

all_idx = np.arange(n_train)
F_fit = cluster_features(cluster_info, all_idx, all_idx, nested=True)
F_test = cluster_features(cluster_info, all_idx, n_train + np.arange(len(test_df)), nested=False)
clf_c = lgbm(np.hstack([X_full, F_fit]), y)
test_c = expected_gain(clf_c, np.hstack([X_test_full, F_test]))

mask_c = (train_df["venue"] != "iclp").values
print("\n=== So sánh LightGBM có / không có feature cụm (300 tập con) ===")
scores = {}
for name, ol in [("LGBM cũ", oof), ("LGBM + cụm", oof_c)]:
    b = 0.5 * to_rank(oof_ridge) + 0.5 * to_rank(ol)
    auc_l = roc_auc_score(y[mask_c] > 1, ol[mask_c])
    auc_b = roc_auc_score(y[mask_c] > 1, b[mask_c])
    scores[name] = evaluate(b[mask_c], y[mask_c], keep_frac=0.65, n_sub=300)[2]
    print(f"{name:<12} AUC lgbm={auc_l:.4f}  AUC blend={auc_b:.4f}  score(w0.5, keep0.65)={scores[name]:.4f}")

gain = scores["LGBM + cụm"] - scores["LGBM cũ"]
oof, test_lgbm = oof_c, test_c
# ===================================================================

mask = (train_df["venue"] != "iclp").values
y_eval = y[mask]
keep_grid = np.round(np.arange(0.50, 1, 0.05), 2)


weight_list = []

for w_ridge in np.linspace(0, 1, 11):
    for w_lgbm in np.linspace(0, 1, 11):
        w_knn = 1 - w_ridge - w_lgbm
        if w_knn >= -1e-9:
            weight_list.append((round(w_ridge, 1), round(w_lgbm, 1), round(max(w_knn, 0), 1)))


best = (None, None, -1)
best_info = (None, None)
for w_ridge, w_lgbm, w_knn in weight_list:
    p_eval = blend(oof_ridge, oof, oof_knn, w_ridge, w_lgbm, w_knn)[mask]
    for f in keep_grid:
        Q, R, S = evaluate(p_eval, y_eval, keep_frac=f)
        if S > best[2]:
            best = ((w_ridge, w_lgbm, w_knn), f, S)
            best_info = (Q, R)

for name, ws in [("chỉ ridge", (1, 0, 0)), ("chỉ lgbm", (0, 1, 0)), ("chỉ knn", (0, 0, 1)), ("ridge+lgbm 0.5/0.5", (0.5, 0.5, 0))]:
    p_eval = blend(oof_ridge, oof, oof_knn, *ws)[mask]
    s = max(evaluate(p_eval, y_eval, keep_frac=f)[2] for f in keep_grid)
    print(f"{name:<22} score={s:.4f}")

(best_wr, best_wl, best_wk), best_keep, best_score = best
Q, R = best_info
print(f"\nTốt nhất: w_ridge={best_wr}, w_lgbm={best_wl}, w_knn={best_wk}, keep={best_keep}, "
      f"score={best_score:.4f}, Q={Q:.4f}, R={R:.4f}")

test_pred = blend(test_ridge, test_lgbm, test_knn, best_wr, best_wl, best_wk)
priority = make_priority(test_pred, best_keep)

submission = pd.DataFrame({"id": test_df["id"], "priority": priority})
submission.to_csv("submission.csv", index=False)
print(submission.head())
print("Number of recommended papers:", (submission["priority"] > 0).sum())
print("Number of excluded papers:", (submission["priority"] == 0).sum())