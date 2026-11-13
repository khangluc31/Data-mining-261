import pandas as pd
import numpy as np
import torch
import lightgbm
import scipy.sparse as sp

from pathlib import Path
from transformers import AutoTokenizer
from adapters import AutoAdapterModel
from sklearn.neighbors import NearestNeighbors
from collections import Counter
from sklearn.model_selection import KFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.preprocessing import OneHotEncoder
from scipy.stats import rankdata
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
    mask=token["attention_mask"].unsqueeze(-1)
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


def to_rank(p):
    return rankdata(p) / len(p)


def blend(p_ridge, p_lgbm, w):
    return w * to_rank(p_ridge) + (1 - w) * to_rank(p_lgbm)


mask = (train_df["venue"] != "iclp").values
y_eval = y[mask]
keep_grid = np.round(np.arange(0.50, 0.851, 0.05), 2)


best = (None, None, -1)
best_info=(None, None)
w_list = np.linspace(0.0, 1.0, 101)
for w in w_list:
    p_eval = blend(oof_ridge, oof, w)[mask]
    for f in keep_grid:
        Q, R, S = evaluate(p_eval, y_eval, keep_frac=f)
        if S > best[2]:
            best = (w, f, S)
            best_info= (Q, R)
    w_best_f = max(keep_grid, key=lambda f: evaluate(p_eval, y_eval, keep_frac=f)[2])
    Q, R, S = evaluate(p_eval, y_eval, keep_frac=w_best_f)

best_w, best_keep, best_score = best
Q,R=best_info
print(f"\nTốt nhất: w_ridge={best_w}, keep={best_keep}, score={best_score:.4f}, Q={Q:.4f}, R={R:.4f}")

test_pred = blend(test_ridge, test_lgbm, best_w)
priority = make_priority(test_pred, best_keep)

submission = pd.DataFrame({"id": test_df["id"], "priority": priority})
submission.to_csv("submission.csv", index=False)
print(submission.head())
print("Number of recommended papers:", (submission["priority"] > 0).sum())
print("Number of excluded papers:", (submission["priority"] == 0).sum())