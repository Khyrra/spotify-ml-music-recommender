import os
import time
import base64
import math
import random
import requests
from dotenv import load_dotenv
from flask import Flask, render_template, request

import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor

load_dotenv()
app = Flask(__name__)

SPOTIFY_CLIENT_ID = os.getenv("SPOTIFY_CLIENT_ID", "").strip()
SPOTIFY_CLIENT_SECRET = os.getenv("SPOTIFY_CLIENT_SECRET", "").strip()

_token_cache = {"access_token": None, "expires_at": 0}


# ---------------- Spotify Auth ----------------
def get_access_token():
    now = int(time.time())
    if _token_cache["access_token"] and now < _token_cache["expires_at"] - 30:
        return _token_cache["access_token"]

    if not SPOTIFY_CLIENT_ID or not SPOTIFY_CLIENT_SECRET:
        return None

    auth = base64.b64encode(
        f"{SPOTIFY_CLIENT_ID}:{SPOTIFY_CLIENT_SECRET}".encode("utf-8")
    ).decode("utf-8")

    headers = {"Authorization": f"Basic {auth}"}
    data = {"grant_type": "client_credentials"}

    r = requests.post(
        "https://accounts.spotify.com/api/token",
        headers=headers,
        data=data,
        timeout=15
    )
    r.raise_for_status()
    payload = r.json()

    token = payload["access_token"]
    expires_in = int(payload.get("expires_in", 3600))

    _token_cache["access_token"] = token
    _token_cache["expires_at"] = now + expires_in
    return token


def spotify_get(url, params=None):
    token = get_access_token()
    if not token:
        return None, "Spotify Client ID/Secret bulunamadı. .env ayarla."

    headers = {"Authorization": f"Bearer {token}"}
    r = requests.get(url, headers=headers, params=params, timeout=15)

    # token expired
    if r.status_code == 401:
        _token_cache["access_token"] = None
        token = get_access_token()
        headers = {"Authorization": f"Bearer {token}"}
        r = requests.get(url, headers=headers, params=params, timeout=15)

    if r.status_code >= 400:
        return None, f"Spotify API hata: {r.status_code} - {r.text[:220]}"
    return r.json(), None


# ---------------- Data Fetch ----------------
def spotify_search_tracks(query: str, limit: int = 50, market: str = "TR"):
    """
    Spotify Search limit maksimum 50. 50 üstü -> 400 Invalid limit.
    """
    limit = max(1, min(int(limit), 50))

    data, err = spotify_get(
        "https://api.spotify.com/v1/search",
        params={"q": query, "type": "track", "limit": limit, "market": market}
    )
    if err:
        return [], err

    items = (data.get("tracks") or {}).get("items") or []
    tracks = []
    for it in items:
        artists = ", ".join([a.get("name", "") for a in it.get("artists", [])])
        album = ((it.get("album") or {}).get("name") or "")
        images = (it.get("album") or {}).get("images") or []
        image = images[0].get("url") if images else None

        tracks.append({
            "id": it.get("id"),
            "name": it.get("name"),
            "artists": artists,
            "album": album,
            "image": image,
            "spotify_url": (it.get("external_urls") or {}).get("spotify"),
            "preview_url": it.get("preview_url"),
            "popularity": int(it.get("popularity") or 0),

            # Metin tabanlı özellikler: isim + sanatçı + albüm
            "features_text": f"{it.get('name','')} {artists} {album}".lower().strip(),
        })

    return tracks, None


# ---------------- Helpers ----------------
def normalize_scores_to_percent(scores):
    """
    Her algoritmanın çıktısı farklı ölçeklerde olabilir.
    UI tablosu için 0-100 normalize ediyoruz.
    """
    if not scores:
        return []
    mn, mx = float(min(scores)), float(max(scores))
    if abs(mx - mn) < 1e-12:
        return [100.0 for _ in scores]
    return [((float(s) - mn) / (mx - mn)) * 100.0 for s in scores]


def safe_topk(order, picked_idx, k):
    order = [i for i in order if i != picked_idx]
    return order[:k]


# ---------------- Core Feature Building ----------------
def build_tfidf_matrix(tracks):
    texts = [t["features_text"] for t in tracks]
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
    X = vec.fit_transform(texts)
    return X


def build_svd_embedding(X_tfidf, n_components=60, seed=42):
    """
    TF-IDF -> SVD (LSA). Dense embedding üretir.
    """
    n = X_tfidf.shape[0]
    comps = min(int(n_components), max(2, n - 1))
    svd = TruncatedSVD(n_components=comps, random_state=seed)
    Z = svd.fit_transform(X_tfidf)
    return Z


def cosine_sim_dense(Z, idx):
    a = Z[idx]
    denom = (np.linalg.norm(Z, axis=1) * (np.linalg.norm(a) + 1e-12))
    sims = (Z @ a) / (denom + 1e-12)
    return sims


# ---------------- Algorithms: Recommenders ----------------
def rec_tfidf(tracks, picked_idx, k):
    X = build_tfidf_matrix(tracks)
    sims = cosine_similarity(X[picked_idx], X).flatten()
    order = np.argsort(-sims)
    order = safe_topk(order.tolist(), picked_idx, k)
    return order, sims


def rec_knn(tracks, picked_idx, k):
    X = build_tfidf_matrix(tracks)
    nn = NearestNeighbors(n_neighbors=min(k + 1, X.shape[0]), metric="cosine")
    nn.fit(X)

    dists, idxs = nn.kneighbors(X[picked_idx], n_neighbors=min(k + 1, X.shape[0]))
    idxs = idxs.flatten().tolist()
    dists = dists.flatten().tolist()

    # cosine distance -> similarity
    pairs = [(i, 1.0 - float(d)) for i, d in zip(idxs, dists) if i != picked_idx]
    pairs = pairs[:k]

    sims = np.zeros(len(tracks), dtype=float)
    order = []
    for i, s in pairs:
        sims[i] = s
        order.append(i)

    return order, sims


def rec_svd(tracks, picked_idx, k):
    X = build_tfidf_matrix(tracks)
    Z = build_svd_embedding(X, n_components=60, seed=42)
    sims = cosine_sim_dense(Z, picked_idx)
    order = np.argsort(-sims)
    order = safe_topk(order.tolist(), picked_idx, k)
    return order, sims


# ---------------- ML Models (LR / RF / NN) without Spotify audio ----------------
def make_pair_features(Z, q_idx, i_idx):
    """
    Özellik vektörü:
    - |q - x| (fark)
    - x (aday)
    - q (sorgu)
    """
    q = Z[q_idx]
    x = Z[i_idx]
    return np.concatenate([np.abs(q - x), x, q])


def train_similarity_model(model, Z, q_idx, target_sims, n_samples=1200, seed=42):
    """
    Hedef/etiket:
    - target_sims: q ile adaylar arası similarity (0..1 gibi)
    Burada hedefi SVD uzayındaki cosine similarity olarak alıyoruz.
    (Tamamen veri içinden, Spotify’a bağımlı değil.)
    """
    rnd = random.Random(seed)
    n = Z.shape[0]

    cand = [i for i in range(n) if i != q_idx]
    if len(cand) < 6:
        return None, "Veri az olduğu için model eğitimi zayıf (çok az şarkı geldi)."

    Xtr, ytr = [], []
    for _ in range(min(n_samples, len(cand) * 6)):
        i = rnd.choice(cand)
        Xtr.append(make_pair_features(Z, q_idx, i))
        ytr.append(float(target_sims[i]))

    Xtr = np.array(Xtr, dtype=float)
    ytr = np.array(ytr, dtype=float)

    model.fit(Xtr, ytr)

    # tüm adaylar için skor tahmini
    feats_all = []
    for i in range(n):
        feats_all.append(make_pair_features(Z, q_idx, i))
    feats_all = np.array(feats_all, dtype=float)

    pred = model.predict(feats_all)
    pred = np.clip(pred, 0.0, 1.0)
    pred[q_idx] = -1.0
    return pred, None


def rec_linear(tracks, picked_idx, k):
    X = build_tfidf_matrix(tracks)
    Z = build_svd_embedding(X, n_components=60, seed=42)

    # hedef: SVD cosine similarity
    target = cosine_sim_dense(Z, picked_idx)
    target = np.clip(target, 0.0, 1.0)

    model = make_pipeline(StandardScaler(), LinearRegression())
    pred, err = train_similarity_model(model, Z, picked_idx, target, n_samples=1200, seed=42)
    if err:
        return [], None, err

    order = np.argsort(-pred).tolist()
    order = safe_topk(order, picked_idx, k)
    return order, pred, None


def rec_rf(tracks, picked_idx, k):
    X = build_tfidf_matrix(tracks)
    Z = build_svd_embedding(X, n_components=60, seed=42)

    target = cosine_sim_dense(Z, picked_idx)
    target = np.clip(target, 0.0, 1.0)

    rf = RandomForestRegressor(
        n_estimators=220,
        random_state=42,
        max_depth=12,
        n_jobs=-1
    )
    pred, err = train_similarity_model(rf, Z, picked_idx, target, n_samples=1200, seed=42)
    if err:
        return [], None, err

    order = np.argsort(-pred).tolist()
    order = safe_topk(order, picked_idx, k)
    return order, pred, None


def rec_nn(tracks, picked_idx, k):
    X = build_tfidf_matrix(tracks)
    Z = build_svd_embedding(X, n_components=60, seed=42)

    target = cosine_sim_dense(Z, picked_idx)
    target = np.clip(target, 0.0, 1.0)

    nn = make_pipeline(
        StandardScaler(),
        MLPRegressor(
            hidden_layer_sizes=(80, 40),
            activation="relu",
            random_state=42,
            max_iter=900
        )
    )
    pred, err = train_similarity_model(nn, Z, picked_idx, target, n_samples=1200, seed=42)
    if err:
        return [], None, err

    order = np.argsort(-pred).tolist()
    order = safe_topk(order, picked_idx, k)
    return order, pred, None


# ---------------- UI ----------------
ALGO_LABELS = {
    "tfidf": "TF-IDF (Cosine Similarity)",
    "knn": "K-En Yakın Komşu (KNN)",
    "svd": "SVD (LSA / TruncatedSVD)",
    "linear": "Lineer Regresyon",
    "rf": "Rastgele Orman (Random Forest)",
    "nn": "Sinir Ağı (MLP)"
}


@app.route("/", methods=["GET", "POST"])
def index():
    query = ""
    algo = "tfidf"
    k = 10

    error = None
    warn = None

    picked = None
    recs = []
    table = []

    if request.method == "POST":
        query = (request.form.get("song") or "").strip()
        algo = (request.form.get("algo") or "tfidf").strip()

        try:
            k = int(request.form.get("k") or 10)
            k = max(3, min(20, k))
        except Exception:
            k = 10

        if not query:
            error = "Boş arama."
        else:
            tracks, err = spotify_search_tracks(query, limit=50, market="TR")
            if err:
                error = err
            elif not tracks:
                error = "Sonuç bulunamadı."
            else:
                picked_idx = 0
                picked = tracks[picked_idx]

                raw_scores = []
                order = []

                if algo == "tfidf":
                    order, sims = rec_tfidf(tracks, picked_idx, k)
                    raw_scores = [float(sims[i]) for i in order]

                elif algo == "knn":
                    order, sims = rec_knn(tracks, picked_idx, k)
                    raw_scores = [float(sims[i]) for i in order]

                elif algo == "svd":
                    order, sims = rec_svd(tracks, picked_idx, k)
                    raw_scores = [float(sims[i]) for i in order]

                elif algo == "linear":
                    order, pred, e = rec_linear(tracks, picked_idx, k)
                    if e:
                        warn = e
                    raw_scores = [float(pred[i]) for i in order] if pred is not None else []

                elif algo == "rf":
                    order, pred, e = rec_rf(tracks, picked_idx, k)
                    if e:
                        warn = e
                    raw_scores = [float(pred[i]) for i in order] if pred is not None else []

                elif algo == "nn":
                    order, pred, e = rec_nn(tracks, picked_idx, k)
                    if e:
                        warn = e
                    raw_scores = [float(pred[i]) for i in order] if pred is not None else []

                else:
                    algo = "tfidf"
                    order, sims = rec_tfidf(tracks, picked_idx, k)
                    raw_scores = [float(sims[i]) for i in order]

                perc = normalize_scores_to_percent(raw_scores)

                for rank, (idx, p) in enumerate(zip(order, perc), start=1):
                    t = tracks[idx]
                    recs.append(t)
                    table.append({
                        "rank": rank,
                        "name": t["name"],
                        "artists": t["artists"],
                        "score_pct": round(float(p), 1),
                    })

                # hocaya savunma notu: spotify audio yok
                if warn is None:
                    warn = "Not: Spotify audio-features endpoint erişimi kısıtlı olabildiği için (403), tüm algoritmalar metin tabanlı özellikler (TF-IDF/SVD) üzerinden çalışır."

    return render_template(
        "index.html",
        query=query,
        algo=algo,
        k=k,
        algo_labels=ALGO_LABELS,
        picked=picked,
        recs=recs,
        table=table,
        error=error,
        warn=warn
    )


if __name__ == "__main__":
    app.run(debug=True)
