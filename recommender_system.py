import json
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans


# ==============================
# 1. LOAD & FEATURE ENGINEERING
# ==============================

def load_and_engineer(csv_path: str = "videos_raw.csv"):
    df = pd.read_csv(csv_path)

    # ---- Basic cleaning ----
    df["description"] = df["description"].fillna("")
    # transcript column is empty in your current data; ignore for now
    # If later you add transcripts, you can bring them in here.

    # Combine title + description as base text
    base_text = df["title"].astype(str) + " " + df["description"].astype(str)

    # ---- Tags: parse JSON, count, and turn into text ----
    def parse_tags(x):
        try:
            return json.loads(x) if isinstance(x, str) else []
        except json.JSONDecodeError:
            return []

    df["tags_list"] = df["tags"].apply(parse_tags)
    df["num_tags"] = df["tags_list"].apply(len)
    tags_text = df["tags_list"].apply(lambda tags: " ".join(tags))

    # Combined text (title + description + tags)
    df["text"] = base_text + " " + tags_text

    # ---- Engagement features ----
    df["view_count"] = df["view_count"].clip(lower=1)  # avoid div by zero
    df["like_rate"] = df["like_count"] / df["view_count"]
    df["comment_rate"] = df["comment_count"] / df["view_count"]
    # simple engagement score; you can tweak weights
    df["engagement_score"] = df["like_rate"] + 2 * df["comment_rate"]
    df["log_views"] = np.log1p(df["view_count"])

    # ---- Duration buckets (style-ish) ----
    df["duration_bucket"] = pd.cut(
        df["duration_seconds"],
        bins=[0, 15, 30, 60],
        labels=["0-15s", "15-30s", "30-60s"],
        include_lowest=True,
    )

    return df


# ==============================
# 2. TOPIC MODEL (TF-IDF + KMEANS)
# ==============================

def build_topic_model(df: pd.DataFrame, n_clusters: int = 8):
    """
    Build a simple topic model with TF-IDF + KMeans.
    Returns: vectorizer, kmeans, updated df, cluster_terms, cluster_stats, creator_cluster
    """
    # TF-IDF on combined text
    vectorizer = TfidfVectorizer(max_features=3000, stop_words="english")
    X_text = vectorizer.fit_transform(df["text"])

    # KMeans clustering on text to get "topics"
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df["topic_cluster"] = kmeans.fit_predict(X_text)

    # ---- Human-readable terms per cluster ----
    terms = np.array(vectorizer.get_feature_names_out())

    def top_terms_for_cluster(cluster_idx, n_terms=10):
        # IMPORTANT: convert mask to numpy array before indexing sparse matrix
        mask = (df["topic_cluster"] == cluster_idx).values  # <-- fix here
        cluster_docs = X_text[mask]
        if cluster_docs.shape[0] == 0:
            return []
        mean_tfidf = cluster_docs.mean(axis=0).A1
        top_idx = mean_tfidf.argsort()[::-1][:n_terms]
        return terms[top_idx].tolist()

    cluster_terms = {c: top_terms_for_cluster(c) for c in range(n_clusters)}

    # ---- Global cluster stats (for trends) ----
    cluster_stats = (
        df.groupby("topic_cluster")
        .agg(
            n_videos=("video_id", "count"),
            mean_engagement=("engagement_score", "mean"),
            mean_views=("view_count", "mean"),
        )
        .reset_index()
    )

    # ---- Per-creator cluster stats (for personalization) ----
    creator_cluster = (
        df.groupby(["channel_id", "topic_cluster"])
        .agg(
            n_videos=("video_id", "count"),
            mean_engagement=("engagement_score", "mean"),
        )
        .reset_index()
    )

    return vectorizer, kmeans, df, cluster_terms, cluster_stats, creator_cluster


# ==============================
# 3. RECOMMENDER LOGIC
# ==============================

def build_creator_cluster_dict(creator_cluster: pd.DataFrame):
    """Convenience dict: (channel_id, cluster) -> stats"""
    creator_cluster_dict = {}
    for _, row in creator_cluster.iterrows():
        creator_cluster_dict[(row["channel_id"], int(row["topic_cluster"]))] = {
            "n_videos": int(row["n_videos"]),
            "mean_engagement": float(row["mean_engagement"]),
        }
    return creator_cluster_dict


def compute_recommendations_for_creator(
    df: pd.DataFrame,
    cluster_terms: dict,
    cluster_stats: pd.DataFrame,
    creator_cluster_dict: dict,
    channel_id: str,
    top_k_double: int = 3,
    top_k_new: int = 3,
):
    """
    Returns a dict with:
        - "double_down": clusters where creator is strong
        - "new_topics": clusters that are globally strong but unused by creator
    """

    creator_videos = df[df["channel_id"] == channel_id]
    if creator_videos.empty:
        raise ValueError(f"No videos found for channel_id={channel_id}")

    overall_mean_eng = df["engagement_score"].mean()
    overall_std_eng = df["engagement_score"].std(ddof=0) + 1e-9  # avoid div 0

    double_down_candidates = []
    new_topic_candidates = []

    for _, row in cluster_stats.iterrows():
        cluster = int(row["topic_cluster"])
        global_mean = row["mean_engagement"]
        global_norm = (global_mean - overall_mean_eng) / overall_std_eng

        key = (channel_id, cluster)
        creator_info = creator_cluster_dict.get(key, None)

        if creator_info is not None:
            c_n = creator_info["n_videos"]
            c_mean = creator_info["mean_engagement"]
            # how much better this creator does than global avg in that cluster
            creator_norm = (c_mean - global_mean) / overall_std_eng
            score_double = 0.6 * creator_norm + 0.4 * global_norm

            # only suggest "double down" if they have at least a couple of videos
            if c_n >= 2:
                double_down_candidates.append(
                    {
                        "cluster": cluster,
                        "score": score_double,
                        "creator_n_videos": c_n,
                        "creator_mean_eng": c_mean,
                        "global_mean_eng": global_mean,
                    }
                )
        else:
            # Creator has no videos in this cluster → potential new direction
            novelty = 1.0
            score_new = 0.7 * global_norm + 0.3 * novelty
            new_topic_candidates.append(
                {
                    "cluster": cluster,
                    "score": score_new,
                    "global_mean_eng": global_mean,
                }
            )

    double_down_sorted = sorted(
        double_down_candidates, key=lambda x: x["score"], reverse=True
    )[:top_k_double]
    new_topic_sorted = sorted(
        new_topic_candidates, key=lambda x: x["score"], reverse=True
    )[:top_k_new]

    # Helper: pick example videos for a cluster
    def example_videos_for_cluster(cluster, n_examples=3, within_creator=None):
        if within_creator is None:
            subset = df[df["topic_cluster"] == cluster]
        else:
            subset = df[
                (df["topic_cluster"] == cluster)
                & (df["channel_id"] == within_creator)
            ]
        subset = subset.sort_values("engagement_score", ascending=False).head(
            n_examples
        )
        return subset[
            [
                "video_id",
                "title",
                "view_count",
                "like_count",
                "comment_count",
                "engagement_score",
                "duration_bucket",
            ]
        ]

    recommendations = {"double_down": [], "new_topics": []}

    for item in double_down_sorted:
        c = item["cluster"]
        rec = {
            "cluster": c,
            "cluster_terms": cluster_terms.get(c, []),
            "score": item["score"],
            "creator_n_videos": item["creator_n_videos"],
            "creator_mean_eng": item["creator_mean_eng"],
            "global_mean_eng": item["global_mean_eng"],
            "examples_from_creator": example_videos_for_cluster(
                c, within_creator=channel_id
            ).to_dict(orient="records"),
            "top_global_examples": example_videos_for_cluster(
                c, within_creator=None
            ).to_dict(orient="records"),
        }
        recommendations["double_down"].append(rec)

    for item in new_topic_sorted:
        c = item["cluster"]
        rec = {
            "cluster": c,
            "cluster_terms": cluster_terms.get(c, []),
            "score": item["score"],
            "global_mean_eng": item["global_mean_eng"],
            "top_global_examples": example_videos_for_cluster(
                c, within_creator=None
            ).to_dict(orient="records"),
        }
        recommendations["new_topics"].append(rec)

    return recommendations


# ==============================
# 4. EXAMPLE USAGE
# ==============================

"""def main():
    # 1. Load data + engineer features
    df = load_and_engineer("videos_raw.csv")

    # 2. Build topic model + stats
    vectorizer, kmeans, df, cluster_terms, cluster_stats, creator_cluster = \
        build_topic_model(df, n_clusters=8)

    # 3. Build creator-cluster lookup
    creator_cluster_dict = build_creator_cluster_dict(creator_cluster)

    # 4. Pick a creator to test (one with most videos)
    top_channels = df["channel_id"].value_counts()
    example_channel = top_channels.index[0]
    print(f"Example channel_id: {example_channel}")

    recs = compute_recommendations_for_creator(
        df,
        cluster_terms,
        cluster_stats,
        creator_cluster_dict,
        channel_id=example_channel,
        top_k_double=3,
        top_k_new=3,
    )

    # 5. Pretty-print a summary
    print("\n=== DOUBLE DOWN TOPICS (you already do this well) ===\n")
    for r in recs["double_down"]:
        print(f"- Cluster {r['cluster']} | keywords: {', '.join(r['cluster_terms'][:8])}")
        print(f"  Your videos in this cluster: {r['creator_n_videos']}")
        print(f"  Your mean engagement: {r['creator_mean_eng']:.4f}")
        print(f"  Global mean engagement: {r['global_mean_eng']:.4f}")
        print("  Example videos from you:")
        for ex in r["examples_from_creator"]:
            print(f"    • {ex['title']} "
                  f"(views={ex['view_count']}, eng={ex['engagement_score']:.4f}, dur={ex['duration_bucket']})")
        print("  Top global examples in this topic:")
        for ex in r["top_global_examples"][:3]:
            print(f"    • {ex['title']} "
                  f"(views={ex['view_count']}, eng={ex['engagement_score']:.4f}, dur={ex['duration_bucket']})")
        print()

    print("\n=== NEW TOPICS TO EXPLORE (globally strong, you don't post here) ===\n")
    for r in recs["new_topics"]:
        print(f"- Cluster {r['cluster']} | keywords: {', '.join(r['cluster_terms'][:8])}")
        print(f"  Global mean engagement: {r['global_mean_eng']:.4f}")
        print("  Top global examples in this topic:")
        for ex in r["top_global_examples"][:3]:
            print(f"    • {ex['title']} "
                  f"(views={ex['view_count']}, eng={ex['engagement_score']:.4f}, dur={ex['duration_bucket']})")
        print()
"""
def main():
    # 1. Load data + engineer features
    df = load_and_engineer("videos_raw.csv")

    # 2. Build topic model + stats
    vectorizer, kmeans, df, cluster_terms, cluster_stats, creator_cluster = \
        build_topic_model(df, n_clusters=8)

    # 3. Build creator-cluster lookup
    creator_cluster_dict = build_creator_cluster_dict(creator_cluster)

    # 4. Pick multiple channels (by #videos)
    top_channels = df["channel_id"].value_counts().head(5)
    print("Top channels by video count:")
    print(top_channels)

    for channel_id in top_channels.index:
        print("\n" + "=" * 80)
        print(f"RECOMMENDATIONS FOR CHANNEL: {channel_id}")
        print("=" * 80)

        recs = compute_recommendations_for_creator(
            df,
            cluster_terms,
            cluster_stats,
            creator_cluster_dict,
            channel_id=channel_id,
            top_k_double=3,
            top_k_new=3,
        )

        print("\n=== DOUBLE DOWN TOPICS ===\n")
        for r in recs["double_down"]:
            print(f"- Cluster {r['cluster']} | keywords: {', '.join(r['cluster_terms'][:8])}")
            print(f"  Your videos in this cluster: {r['creator_n_videos']}")
            print(f"  Your mean engagement: {r['creator_mean_eng']:.4f}")
            print(f"  Global mean engagement: {r['global_mean_eng']:.4f}")
            print("  Example videos from you:")
            for ex in r["examples_from_creator"]:
                print(f"    • {ex['title']} "
                      f"(views={ex['view_count']}, eng={ex['engagement_score']:.4f}, dur={ex['duration_bucket']})")
            print()

        print("\n=== NEW TOPICS TO EXPLORE ===\n")
        for r in recs["new_topics"]:
            print(f"- Cluster {r['cluster']} | keywords: {', '.join(r['cluster_terms'][:8])}")
            print(f"  Global mean engagement: {r['global_mean_eng']:.4f}")
            print("  Top global examples in this topic:")
            for ex in r["top_global_examples"][:3]:
                print(f"    • {ex['title']} "
                      f"(views={ex['view_count']}, eng={ex['engagement_score']:.4f}, dur={ex['duration_bucket']})")
            print()


if __name__ == "__main__":
    main()

