import pandas as pd
import seaborn as sns
from sklearn.manifold import MDS
from matplotlib import pyplot as plt
import numpy as np

def show_distance_heatmap(distances, docs):
    distance_matrix = pd.DataFrame(index=docs, columns=docs, dtype=float)

    # Fill matrix with given distances
    for (a1, a2), dist in distances.items():
        distance_matrix.loc[a1, a2] = dist
        distance_matrix.loc[a2, a1] = dist  # Symmetric
    # Plot heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(distance_matrix, cmap="coolwarm", annot=True, fmt=".2f", annot_kws={'fontsize':6})
    plt.title("Heatmap of Pairwise Distances")
    plt.show()

def show_author_mds_embedding(distances, authors):
    distance_matrix = pd.DataFrame(index=authors, columns=authors, dtype=float)

    for (a1, a2), dist in distances.items():
        distance_matrix.loc[a1, a2] = dist
        distance_matrix.loc[a2, a1] = dist  # Symmetric


    np.fill_diagonal(distance_matrix.values, 0)

    mds = MDS(n_components=2, dissimilarity="precomputed")
    coords = mds.fit_transform(distance_matrix)

    mds_df = pd.DataFrame(coords, index=authors, columns=["x", "y"])

    plt.figure(figsize=(8, 6))
    plt.scatter(mds_df["x"], mds_df["y"], color="red")

    for author, (x, y) in mds_df.iterrows():
        plt.text(x, y, author, fontsize=12, ha='right', va='bottom')

    plt.title("MDS Representation of Author Distances")
    plt.xlabel("MDS Dimension 1")
    plt.ylabel("MDS Dimension 2")
    plt.grid(True)
    plt.show()


def show_book_mds_embedding(distances):
    author_dict = {
    "mt": "matthew", "mk": "mark", "lk": "luke", "j": "john", "acts": "luke",
    "rom": "paul", "1kor": "paul", "2kor": "paul", "gal": "paul", "eph": "paul", 
    "phi": "paul", "col": "paul", "1tes": "paul", "2tes": "paul", "1tym": "paul", 
    "2tym": "paul", "tit": "paul", "fil": "paul", "heb": "unknown", "jam": "james", 
    "1pet": "peter", "2pet": "peter", "1j": "john", "2j": "john", "3j": "john", 
    "jd": "jude", "rev": "john"
    }

    books = list(author_dict.keys())

    distance_matrix = pd.DataFrame(index=books, columns=books, dtype=float)
    for (a1, a2), dist in distances.items():
        distance_matrix.loc[a1, a2] = dist
        distance_matrix.loc[a2, a1] = dist  # Ensure symmetry
    np.fill_diagonal(distance_matrix.values, 0)

    mds = MDS(n_components=2, dissimilarity="precomputed", random_state=42)
    coords = mds.fit_transform(distance_matrix)

    mds_df = pd.DataFrame(coords, index=books, columns=["x", "y"])
    mds_df["author"] = mds_df.index.map(author_dict)  # Assign authors

    unique_authors = mds_df["author"].unique()
    palette = sns.color_palette("husl", len(unique_authors))  # Generate distinct colors
    color_map = dict(zip(unique_authors, palette))  # Map authors to colors
    mds_df["color"] = mds_df["author"].map(color_map)  # Assign colors to points

    plt.figure(figsize=(10, 7))
    for author in unique_authors:
        subset = mds_df[mds_df["author"] == author]
        plt.scatter(subset["x"], subset["y"], color=color_map[author], label=author, alpha=0.8,edgecolors="black", linewidth=1, s=100)

    for doc, (x, y) in mds_df[["x", "y"]].iterrows():
        plt.text(x, y, doc, fontsize=10, ha='right', va='bottom')

    plt.title("MDS Representation of Author Distances")
    plt.xlabel("MDS Dimension 1")
    plt.ylabel("MDS Dimension 2")
    plt.legend(title="Author", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.show()
