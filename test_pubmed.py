from abstract_retriever import AbstractRetriever

db_file = "pubmed_abstracts_2024.db"
h5_file = "pubmed_embeddings.h5"

retriever = AbstractRetriever(h5_file, db_file, chunk_size=250000, use_cuda=True)

while True:

    # Example query embedding (replace with an actual embedding)
    query = input("Enter your query: ").strip()

    if query == "":
        query = "What is the role of GLP-1 and GLP-1 agonists in losing excess weight?"

    if query.lower() == "exit":
        break

    print(f"Looking up the PubMed abstracts using \"{query}\"...")

    query_embedding = retriever.embed_query(query)

    top_k = 5
    pmids, distances, documents = retriever.search(query, top_k)

    # sort documents according to distances (descending)
    distances, documents = zip(*sorted(zip(distances, documents), key=lambda x: x[0], reverse=True))

    for i, (abstract, similarity) in enumerate(zip(documents, distances)):
        print(f"Rank {i + 1}, Similarity: {similarity}")
        print(f"PMID: {abstract['pmid']}")
        print(f"Title: {abstract['title']}")
        print(f"Authors: {abstract['authors']}")
        print(f"Abstract: {abstract['abstract']}")
        print(f"Publication Year: {abstract['publication_year']}")
        print("-----")






