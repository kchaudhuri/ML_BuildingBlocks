
"""
Similarity and distance metrics
"""

class Metrics():
    def euclidean_distance(self, X, Y):
        return (sum([(x-y)**2 for x, y in zip(X, Y)])) ** (1/2)

    def manhattan_distance(self, X, Y):
        return sum(abs(x-y) for x, y in zip(X, Y))

    def cosine_similarity(self, X, Y):
        numerator = sum(x*y for x, y in zip(X, Y))
        denominator = (sum(x**2 for x in X) ** (1/2)) * (sum(y**2 for y in Y) ** (1/2))
        return numerator/denominator

    def jaccard_similarity(self, X, Y):
        return len(set.intersection(set(X), set(Y)))/len(set.union(set(X), set(Y)))
       


def distances_and_similarities(X, Y):
    metrics = Metrics()
    return [metrics.euclidean_distance(X, Y),
            metrics.manhattan_distance(X, Y),
            metrics.cosine_similarity(X, Y),
            metrics.jaccard_similarity(X, Y)]