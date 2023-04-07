"""
Use Student’s t-distribution to measure the pairwise similarities in the given
ref: Learning Embedding Space for Clustering From Deep Representations
"""
function pairwiseSimilar(X,α)
a1 = pairwise(Euclidean(),X)
b1 = sum(a1,dims=1)

(1 .+ a1 .^ 2 / α) .^ (-(α + 1)/2) /
    (sum( (1 .+ a1 ./ α ) .^ (-(α + 1)/2)))
end
