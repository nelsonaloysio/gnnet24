#!/usr/bin/env bash

ROOT="$(dirname $(realpath "$0"))"
USAGE="usage: $(basename "$0") [-h] [--retrain] {$(ls -1 "$ROOT/data" | tr '\n' ',' | sed 's/,$//')}"
export PYTHONPATH="$ROOT/code"

# Parse command line arguments.
while :; do
    case $1 in
        -h|--help)
            echo -e $USAGE && exit 0
            ;;
        *)
            [ -n "$1" ] && names="$names $1" && shift || break
            ;;
    esac
done

# Display usage if no dataset is provided.
[ -z "$names" ] && echo -e $USAGE && exit 1

# Run experiments for each dataset.
for name in $names; do
    [ -d "data/$name" ] || { echo "Dataset '$name' not found in '$ROOT/data'." && exit 1; }

    echo "Running experiments for dataset '$name'..."
    mkdir -p logs/$name

    ## Train SkipGram-based models.
    # echo "Training Node2Vec model..."
    # python -m gnnet24 node2vec $name \
    #                 --log-level notset \
    #                 --params params/node2vec.json \
    #                 --output "$ROOT/data/$name/node2vec/x.npz"
    # echo "Training DynNode2Vec model..."
    # python -m gnnet24 dynnode2vec $name \
    #                 --log-level notset \
    #                 --params params/node2vec.json \
    #                 --output "$ROOT/data/$name/dynnode2vec/x.npz"
    # echo "Training tNodeEmbed model..."
    # python -m gnnet24 tnodeembed $name \
    #                 --log-level notset \
    #                 --params params/node2vec.json \
    #                 --output "$ROOT/data/$name/tnodeembed/x.npz"

    echo "Running spectral clustering..."
    python -m gnnet24 spectral $name \
                        --log-file logs/$name/spectral.log

    echo "Running Leiden algorithm with modularity optimization..."
    python -m gnnet24 leiden $name \
                        --features node2vec \
                        --log-file logs/$name/leiden.log

    [ -f "data/$name/x.npy" ] &&
    echo "Clustering original node features with K-Means..." &&
    python -m gnnet24 kmeans $name \
                        --params params/kmeans.json \
                        --log-file logs/$name/kmeans.log

    [ -f "data/$name/node2vec/x.npy" ] &&
    echo "Clustering Node2Vec features with K-Means..." &&
    python -m gnnet24 kmeans $name \
                        --features node2vec \
                        --params params/kmeans.json \
                        --log-file logs/$name/node2vec.log

    [ -f "data/$name/dynnode2vec/x.npy" ] &&
    echo "Clustering DynNode2Vec features with K-Means..." &&
    python -m gnnet24 kmeans $name \
                        --features dynnode2vec \
                        --params params/kmeans.json \
                        --log-file logs/$name/dynnode2vec.log

    [ -f "data/$name/tnodeembed/x.npy" ] &&
    echo "Clustering tNodeEmbed features with K-Means..." &&
    python -m gnnet24 kmeans $name \
                        --features tnodeembed \
                        --params params/kmeans.json \
                        --log-file logs/$name/tnodeembed.log

    echo "Training DAEGC model..."
    python -m gnnet24 daegc $name \
                        --features node2vec \
                        --params params/daegc.json \
                        --log-file logs/$name/daegc.log

    # echo "Training TGC model..."
    # python -m gnnet24 tgc $name \
    #                     --features node2vec \
    #                     --params params/tgc.json \
    #                     --log-file logs/$name/tgc.log

done
