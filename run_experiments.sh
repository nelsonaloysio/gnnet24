#!/usr/bin/env bash

ROOT="$(dirname $(realpath "$0"))"
USAGE="usage: $(basename "$0") [-h] [--retrain] {$(ls -1 "$ROOT/data" | tr '\n' ',' | sed 's/,$//')}"
export PYTHONPATH="$ROOT/code"

while :; do
    case $1 in
        -h|--help)
            echo -e $USAGE
            exit 0
            ;;
        --retrain)
            RETRAIN=1
            shift
            ;;
        *)
            [ -n "$1" ] &&
            names="$names $1" &&
            shift ||
            break
            ;;
    esac
done

[ -z "$names" ] && echo -e $USAGE && exit 1

for name in $names; do
    [ -d "data/$name" ] || { echo "Dataset '$name' not found in '$ROOT/data'." && exit 1; }

    echo "Running experiments on dataset '$name'..."
    mkdir -p logs/$name

    # Train SkipGram-based models.
    if [ "$RETRAIN" = 1 ]; then
        echo "Training Node2Vec model..."
        python -m gnnet24 node2vec $name \
               --params params/node2vec.json \
               --output "$ROOT/data/$name/raw/node2vec/x.npz"

        echo "Training DynNode2Vec model..."
        python -m gnnet24 dynnode2vec $name \
               --params params/node2vec.json \
               --output "$ROOT/data/$name/raw/dynnode2vec/x.npz"

        echo "Training tNodeEmbed model..."
        python -m gnnet24 tnodeembed $name \
               --params params/node2vec.json \
               --output "$ROOT/data/$name/raw/tnodeembed/x.npz"
    fi

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
           --log-file logs/$name/kmeans.log \
           --params params/kmeans.json

    [ -f "data/$name/node2vec/x.npy" ] &&
    echo "Clustering Node2Vec features with K-Means..." &&
    python -m gnnet24 kmeans $name \
           --features node2vec \
           --log-file logs/$name/node2vec.log \
           --params params/kmeans.json

    [ -f "data/$name/dynnode2vec/x.npy" ] &&
    echo "Clustering DynNode2Vec features with K-Means..." &&
    python -m gnnet24 kmeans $name \
           --features dynnode2vec \
           --log-file logs/$name/dynnode2vec.log \
           --params params/kmeans.json

    [ -f "data/$name/tnodeembed/x.npy" ] &&
    echo "Clustering tNodeEmbed features with K-Means..." &&
    python -m gnnet24 kmeans $name \
           --features tnodeembed \
           --log-file logs/$name/tnodeembed.log \
           --params params/kmeans.json

    echo "Training DAEGC model..."
    python -m gnnet24 daegc $name \
           --features node2vec \
           --log-file logs/$name/daegc.log \
           --params params/daegc.json

    # echo "Training TGC model..."
    # python -m gnnet24 tgc $name \
   #                   --features node2vec \
   #                   --params params/tgc.json \
    #                   --log-file logs/$name/tgc.log
done
