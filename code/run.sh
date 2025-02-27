#!/usr/bin/env bash

name="${1,,}"

[ -z "$name"] &&
echo "Error: missing dataset name." &&
exit 1

[ ! -d "data/$name"] &&
echo "Error: missing 'data/$name' folder; please extract files first." &&
exit 1

mkdir -p logs/$name

# Obtain node features with Node2Vec, DynNode2Vec, tNodeEmbed.
./run.py node2vec --data $name \
                  --params ../params/node2vec.json \
                  --weights data/$name/x_node2vec

./run.py dynnode2vec --data $name \
                     --params ../params/node2vec.json \
                     --weights data/$name/x_dynnode2vec

./run.py tnodeembed --data $name \
                    --params ../params/node2vec.json \
                    --weights data/$name/x_tnodeembed

# Cluster the obtained node features with K-Means.
./run..py kmeans --data $name \
                 --features node2vec \
                 --params ../params/kmeans.json \
                 --log-file logs/$name/node2vec.log

./run..py kmeans --data $name \
                 --features dynnode2vec \
                 --params ../params/kmeans.json \
                 --log-file logs/$name/dynnode2vec.log

./run..py kmeans --data $name \
                 --features tnodeembed \
                 --params ../params/kmeans.json \
                 --log-file logs/$name/tnodeembed.log

# Train DAEGC and TGC w. pretrained features from Node2Vec, cluster w. K-Means.
./run.py daegc --data $name \
               --features node2vec \
               --params ../params/daegc.json \
               --log-file logs/$name/daegc.log

./run.py tgc --data $name \
             --features node2vec \
             --params ../params/tgc.json \
             --log-file logs/$name/tgc.log

# Run spectral clustering algorithm on graph Laplacian.
./run.py spectral --data $name \
                  --log-file logs/$name/spectral.log

# Run Leiden algorithm with modularity optimization.
./run.py leiden --data $name \
                --log-file logs/$name/leiden.log

# Run K-Means on node features from the original dataset (e.g., PubMed).
[ -f "data/$name/x.py" &&
./run.py kmeans --data $name \
                --params ../params/kmeans.json \
                --log-file logs/$name/kmeans.log
