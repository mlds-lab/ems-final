
# data_file=$1
# edd_dir=$2

imputerset=(mean SVD SVD SVD SVD SVD KNN KNN KNN KNN KNN)
rankset=(0 10 15 20 25 30 10 15 20 25 30)

seedset=(0 10 20 30 40 50 60 70 80 90 10 11 12 13 14 15 16 17 18 19)

results_file_name=imp_seed_results.pkl

for i in 0 1 2 3 4 5 6 7 8 9 10; do
    echo $(date) ' starting ' ${rank[i]}
    imputer=${imputerset[i]}
    rank=${rankset[i]}
    echo $imputer
    echo $rank
    for j in 0 1 2 3 4 5 6 7 8 9 10; do
        seed=${seedset[j]}
        echo $seed
        python3.6 ee7_imp_seed.py --data-file ./master_summary_dataframe-20180514-190154.pkl --edd-dir ./data/experiment_definitions_300/daily/ --edd-name cwb.d.json --imputer $imputer --rank $rank --random-seed $seed --results $results_file_name
    done
done