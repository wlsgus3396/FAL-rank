for k in 1
do
    python3 main.py --execute 'RANDOM' --gpu 1 --lr 0.000001 --K $k
done