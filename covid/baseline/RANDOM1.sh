for k in 1
do
    python3 main.py --execute 'RANDOM' --gpu 0 --lr 0.00001 --K $k
done