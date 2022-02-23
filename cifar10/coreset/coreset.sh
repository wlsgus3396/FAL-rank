for k in 1 2 3
do
    python3 main2.py --execute 'coreset' --gpu 0 --K $k
done