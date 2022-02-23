for k in 1 2 3
do
    python3 main2.py --execute 'uncertainty' --gpu 2 --K $k
done
