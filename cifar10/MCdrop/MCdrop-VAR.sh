for k in 1 2 3
do
    python3 main2.py --execute 'MCdrop-VAR' --gpu 2 --K $k
done
