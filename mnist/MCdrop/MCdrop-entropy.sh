for k in 1 2 3
do
    python3 main2.py --execute 'MCdrop-entropy' --gpu 3 --K $k
done
