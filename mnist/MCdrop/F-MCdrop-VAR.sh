for k in 1 2 3
do
    python3 main.py --execute 'F-MCdrop-VAR' --gpu 3 --K $k
done