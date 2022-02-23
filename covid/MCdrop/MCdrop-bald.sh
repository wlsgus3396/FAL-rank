for k in 1
do
    python3 main2.py --execute 'MCdrop-bald' --gpu 1 --K $k
done

for k in 1
do
    python3 main.py --execute 'F-MCdrop-bald' --gpu 1 --K $k
done