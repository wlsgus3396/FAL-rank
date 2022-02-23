for k in 1 2 3
do
    python3 main.py --execute 'F-uncertainty' --gpu 1 --K $k
done

