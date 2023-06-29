python3 -m cProfile -o output.pstats elementwise_mul.py
# python -m pip install snakeviz
snakeviz output.pstats
