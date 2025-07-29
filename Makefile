all:
	maturin develop --release
	pip install .
