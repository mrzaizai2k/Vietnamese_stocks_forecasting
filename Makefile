install:
	pip install -r setup.txt

fetch_DB:
	python src/database/create_fake_DB.py --stocks MWG DGC FPT --start_date 2021-01-01 --end_date 2021-07-18
