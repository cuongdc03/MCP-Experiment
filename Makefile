clean:
	$(call show_header, "Cleaning Source code...")
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".ipynb_checkpoints" -exec rm -rf {} +
	find . -type f -name ".DS_Store" -delete
	rm -rf temp/*

