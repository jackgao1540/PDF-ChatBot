.PHONY: llama2

llama2:
	@echo "Setting up virtual environment and running scripts..."
	./llm-venv/bin/python promptModel.py $(filter-out $@,$(MAKECMDGOALS))