.PHONY: test backtest run-api run-web paper-trade clean help

help:
	@echo "NSE Systematic 5-Minute Algorithmic Trading Platform"
	@echo "Available commands:"
	@echo "  make test         - Run full pytest test suite"
	@echo "  make backtest     - Execute systematic historical backtest"
	@echo "  make run-api      - Launch FastAPI backend server (port 8000)"
	@echo "  make run-web      - Launch Streamlit operations dashboard (port 8501)"
	@echo "  make paper-trade  - Run continuous 5-minute paper trading daemon"
	@echo "  make clean        - Remove Python cache files"

test:
	pytest tests/ -v

backtest:
	python scripts/run_backtest.py

run-api:
	uvicorn apps.api.src.main:app --reload --port 8000

run-web:
	streamlit run frontend/app.py --server.port 8501

paper-trade:
	python scripts/paper_trade.py

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
