api:
	uvicorn vectyfi_src.api.fast:app &
	sleep 2 && open http://localhost:8000/docs

stop:
	kill $(lsof -t -i:8000)
