api:
	uvicorn vectyfi_src.api.fast:app --reload &
	sleep 2 && open http://localhost:8000/docs
