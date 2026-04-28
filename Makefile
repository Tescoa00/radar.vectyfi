api_local:
	uvicorn vectyfi_src.api.fast:app --reload &
	sleep 2 && open http://localhost:8000/docs

stop:
	kill $(lsof -t -i:8000)

api:
	open https://vectyfi-api-828368828432.europe-west1.run.app/docs

local:
	-@streamlit run vectyfi_src/frontend/app.py

app:
	open https://radarvectyfi-tenderpredicter.streamlit.app/
