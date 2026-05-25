# NBA / MLB Monte Carlo Simulator

Streamlit app for simulating NBA and MLB matchups 10,000 times.

## Run Locally

```powershell
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Then open:

```text
http://localhost:8501
```

## Deploy On Streamlit Community Cloud

1. Push this repository to GitHub.
2. Go to https://share.streamlit.io/
3. Create a new app.
4. Select this repository: `lolochen86-jpg/NBA-MLB-AI`
5. Set main file path to:

```text
app.py
```

6. Deploy.

The app uses public sports data APIs and does not require secrets by default.
