.PHONY: setup run sanity clean

setup:
	python3 -m pip install -r requirements.txt

run:
	python3 -m streamlit run app/main.py

sanity:
	python3 -m app.sanity

clean:
	rm -rf chroma_db data/chromadb artifacts/sanity_output.json
