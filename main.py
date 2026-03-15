import os
import pickle
import time
from langchain_docling import DoclingLoader
from langchain_docling.loader import ExportType

data_folder_path = "./data"
cache_folder_path = "./cache"
def main():
    print("Loading documents...")

    chunks = []

    start_time = time.time()

    for subject in os.listdir(data_folder_path):
        subject_path = os.path.join(data_folder_path, subject)
        for slide in os.listdir(subject_path):
            if slide.endswith((".pdf", ".pptx")):
                slide_path = os.path.join(subject_path, slide)
                cache_slide_path = os.path.join(cache_folder_path, f"{subject}_{slide}.pkl")
                print(f"📄 Loading {slide}...")
                if os.path.exists(cache_slide_path):
                    print(f"⚡ [CACHE] Đọc siêu tốc file: {slide}")
                    with open(cache_slide_path, "rb") as f:
                        doc = pickle.load(f)
                else:
                    print(f"🐌 [DOCLING] Đang bóc tách file mới: {slide}...")
                    loader = DoclingLoader(file_path=slide_path, export_type=ExportType.MARKDOWN)
                    doc = loader.load()
                    with open(cache_slide_path, "wb") as f:
                        pickle.dump(doc, f)
                chunks.extend(doc)

    print(f"\n📊 Total chunks loaded: {len(chunks)}")
    for i, chunk in enumerate(chunks[:5]):
        print(f"\n--- Chunk {i+1} ---")
        print("Content:", chunk.page_content)

    end_time = time.time()
    print(f"\n⏱️ Total loading time: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    main()