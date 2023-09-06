from pdfminer.high_level import extract_text

def extract_paragraphs_from_pdf(pdf_path):
    txt=extract_text(pdf_path)
    paragraphs=[p for p in txt.split("\n\n") if p.strip()]
    return paragraphs

def save_paragraphs_to_txt(paragraphs, output_path):
    with open(output_path, 'w', encoding='utf-8') as f:
        for p in paragraphs:
            if len(p) > 1:
                f.write(p + "\n\n")

pdf_paths=["HP.pdf"]

for idx, pdf_path in enumerate(pdf_paths, 1):
    paragraphs = extract_paragraphs_from_pdf(pdf_path)
    for p in paragraphs:
        print(p)
        print('-' * 50)

    output_file_name = f"data{idx}.txt"
    save_paragraphs_to_txt(paragraphs, output_file_name)
    print(f"Saved paragraphs from {pdf_path} to {output_file_name}")