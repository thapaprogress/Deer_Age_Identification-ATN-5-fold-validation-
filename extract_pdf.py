import PyPDF2
import sys

pdf_path = r'c:\Users\PRAJNA WORLD TECH\OneDrive\Desktop\atn\deer data\Augmented_Triplet_Network_for_Individual_Organism_and_Unique_Object_Classification_for_Reliable_Monitoring_of_Ezoshika_Deer.pdf'

all_text = ""

with open(pdf_path, 'rb') as pdf_file:
    reader = PyPDF2.PdfReader(pdf_file)
    print(f'Total pages: {len(reader.pages)}')
    
    for i in range(len(reader.pages)):
        text = reader.pages[i].extract_text()
        all_text += f'\n{"="*80}\nPAGE {i+1}\n{"="*80}\n\n{text}\n'

# Write all text to a single file
with open('pdf_content.txt', 'w', encoding='utf-8') as f:
    f.write(all_text)

print("PDF content extracted to pdf_content.txt")
