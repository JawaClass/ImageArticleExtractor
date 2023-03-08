import pdfbox

p = pdfbox.PDFBox()

file = r'C:\Users\masteroflich\Desktop\HELLO.pdf'

text = p.extract_text(file)  # writes text to /path/to/my_file.txt
#p.pdf_to_images(file)  # writes images to /path/to/my_file1.jpg, /path/to/my_file2.jpg, etc.
#p.extract_images(file)  # writes images to /path/to/my_file-1.png, /path/to/my_file-2.png, etc.

print('TEXT')
print(text)