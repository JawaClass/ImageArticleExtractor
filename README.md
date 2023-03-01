# ImageArticleExtractor
extract ofml article info from pdf



dependencies:
 
run:
conda create --name ImageArticleExtractor
conda activate ImageArticleExtractor
conda env update --file environment.yaml

install this
- https://github.com/tesseract-ocr/tesseract


TODO:
- Testing:
  create sample data (100-500 PDF documents) and extract the important information per hand
  - articles with configuration
  - amount per article
  - extract order number
  - extract other customer data
  and create statistics based on comparison with result from this program
