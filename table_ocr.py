from img2table.document import Image
from img2table.ocr import TesseractOCR
import pytesseract

pytesseract.pytesseract.tesseract_cmd = r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"

ocr = TesseractOCR(n_threads=1, lang="eng")
src = "arj.png"
doc = Image(src)
import cv2
cv2.imshow("image", doc.images[0])

extracted_tables = doc.extract_tables(ocr=ocr)

print(extracted_tables)