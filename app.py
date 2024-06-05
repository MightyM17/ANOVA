import streamlit as st
from anova_steps import anova_steps_cbd, anova_steps_rbd
import pandas as pd
import re

def ocr(file):
    import pytesseract
    from img2table.document import Image
    from img2table.ocr import TesseractOCR
    import pytesseract

    pytesseract.pytesseract.tesseract_cmd = r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"

    # Read the image file
    doc = Image(file)

    ocr = TesseractOCR(n_threads=4, lang="eng")
    extracted_tables = doc.extract_tables(ocr=ocr,
                                      implicit_rows=True,
                                      borderless_tables=False,
                                      min_confidence=0)
    if(extracted_tables == []):
        return None
    return extracted_tables[0].df

st.title("ANOVA Solver")
uploaded_file = st.file_uploader("Choose an file")
question = st.text_input("Enter the question")

if question:
    # Search for blocking for RBD
    if 'blocking' in question:
        type = 'rbd'
        st.write("Using RBD")
    else:
        type = 'crd'
        st.write("Using CRD")
    
    # Regex search for alpha value
    alpha_search = re.search(r'α = (\d*\.?\d+)', question)
    if alpha_search:
        a = float(alpha_search.group(1))
        st.write("α = ", a)
    else:
        a = 0.01
        st.write("α = 0.01 by default")

if uploaded_file is not None and question:
    # Read CSV
    if uploaded_file.name.endswith('.csv'):
        data = pd.read_csv(uploaded_file)
    
    # Read XLSX/Excels
    if uploaded_file.name.endswith('.xlsx'):
        data = pd.read_excel(uploaded_file)
    
    # Read Images
    if uploaded_file.name.endswith('.jpg') or uploaded_file.name.endswith('.jpeg') or uploaded_file.name.endswith('.png'):
        st.image(uploaded_file, caption='Uploaded Image.', use_column_width=True)
        data = ocr(uploaded_file)
        if(data is not None):
            data = data.tail(-1)
            data = data.astype('int64')
        else:
            st.write("No tables found in the image")
            st.stop()

    st.write("Data: ", data)
    
    if type == 'crd':
        mean, total_mean, SSC, df_c, SSE, df_e, SST, df_t, MSC, MSE, F, F_critical = anova_steps_cbd(data, a)
    if type == 'rbd':    
        mean, total_mean, SSC, df_c, SSR, df_r, SSE, df_e, SST, df_t, MSC, MSR, MSE, F_treat, F_block, F_critical_treat, F_critical_block = anova_steps_rbd(data, a)
    st.write("Mean: ", mean)
    st.write("Total Mean: ", total_mean)
    
    if type == 'crd':
        table_data = {
            "SS": [SSC, SSE, SST],
            "df": [df_c, df_e, df_t],
            "MS": [MSC, MSE, ''],
            "F": [F, '', '']
        }
        table_index = ["Between Groups", "Within Groups", "Total"]
    if type == 'rbd':
        table_data = {
            "SS": [SSC, SSR, SSE, SST],
            "df": [df_c, df_r, df_e, df_t],
            "MS": [MSC, MSR, MSE, ''],
            "F": [F_treat, F_block, '', '']
        }
        table_index = ["Treatments", "Blocks", "Error", "Total"]
    
    table_df = pd.DataFrame(table_data, index=table_index)
    st.table(table_df)
    if type == 'cbd':
        st.write("F Critical: ", F_critical, " alpha = ", a)
        if F > F_critical:
            st.write("Reject Null Hypothesis")
        else:
            st.write("Accept Null Hypothesis")
    if type == 'rbd':
        st.write("F Critical (Treatments): ", F_critical_treat, " alpha = ", a)
        st.write("F Critical (Blocks): ", F_critical_block, " alpha = ", a)
        if F_treat > F_critical_treat:
            st.write("Reject Null Hypothesis for Treatments")
        else:
            st.write("Accept Null Hypothesis for Treatments")
        if F_block > F_critical_block:
            st.write("Reject Null Hypothesis for Blocks")
        else:
            st.write("Accept Null Hypothesis for Blocks")