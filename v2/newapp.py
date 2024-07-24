import streamlit as st
from anova_steps import anova_steps_cbd, anova_steps_rbd, anova_steps_fd
import pandas as pd
import cv2
import pytesseract
import numpy as np

def select_table_region(image_path):
    image = cv2.imread(image_path)
    clone = image.copy()
    table_roi = []

    def select_roi(event, x, y, flags, param):
        nonlocal table_roi, clone
        if event == cv2.EVENT_LBUTTONDOWN:
            table_roi = [(x, y)]
        elif event == cv2.EVENT_LBUTTONUP:
            table_roi.append((x, y))
            cv2.rectangle(clone, table_roi[0], table_roi[1], (0, 255, 0), 2)
            cv2.imshow("image", clone)

    cv2.namedWindow("image")
    cv2.setMouseCallback("image", select_roi)
    while True:
        cv2.imshow("image", clone)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("r"):
            clone = image.copy()
        elif key == ord("c"):
            break
    cv2.destroyAllWindows()
    return table_roi


def extract_text_from_image(image):
    custom_config = r'--oem 3 --psm 6'
    return pytesseract.image_to_string(image, config=custom_config)


def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    dilated = cv2.dilate(thresh, kernel, iterations=1)
    contours, _ = cv2.findContours(
        dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for contour in contours:
        [x, y, w, h] = cv2.boundingRect(contour)
        if h < 40 or w < 40:
            continue
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 255), 2)
    return image

def process_image(image_path, table_roi):
    image = cv2.imread(image_path)
    image = preprocess_image(image)
    table_image = image[table_roi[0][1]:table_roi[1]
                        [1], table_roi[0][0]:table_roi[1][0]]

    mask = np.ones(image.shape[:2], dtype="uint8") * 255
    cv2.rectangle(mask, table_roi[0], table_roi[1], 0, -1)
    text_image = cv2.bitwise_and(image, image, mask=mask)

    remaining_text = extract_text_from_image(text_image)

    table_text = extract_text_from_image(table_image)
    rows = [row.split() for row in table_text.split('\n') if row.strip()]
    try:
        table_df = pd.DataFrame(rows[1:], columns=rows[0])
    except ValueError as ve:
        print("Error in DataFrame creation:", ve)
        table_df = pd.DataFrame(rows)

    return remaining_text, table_df

def predict_alpha_value(question):
    import pickle
    import joblib

    vectorizer = joblib.load("vectorizer.pkl")
    text_sample = [question]
    text_vectorized_sample = vectorizer.transform(text_sample)

    alpha_predict = joblib.load('alpha_model.pkl')
    return alpha_predict.predict(text_vectorized_sample).round(2)

def distinct_elements(data):
    import numpy as np
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity

    model = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens')

    headers = data.columns
    headers = [
        header.split()[0] if ' ' in header else header for header in headers]
    numeric_headers = [int(header) for header in headers if header.isdigit()]
    max_header = max(numeric_headers) if numeric_headers else None
    headers = [header if not header.isdigit() else str(max_header)
            for header in headers]
    header_embeddings = model.encode(headers)
    similarity_matrix = cosine_similarity(header_embeddings)
    np.fill_diagonal(similarity_matrix, 0)
    max_similarity = np.max(similarity_matrix, axis=1)
    distinct_entities = np.sum(max_similarity < 0.7)
    return distinct_entities + 1

def isNested(data):
    nested = True
    for i in range(len(data)):
        if data.iloc[i].iloc[0] != data.iloc[0].iloc[0]:
            nested = False
            break

    return nested

def predict_type(col, entities, nested):
    import joblib 
    model = joblib.load('model.pkl')

    X_sample = [[col, entities, nested]]
    y_sample = model.predict(X_sample)[0]

    st.write(f"Predicted type for sample data: {y_sample}")
    
    return y_sample


st.title("ANOVA Solver")
uploaded_file = st.file_uploader("Choose an file")
question = st.text_input("Enter the question")

if uploaded_file is not None:
    # Read CSV
    if uploaded_file.name.endswith('.csv'):
        data = pd.read_csv(uploaded_file)
    
    # Read XLSX/Excels
    if uploaded_file.name.endswith('.xlsx'):
        data = pd.read_excel(uploaded_file)
    
    # Read Images
    if uploaded_file.name.endswith('.jpg') or uploaded_file.name.endswith('.jpeg') or uploaded_file.name.endswith('.png'):
        st.image(uploaded_file, caption='Uploaded Image.', use_column_width=True)
        import tempfile
        # Create a temporary file to save the uploaded image
        with tempfile.NamedTemporaryFile(delete=False, suffix=uploaded_file.name[-4:]) as tmp_file:
            # Write the uploaded file's content to the temporary file
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name

        table_roi = select_table_region(tmp_file_path)
        question, data = process_image(tmp_file_path, table_roi)
        
        if(data is not None):
            #data = data.tail(-1)
            st.write("Data: ", data)
            data = data.astype('int64')
            
        else:
            st.write("No tables found in the image")
            st.stop()

    if(data.columns.size < 3):
        st.write("Not enough columns to perform ANOVA, use other methods which are easier than anova")
        st.stop()
    
    if(len(question) == 0):
        st.write("Please enter a question to proceed")
        st.stop()
    
    a = predict_alpha_value(question)[0]
    st.write("α = ", a)
    type = predict_type(len(data.columns), distinct_elements(data), isNested(data))
    type = 'FD'
    
    if type == 'CRD':
        mean, total_mean, SSC, df_c, SSE, df_e, SST, df_t, MSC, MSE, F, F_critical = anova_steps_cbd(data, a)
        st.write("Mean: ", mean)
        st.write("Total Mean: ", total_mean)
    if type == 'RBD':
        #drop 1st row
        data = data.tail(-1)
        #drop 1st and 2nd columns coz in data we have row names and BLOCKING written
        data = data.drop(columns=[data.columns[0], data.columns[1]])
        mean, total_mean, SSC, df_c, SSR, df_r, SSE, df_e, SST, df_t, MSC, MSR, MSE, F_treat, F_block, F_critical_treat, F_critical_block = anova_steps_rbd(data, a)
        st.write("Mean: ", mean)
        st.write("Total Mean: ", total_mean)
    if type == 'FD':
        # Clean data
        n = 4
        row_treat = 2
        SST, df_t, SSC, df_c, SSR, df_r, SSI, df_i, SSE, df_e, MSR, MSC, MSI, MSE, F_r, F_c, F_i, F_critical_row, F_critical_col, F_critical_iter = anova_steps_fd(data, a, n, row_treat)
    
    st.write("Data: ", data)
    if type == 'CRD':
        table_data = {
            "SS": [SSC, SSE, SST],
            "df": [df_c, df_e, df_t],
            "MS": [MSC, MSE, ''],
            "F": [F, '', '']
        }
        table_index = ["Between Groups", "Within Groups", "Total"]
    if type == 'RBD':
        table_data = {
            "SS": [SSC, SSR, SSE, SST],
            "df": [df_c, df_r, df_e, df_t],
            "MS": [MSC, MSR, MSE, ''],
            "F": [F_treat, F_block, '', '']
        }
        table_index = ["Treatments", "Blocks", "Error", "Total"]
        
    if type == 'FD':
        table_data = {
            "SS": [SSR, SSC, SSI, SSE, SST],
            "df": [df_r, df_c, df_i, df_e, df_t],
            "MS": [MSR, MSC, MSI, MSE, ''],
            "F": [F_critical_row, F_critical_col, F_critical_iter, '', '']
        }
        table_index = ["Rows", "Columns", "Iters", "Error", "Total"]
    
    table_df = pd.DataFrame(table_data, index=table_index)
    st.table(table_df)
    if type == 'CRD':
        st.write("F Critical: ", F_critical, " alpha = ", a)
        if F > F_critical:
            st.write("Reject Null Hypothesis")
        else:
            st.write("Accept Null Hypothesis")
    if type == 'RBD':
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
    if type == 'FD':
        st.write("F Critical (Rows): ", F_critical_row, " alpha = ", a)
        st.write("F Critical (Columns): ", F_critical_col, " alpha = ", a)
        st.write("F Critical (Iters): ", F_critical_iter, " alpha = ", a)
        if F_r > F_critical_row:
            st.write("Reject Null Hypothesis for Rows")
        else:
            st.write("Accept Null Hypothesis for Rows")
        if F_c > F_critical_col:
            st.write("Reject Null Hypothesis for Columns")
        else:
            st.write("Accept Null Hypothesis for Columns")
        if F_i > F_critical_iter:
            st.write("Reject Null Hypothesis for Iters")
        else:
            st.write("Accept Null Hypothesis for Iters")