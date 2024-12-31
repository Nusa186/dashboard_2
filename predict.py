import joblib 
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Dashboard Analisis Dropout Mahasiswa",
    page_icon=":bookmark_tabs:",
    layout="wide",
    initial_sidebar_state="expanded", 
)

def load_data():
    data = pd.read_csv('data.csv', sep=';')
    return data

def load_model():
    model = joblib.load('xgboost_model.joblib')
    return model

def scaling_data():
    return joblib.load('scaler.joblib')

def predict_data(new_data):
    new_data_scaled = scaler.transform(new_data)
    prediction = model.predict(new_data_scaled)
    return prediction

df = load_data()

model = load_model()

scaler = StandardScaler()

scaled_data = scaling_data()

scaled = scaler.fit(scaled_data)
categorycal = {1:'Yes', 0:'No'}
gender = {1:'Male', 0:'Female'}
daytime= {1:'daytime', 0:'evening'}
marital_status_mapping = {
    1: "Single",
    2: "Married",
    3: "Widower",
    4: "Divorced",
    5: "Facto union",
    6: "Legally separated"
}
course_options = {
                33: 'Biofuel Production Technologies',
                171: 'Animation and Multimedia Design',
                8014: 'Social Service (evening attendance)',
                9003: 'Agronomy',
                9070: 'Communication Design',
                9085: 'Veterinary Nursing',
                9119: 'Informatics Engineering',
                9130: 'Equinculture',
                9147: 'Management',
                9238: 'Social Service',
                9254: 'Tourism',
                9500: 'Nursing',
                9556: 'Oral Hygiene',
                9670: 'Advertising and Marketing Management',
                9773: 'Journalism and Communication',
                9853: 'Basic Education',
                9991: 'Management (evening attendance)'
            }

qualification_options = {
            1: 'Secondary education',
            2: 'Higher education - bachelor\'s degree',
            3: 'Higher education - degree',
            4: 'Higher education - master\'s',
            5: 'Higher education - doctorate',
            6: 'Frequency of higher education',
            9: '12th year of schooling - not completed',
            10: '11th year of schooling - not completed',
            12: 'Other - 11th year of schooling',
            14: '10th year of schooling',
            15: '10th year of schooling - not completed',
            19: 'Basic education 3rd cycle (9th/10th/11th year) or equiv.',
            38: 'Basic education 2nd cycle (6th/7th/8th year) or equiv.',
            39: 'Technological specialization course',
            40: 'Higher education - degree (1st cycle)',
            42: 'Professional higher technical course',
            43: 'Higher education - master (2nd cycle)'
        }
occupation_mapping = {
                0: 'Student',
                1: 'Managers/Directors',
                2: 'Professionals',
                3: 'Technicians',
                4: 'Administrative',
                5: 'Services/Sales',
                6: 'Agriculture/Fisheries',
                7: 'Industry/Construction',
                8: 'Machine Operators',
                9: 'Unskilled Workers',
                10: 'Armed Forces',
                90: 'Other',
                99: 'Unknown/Blank',
                
                # Unik Father Occupation
                101: 'Armed Forces Officers',
                102: 'Armed Forces Sergeants',
                103: 'Other Armed Forces',
                112: 'Commercial Services Directors',
                114: 'Hotel/Trade Directors',
                121: 'Science/Engineering Specialists',
                124: 'Finance/Admin Specialists',
                135: 'ICT Technicians',
                161: 'Market Farmers',
                163: 'Subsistence Farmers',
                171: 'Construction Workers',
                172: 'Metal/Metallurgy Workers',
                174: 'Electricians/Electronics',
                181: 'Machine Operators',
                182: 'Assembly Workers',
                183: 'Drivers/Operators',
                195: 'Street Vendors/Service',
                
                # Unik Mother Occupation
                122: 'Health Professionals',
                123: 'Teachers',
                125: 'ICT Specialists',
                131: 'Science Technicians',
                132: 'Health Technicians',
                134: 'Legal/Social Technicians',
                141: 'Office Workers',
                143: 'Financial/Registry Operators',
                144: 'Admin Support',
                151: 'Personal Services',
                152: 'Sellers',
                153: 'Care Workers',
                173: 'Craft Workers',
                175: 'Food/Clothing Workers',
                191: 'Cleaning Workers',
                192: 'Agriculture Unskilled',
                193: 'Construction Unskilled',
                194: 'Meal Assistants'
            }
qualification_mapping = {
    1: 'Secondary Education - 12th Year of Schooling or Eq.',
    2: 'Higher Education - Bachelor\'s Degree',
    3: 'Higher Education - Degree',
    4: 'Higher Education - Master\'s',
    5: 'Higher Education - Doctorate',
    6: 'Frequency of Higher Education',
    9: '12th Year of Schooling - Not Completed',
    10: '11th Year of Schooling - Not Completed',
    11: '7th Year (Old)',
    12: 'Other - 11th Year of Schooling',
    14: '10th Year of Schooling',
    18: 'General commerce course',
    19: 'Basic Education 3rd Cycle (9th/10th/11th Year) or Equiv.',
    22: 'Technical-professional course',
    26: '7th year of schooling',
    27: '2nd cycle of the general high school course',
    29: '9th Year of Schooling - Not Completed',
    30: '8th year of schooling',
    34: 'Unknown',
    35: 'Can\'t read or write',
    36: 'Can read without having a 4th year of schooling',
    37: 'Basic education 1st cycle (4th/5th year) or equiv.',
    38: 'Basic Education 2nd Cycle (6th/7th/8th Year) or Equiv.',
    39: 'Technological specialization course',
    40: 'Higher education - degree (1st cycle)',
    41: 'Specialized higher studies course',
    42: 'Professional higher technical course',
    43: 'Higher Education - Master (2nd cycle)',
    44: 'Higher Education - Doctorate (3rd cycle)'
}

prev_qualification_mapping_dict = {
    1: "Secondary education",
    2: "Higher education - bachelor's degree",
    3: "Higher education - degree",
    4: "Higher education - master's",
    5: "Higher education - doctorate",
    6: "Frequency of higher education",
    9: "12th year of schooling - not completed",
    10: "11th year of schooling - not completed",
    12: "Other - 11th year of schooling",
    14: "10th year of schooling",
    15: "10th year of schooling - not completed",
    19: "Basic education 3rd cycle (9th/10th/11th year) or equiv.",
    38: "Basic education 2nd cycle (6th/7th/8th year) or equiv.",
    39: "Technological specialization course",
    40: "Higher education - degree (1st cycle)",
    42: "Professional higher technical course",
    43: "Higher education - master (2nd cycle)"
}

def menu():

    selected = st.sidebar.selectbox(
    "Menu",
    ["Prediksi", 'Analisis Data']
)
    match selected:
        case "Prediksi":
            class_labels = {0: 'Dropout', 1: 'Enrolled', 2: 'Graduated'}

            col1, col2, col3 = st.columns(3)

            
            with col1:
                application_order = st.number_input('Application Order(0-first choice-9 last choice)', min_value=0, max_value=5)

                course=st.selectbox('Course:', list(course_options.values()))
                course = [code for code, name in course_options.items() if name == course][0]

                daytime_evening_attendance = st.selectbox('1 Daytime 0 - evening', [0, 1])

                previous_qualification_grade = st.selectbox('Previous Qualification:', list(qualification_options.values()))
                previous_qualification_grade = [code for code, name in qualification_options.items() if name == previous_qualification_grade][0]

                admission_grade = st.number_input('Admission Grade', min_value=0, max_value=200, value=150)
                displaced = st.selectbox('Displaced (1 = Yes, 0 = No)', [0, 1])
                tuition_fees_up_to_date = st.selectbox('Tuition Fees Up to Date (1 = Yes, 0 = No)', [0, 1])         
               
            with col2:
                scholarship_holder = st.selectbox('Scholarship Holder (1 = Yes, 0 = No)', [0, 1])               
                curricular_units_1st_sem_enrolled = st.number_input('1st Sem Curricular Units Enrolled', min_value=0, max_value=19, value=0)
                curricular_units_1st_sem_evaluations = st.number_input('1st Sem Curricular Units Evaluations', min_value=0, max_value=45, value=0)
                curricular_units_1st_sem_approved = st.number_input('1st Sem Curricular Units Approved', min_value=0, max_value=26, value=0)
                curricular_units_1st_sem_grade = st.number_input('1st Sem Curricular Units Grade', min_value=0, max_value=19, value=0)
                curricular_units_1st_sem_credited = st.number_input('1st Sem Curricular Units Credited', min_value=0, max_value=20, value=0)
                international = st.selectbox('International (1 = Yes, 0 = No)', [0, 1])
                
            with col3:
                
                curricular_units_2nd_sem_credited = st.number_input('2nd Sem Curricular Units Credited', min_value=0, max_value=19, value=0)
                curricular_units_2nd_sem_enrolled = st.number_input('2nd Sem Curricular Units Enrolled', min_value=0, max_value=23, value=0)
                curricular_units_2nd_sem_evaluations = st.number_input('2nd Sem Curricular Units Evaluations', min_value=0, max_value=33, value=0)
                curricular_units_2nd_sem_approved = st.number_input('2nd Sem Curricular Units Approved', min_value=0, max_value=20, value=0)
                curricular_units_2nd_sem_grade = st.number_input('2nd Sem Curricular Units Grade', min_value=0, max_value=19, value=0)
                unemployment_rate = st.number_input('Unemployment Rate (%)', min_value=7.0, max_value=16.0, value=10.0)
                gdp = st.number_input('GDP Growth Rate (%)', min_value=-4.0, max_value=4.0, value=1.0)
            input_data = pd.DataFrame({
                    'Application_order': [application_order],
                    'Course': [course],
                    'Daytime_evening_attendance': [daytime_evening_attendance],
                    'Previous_qualification_grade': [previous_qualification_grade],
                    'Admission_grade': [admission_grade],
                    'Displaced': [displaced],
                    'Tuition_fees_up_to_date': [tuition_fees_up_to_date],
                    'Scholarship_holder': [scholarship_holder],
                    'International': [international],
                    'Curricular_units_1st_sem_credited': [curricular_units_1st_sem_credited],
                    'Curricular_units_1st_sem_enrolled': [curricular_units_1st_sem_enrolled],
                    'Curricular_units_1st_sem_evaluations': [curricular_units_1st_sem_evaluations],
                    'Curricular_units_1st_sem_approved': [curricular_units_1st_sem_approved],
                    'Curricular_units_1st_sem_grade': [curricular_units_1st_sem_grade],
                    'Curricular_units_2nd_sem_credited': [curricular_units_2nd_sem_credited],
                    'Curricular_units_2nd_sem_enrolled': [curricular_units_2nd_sem_enrolled],
                    'Curricular_units_2nd_sem_evaluations': [curricular_units_2nd_sem_evaluations],
                    'Curricular_units_2nd_sem_approved': [curricular_units_2nd_sem_approved],
                    'Curricular_units_2nd_sem_grade': [curricular_units_2nd_sem_grade],
                    'Unemployment_rate': [unemployment_rate],
                    'GDP': [gdp]
                })
            new_scaler = scaler.transform(input_data)

            predict_btn = st.button('Predict')

            if predict_btn:
                prediction = predict_data(new_scaler)
                predicted_class = class_labels.get(prediction[0])

                st.write('Prediction Result:', predicted_class)

        case 'Analisis Data':
                df_copy = df.copy()
                df_copy['Fathers_occupation'] = df_copy['Fathers_occupation'].map(occupation_mapping)
                df_copy['Mothers_occupation'] = df_copy['Mothers_occupation'].map(occupation_mapping)
                df_copy['Fathers_qualification'] = df_copy['Fathers_qualification'].map(qualification_mapping)
                df_copy['Mothers_qualification'] = df_copy['Mothers_qualification'].map(qualification_mapping)
                df_copy['Debtor'] = df_copy['Debtor'].map(categorycal)
                df_copy['Scholarship_holder'] = df_copy['Scholarship_holder'].map(categorycal)
                df_copy['Tuition_fees_up_to_date'] = df_copy['Tuition_fees_up_to_date'].map(categorycal)
                df_copy['Gender'] = df_copy['Gender'].map(gender)        
                df_copy['International'] = df_copy['International'].map(categorycal)
                df_copy['Displaced'] = df_copy['Displaced'].map(categorycal)
                df_copy['Daytime_evening_attendance'] = df_copy['Daytime_evening_attendance'].map(daytime)
                df_copy['Previous_qualification'] = df_copy['Previous_qualification'].map(prev_qualification_mapping_dict)
                df_copy['Marital_status'] = df_copy['Marital_status'].map(marital_status_mapping)

                st.markdown("<h1 style='text-align: center;'>Dashboard</h1>", unsafe_allow_html=True)
                btn = st.checkbox('View Student Status Distribution')
                if btn:
                    status_count = df_copy['Status'].value_counts()
                    fig, ax = plt.subplots(figsize=(5, 8), dpi=100)
                    ax.pie(status_count, labels=status_count.index, autopct='%1.1f%%', startangle=90)
                    ax.set_title('Student Status Distribution')
                    fig.patch.set_facecolor('none')
                    ax.set_facecolor('white')
                    plt.tight_layout()
                    st.pyplot(fig)

                col1, col2, = st.columns(2)
                col3, col4, = st.columns(2)
                col5, = st.columns(1)

                with col1:
                    col1.header('Academic Factors')
                    selected_category = col1.selectbox('Choose Category (Academic):', options=[
                                                                                               'Curricular_units_1st_sem_approved',
                                                                                                'Curricular_units_2nd_sem_approved'])
                    col1.write(f"{selected_category} VS Status")

                    if selected_category:                    
                        category_counts = df_copy.groupby([selected_category, 'Status']).size().unstack(fill_value=0)

                        col1.bar_chart(category_counts)
                        st.write('''Mahasiswa dengan jumlah curricular units (satuan kredit) yang disetujui 
                                 pada semester 1 dan 2 sebanyak 0 memiliki tingkat dropout yang tinggi.''')
                with col2:
                    col2.header('Finance and Economic Factors')
                    selected_category = col2.selectbox('Choose Category (Economic):', options=['Scholarship_holder'
                                                                                                ,'Debtor'])
                    col2.write(f"{selected_category} VS Status")

                    if selected_category:                    
                        category_counts = df_copy.groupby([selected_category, 'Status']).size().unstack(fill_value=0)
                        
                        col2.bar_chart(category_counts, stack=False)
                        st.write('''Mahasiswa yang tidak memiliki beasiswa memiliki tingkat 
                                 dropout yang lebih tinggi dibandingkan mahasiswa penerima beasiswa.''')
                with col3:
                    col3.header('Parents Information')
                    selected_category = col3.selectbox('Choose Category (Environment):', options=['Mothers_qualification',
                                                                                                'Fathers_qualification', 
                                                                                                'Mothers_occupation', 
                                                                                                'Fathers_occupation'])
                    col3.write(f'{selected_category} VS Student Status')
                    
                    if selected_category:
                        category_counts = df_copy.groupby([selected_category, 'Status']).size().unstack(fill_value=0)
                        
                        col3.bar_chart(category_counts)
                        st.write('''Mayoritas mahasiswa yang dropout berasal dari keluarga dengan latar 
                                 belakang pekerjaan sebagai unskilled workers dan pendidikan orang tua hanya 
                                 sampai pada basic education 1st cycle.''')

                with col4:
                    col4.header('Social and Demografy')
                    selected_category = col4.selectbox('Choose Category:', options=['International',
                                                                                    'Displaced', 
                                                                                    'Daytime_evening_attendance'])
                    col4.write(f'{selected_category} VS Student Status')
                    
                    if selected_category:
                        category_counts = df_copy.groupby([selected_category, 'Status']).size().unstack(fill_value=0)
                        col4.bar_chart(category_counts, stack=False)
                        st.write('Mahasiswa nasional memiliki tingkat dropout yang lebih tinggi dibandingkan mahasiswa internasional.')

                with col5:
                    col5.header('Student Information')
                    selected_category = col5.selectbox('Choose Category:', options=['Marital_status',
                                                                                    'Gender',
                                                                                    'Age_at_enrollment'])
                                                       
                    col5.write(f'{selected_category} VS Student Status')
                    
                    if selected_category:
                        category_counts = df_copy.groupby([selected_category, 'Status']).size().unstack(fill_value=0)
                        col5.bar_chart(category_counts)
                        st.write('''Mahasiswa dengan status single memiliki tingkat dropout yang lebih tinggi dibandingkan dengan mahasiswa yang sudah menikah atau dalam hubungan.''')
                st.markdown("## Hasil Analisis")
                st.markdown("""
                - Mahasiswa yang tidak lulus satu pun mata kuliah pada dua semester pertama menunjukkan adanya kesulitan dalam beradaptasi dengan sistem perkuliahan. 
                - Mahasiswa yang tidak memiliki dukungan finansial lebih rentan terhadap tekanan ekonomi yang dapat memengaruhi kelangsungan studi mereka.  
                - Latar belakang pendidikan dan pekerjaan orang tua sangat memengaruhi motivasi serta dukungan akademik yang diterima oleh mahasiswa. 
                - Mahasiswa internasional cenderung memiliki motivasi lebih tinggi karena belajar di luar negeri. 
                - Mahasiswa single cenderung memiliki tekanan sosial dan emosional yang lebih tinggi, yang berkontribusi pada ketidakstabilan akademik. 
                """)
if __name__ == "__main__":
    menu()