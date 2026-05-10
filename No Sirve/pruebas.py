
import pandas as pd

# 1. CARGA DE DATOS 

df = pd.read_stata(
    r"C:\Users\Lucia\OneDrive\Documentos\Uni-Pc_Lucia\Master\TFM\Dataset\UKDA-6614-stata\stata\stata14_se\ukhls\n_indresp.dta", 
    convert_categoricals=False
    )

wanted_cols = [
    # Tratamiento - Teletrabajo
    "n_wkhome",
    
    # Outcome - Satisfaccion laboral global
    "n_jbsat",
    
    # Variables demográficas
    "n_pdvage",
    "n_sex",
    "n_qfhigh",    # nivel de educacion mas alto completado
    
    # Estructura laboral
    "n_jbhrs",      # horas trabajadas a la semana
    "n_jbsect",     # tipo de organizacion/empresa para la que trabaja
    "n_jbsectpub",
    "n_jbsize",    # tamaño de la empresa
    "n_jbmngr",    # puesto de trabajo
    
    # Organización del trabajo
    "n_wkaut1",    # autonomia sobre las tareas a realizar
    "n_wkaut2",    # autonomia sobre el ritmo de trabajo
    "n_wkaut3",    # autonomia sobre como hacer el trabajo
    "n_wkaut4",    # autonomia sobre el orden para realizar las tareas
    "n_wkaut5",    # autonomia sobre las horas de trabajo
    "n_wkaut6",    # autonomia sobre el lugar de trabajo
    "n_jbflex7",   # horarios flexibles: trabajar desde casa regularmente
    "n_jbfxuse7",  # uso del trabajo flexible
    "n_wktech1",    # uso de ordenadores de mesa en el trabajo
    "n_wktech2",    # uso deportatiles en el trabajo
    "n_wktech3",    # uso de telefonos en el trabajo
    "n_wktech4",    # uso de tablets en el trabajo
    "n_wktech5",    # uso de 'feature phone/non-touchscreen mobilephone' en el trabajo
    "n_wktech6"     # uso de ordenadores de mesa en el trabajo
]

existing_cols = [col for col in wanted_cols if col in df.columns]
not_existing_cols = [col for col in wanted_cols if col not in df.columns]
df_model = df[existing_cols]

print(not_existing_cols)



for col in df.columns:
    if "wkhome" in col.lower():
        print(col)

