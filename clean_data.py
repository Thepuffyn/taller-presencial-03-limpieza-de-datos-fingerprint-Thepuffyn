"""Taller evaluable presencial"""

from json import load
import nltk
import pandas as pd


def load_data(input_file):
    """Lea el archivo usando pandas y devuelva un DataFrame"""
    df = pd.read_csv(input_file, sep = "\t")
    return df

def create_fingerprint(df):
    """Cree una nueva columna en el DataFrame que contenga el fingerprint de la columna 'text'"""

    df=df.copy()

    # 1. Copie la columna 'text' a la columna 'fingerprint'
    df["fingerprint"] = df["text"]
    # 2. Remueva los espacios en blanco al principio y al final de la cadena
    df["fingerprint"]=df["fingerprint"].str.strip()
    # 3. Convierta el texto a minúsculas
    df["fingerprint"]=df["fingerprint"].str.lower()
    # 4. Transforme palabras que pueden (o no) contener guiones por su version sin guion.
    df["fingerprint"]=df["fingerprint"].str.replace("-","")    
    # 5. Remueva puntuación y caracteres de control
    df["fingerprint"]=df["fingerprint"].str.replace("[^\w\s]","")
    #se pueden encadenar todas las transformaciones o puede hacerse linea por linea   
    # 6. Convierta el texto a una lista de tokens
    df["fingerprint"]=df["fingerprint"].str.split()
    # 7. Transforme cada palabra con un stemmer de Porter
    df["fingerprint"]=df["fingerprint"].apply(lambda x: [nltk.PorterStemmer().stem(w) for w in x])
    
    # 8. Ordene la lista de tokens y remueve duplicados
    df["fingerprint"]=df["fingerprint"].apply(lambda x:sorted(set(x)))
    
    # 9. Convierta la lista de tokens a una cadena de texto separada por espacios
    df["fingerprint"]=df["fingerprint"].str.join(" ")
    #stemer es una función para reducir la palabra a la raiz muy usada en mineria de texto

    return df


def generate_cleaned_column(df):
    """Crea la columna 'cleaned' en el DataFrame"""

    df = df.copy()

    #reemplazar todas las apariciones del fingerprint por la mas frecuente o mas corta o larga.
    #primero que aparezca y esto depende del problema que se tenga para la limpieza.
    
    # 1. Ordene el dataframe por 'fingerprint' y 'text'
    fingerprints = df.sort_values(by = ["fingerprint", "text"]).copy()
    
    # 2. Seleccione la primera fila de cada grupo de 'fingerprint'
    fingerprints = df.groupby("fingerprint").first().reset_index() 
    
    # 3.  Cree un diccionario con 'fingerprint' como clave y 'text' como valor
    fingerprints = fingerprints.set_index("fingerprint")["text"].to_dict()

    # 4. Cree la columna 'cleaned' usando el diccionario
    df["cleaned"] = df["fingerprint"].map(fingerprints)
    return df


def save_data(df, output_file):
    """Guarda el DataFrame en un archivo"""
    df = df.copy()
    df = df[["cleaned"]]
    df = df.rename(columns = {"cleaned":"text"})
    # Solo contiene una columna llamada 'texto' al igual
    # que en el archivo original pero con los datos limpios
    df.to_csv(output_file, sep="\t", index = False)
    

def main(input_file, output_file):
    """Ejecuta la limpieza de datos"""
    df = load_data(input_file)
    df = create_fingerprint(df)
    df = generate_cleaned_column(df)
    df.to_csv("test.csv", index=False)
    save_data(df, output_file)


if __name__ == "__main__":
    main(
        input_file="input.txt",
        output_file="output.txt"
    )
