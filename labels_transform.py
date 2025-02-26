from sklearn.preprocessing import LabelEncoder

# Función 1: Crear el transformador
def create_transformer(list_words:list)->LabelEncoder:
    print(list_words)
    encoder = LabelEncoder()
    encoder.fit(list_words)
    return encoder

# Función 2: Devolver el número correspondiente a la palabra
def code_word(word:str, encoder:LabelEncoder)->int:
    return encoder.transform([word])[0]

# Función 3: Devolver la palabra correspondiente al número
def decode_word(number:int, encoder:LabelEncoder)->str:
    return encoder.inverse_transform([number])[0]
