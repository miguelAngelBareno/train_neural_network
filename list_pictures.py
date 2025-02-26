import os


def list_pictures(path:str)->list:
    if not os.path.isdir(path):
        raise ValueError(f"El path '{path}' no es un directorio válido.")
    
    return [archivo for archivo in os.listdir(path) if os.path.isfile(os.path.join(path, archivo))]


if __name__ == "__main__":
    path_imagenes = "classification_proyect/code/images"
    lista_imagenes = list_pictures(path_imagenes)
    print(lista_imagenes)
