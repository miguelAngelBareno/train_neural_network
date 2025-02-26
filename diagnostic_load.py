import csv

def load_diagnostic(file_name:str, full_data:list):

    if not file_name in full_data:
        raise

    disease = full_data[file_name][0]
    x = full_data[file_name][1]
    y = full_data[file_name][2]
    w = full_data[file_name][3]
    h = full_data[file_name][4]

    return disease, x, y, w, h



def full_data_diagnosis_load(path_csv:str)->dict:
    data = {}
    try:
        with open(path_csv, newline='', encoding='utf-8') as file:
            reader = csv.reader(file)
            next(reader, None)  # Skip the header row
            for row in reader:
                if row:
                    key = row[0]
                    values = row[1:]
                    data[key] = values
    except FileNotFoundError:
        print(f"Error: The file '{path_csv}' was not found.")
    except Exception as e:
        print(f"Error reading the file: {e}")
    return data