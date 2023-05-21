import PySimpleGUI as sg
import torch
import torchvision.transforms as T
import os
import cv2
import numpy as np

from PIL import Image
import torchvision as torchvision
import io


def is_image_file(file_path):
    try:
        image = Image.open(file_path)
        image.verify()
        return True
    except:
        return False

def vid(path):
    # здесь должен быть код для обработки изображения
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    yolo = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    resnet = torch.load("model/resnet_1.pth", map_location=device)
    swans = {-1: "Не удалось распознать лебедя", 0: "Лебедь Кликун", 1: "Малый лебедь", 2: "Лебедь Шипун"}
    results = yolo(path)
    detected_imgs = []
    results = results.pandas().xyxy[0]
    for coords in results.values:
        detected_imgs.append((coords[0], coords[1], coords[2], coords[3]))
    cropped_imgs = []
    img = cv2.imread(path)
    for coords in detected_imgs:
        floatted = np.array(coords).astype(int)
        xmin, ymin, xmax, ymax = floatted
        cropped = img[ymin:ymax, xmin:xmax]
        cropped_imgs.append(cropped)
    preds = []
    for img in cropped_imgs:
        transform = T.Compose([T.ToTensor(), T.Resize((256, 256))])
        img = transform(img)
        with torch.no_grad():
            cur_preds = resnet(img.unsqueeze(0))
            preds.append(cur_preds)

    if len(preds) > 1:
        fs_mean = np.mean([i[0][0] for i in preds])
        sc_mean = np.mean([i[0][1] for i in preds])
        td_mean = np.mean([i[0][2] for i in preds])
        print(torch.tensor([fs_mean, sc_mean, td_mean]))
        result = torch.argmax(torch.tensor([fs_mean, sc_mean, td_mean])).item()
    elif len(preds)== 1:
        result = torch.argmax(preds[0], dim=1).item()
    else:
        result = -1
    return swans[result]

sg.theme('DarkAmber')
file_list=[]
layout = [
    [sg.Text('Выберите папку с изображениями:', font=('Helvetica', 14), tooltip='Нажмите, чтобы выбрать папку')],
    [sg.Input(key='-FOLDER-', enable_events=True), sg.FolderBrowse(font=('Helvetica', 12))],
    [sg.Text('ИЛИ', font=('Helvetica', 14))],
    [sg.Text('Выберите изображение для обработки:', font=('Helvetica', 14), tooltip='Выберите изображение, которое необходимо обработать')],
    [sg.Input(key='-FILE-', enable_events=True), sg.FileBrowse(font=('Helvetica', 12))],
    [sg.Text('Выберите изображения для обработки:', font=('Helvetica', 14), tooltip='Выберите изображения, которые необходимо обработать')],
    [sg.Listbox(values=[], size=(50, 10), key='-FILE LIST-', font=('Helvetica', 12), select_mode=sg.LISTBOX_SELECT_MODE_EXTENDED)],
    [sg.Button('Выбрать все', size=(10, 1), font=('Helvetica', 14), key='-SELECT ALL-'), sg.Button('Выполнить', size=(10, 1), font=('Helvetica', 14)), sg.Button('Очистить', size=(10, 1), font=('Helvetica', 14))],
    [sg.Multiline(key='-text-', font=('Helvetica', 14), size=(50, 10), autoscroll=True)],
    [sg.Image(key='-IMAGE-')],
]

window = sg.Window('Загрузка изображений', layout, size=(800, 800), resizable=True)

while True:
    event, values = window.read()

    if event in (sg.WINDOW_CLOSED, 'Exit'):
        break

    if event == '-FOLDER-':
        folder_path = values['-FOLDER-']
        file_list = os.listdir(folder_path)
        image_list = [file for file in file_list if is_image_file(os.path.join(folder_path, file))]
        window['-FILE LIST-'].update(values=image_list)

    if event == '-FILE-':
        file_path = values['-FILE-']
        if is_image_file(file_path):
            if file_path not in file_list:
                file_list.append(file_path)
            window['-FILE LIST-'].update(values=file_list)
        else:
            sg.popup('Ошибка', 'Выбранный файл не является изображением', font=('Helvetica', 14))

    if event == 'Очистить':
        window['-FILE LIST-'].update(values=[])
        window['-text-'].update('')
        window['-FOLDER-'].update('')
        window['-FILE-'].update('')
        window['-IMAGE-'].update('')
        file_list = []
        folder_path = ''

    if event == '-SELECT ALL-':
        window['-FILE LIST-'].update(set_to_index=list(range(len(window['-FILE LIST-'].get_list_values()))))

    if event == 'Выполнить':
        selected_files = values['-FILE LIST-']
        if selected_files:
            window['-IMAGE-'].update(filename='Sourse/loader.gif')
            for file in selected_files:
                file_path = os.path.join(values['-FOLDER-'], file)
                result = vid(file_path)
                print(file_path)
                window.write_event_value('-UPDATE TEXT-', f"{file}: {result}\n")
            window['-IMAGE-'].update(filename='')
        else:
            sg.popup('Ошибка', 'Необходимо выбрать хотя бы один файл для обработки', font=('Helvetica', 14))

    if event == '-UPDATE TEXT-':
        window['-text-'].print(values[event])

window.close()
'''
import PySimpleGUI as sg
import torch
import torchvision.transforms as T
import os
import cv2
import numpy as np

from PIL import Image
import torchvision as torchvision
import io


def is_image_file(file_path):
    try:
        image = Image.open(file_path)
        image.verify()
        return True
    except:
        return False

def vid(path):
    # здесь должен быть код для обработки изображения
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    yolo = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    resnet = torch.load("model/resnet_1.pth", map_location=device)
    swans = {-1: "Не удалось распознать лебедя", 0: "Лебедь Кликун", 1: "Малый лебедь", 2: "Лебедь Шипун"}
    results = yolo(path)
    detected_imgs = []
    results = results.pandas().xyxy[0]
    for coords in results.values:
        detected_imgs.append((coords[0], coords[1], coords[2], coords[3]))
    cropped_imgs = []
    img = cv2.imread(path)
    for coords in detected_imgs:
        floatted = np.array(coords).astype(int)
        xmin, ymin, xmax, ymax = floatted
        cropped = img[ymin:ymax, xmin:xmax]
        cropped_imgs.append(cropped)
    preds = []
    for img in cropped_imgs:
        transform = T.Compose([T.ToTensor(), T.Resize((256, 256))])
        img = transform(img)
        with torch.no_grad():
            cur_preds = resnet(img.unsqueeze(0))
            preds.append(cur_preds)

    if len(preds) > 1:
        fs_mean = np.mean([i[0][0] for i in preds])
        sc_mean = np.mean([i[0][1] for i in preds])
        td_mean = np.mean([i[0][2] for i in preds])
        print(torch.tensor([fs_mean, sc_mean, td_mean]))
        result = torch.argmax(torch.tensor([fs_mean, sc_mean, td_mean])).item()
    elif len(preds) == 1:
        result = torch.argmax(preds[0], dim=1).item()
    else:
        result = -1
    return swans[result]

sg.theme('DarkAmber')
file_list=[]
layout = [
    [sg.Text('Выберите папку с изображениями:', font=('Helvetica', 14), tooltip='Нажмите, чтобы выбрать папку')],
    [sg.Input(key='-FOLDER-', enable_events=True), sg.FolderBrowse(font=('Helvetica', 12))],
    [sg.Text('ИЛИ', font=('Helvetica', 14))],
    [sg.Text('Выберите изображение для обработки:', font=('Helvetica', 14), tooltip='Выберите изображение, которое необходимо обработать')],
    [sg.Input(key='-FILE-', enable_events=True), sg.FileBrowse(font=('Helvetica', 12))],
    [sg.Text('Выберите изображения для обработки:', font=('Helvetica', 14), tooltip='Выберите изображения, которые необходимо обработать')],
    [sg.Listbox(values=[], size=(50, 10), key='-FILE LIST-', font=('Helvetica', 12), select_mode=sg.LISTBOX_SELECT_MODE_EXTENDED)],
    [sg.Button('Выбрать все', size=(10, 1), font=('Helvetica', 14), key='-SELECT ALL-'), sg.Button('Выполнить', size=(10, 1), font=('Helvetica', 14)), sg.Button('Очистить', size=(10, 1), font=('Helvetica', 14))],
    [sg.Multiline(key='-text-', font=('Helvetica', 14), size=(50, 10), autoscroll=True)]
]

window = sg.Window('Загрузка изображений', layout, size=(800, 700), resizable=True)

while True:
    event,values = window.read()

    if event in (sg.WINDOW_CLOSED,'Exit'):
        break

    if event == '-FOLDER-':
        folder_path = values['-FOLDER-']
        file_list = os.listdir(folder_path)
        image_list = [file for file in file_list if is_image_file(os.path.join(folder_path, file))]
        window['-FILE LIST-'].update(values=image_list)

    if event == '-FILE-':
        file_path = values['-FILE-']
        if is_image_file(file_path):
            if file_path not in file_list:
                file_list.append(file_path)
            window['-FILE LIST-'].update(values=file_list)
        else:
            sg.popup('Ошибка', 'Выбранный файл не является изображением', font=('Helvetica', 14))

    if event == 'Очистить':
        window['-FILE LIST-'].update(values=[])
        window['-text-'].update('')
        window['-FOLDER-'].update('')
        window['-FILE-'].update('')
        file_list = []
        folder_path = ''

    if event == '-SELECT ALL-':
        window['-FILE LIST-'].update(set_to_index=list(range(len(window['-FILE LIST-'].get_list_values()))))

    if event == 'Выполнить':
        selected_files = values['-FILE LIST-']
        if selected_files:
            for file in selected_files:
                file_path = os.path.join(values['-FOLDER-'], file)
                result = vid(file_path)
                print(file_path)
                window.write_event_value('-UPDATE TEXT-', f"{file}: {result}\n")
        else:
            sg.popup('Ошибка', 'Необходимо выбрать хотя бы один файл для обработки', font=('Helvetica', 14))

    if event == '-UPDATE TEXT-':
        window['-text-'].print(values[event])

window.close()
'''