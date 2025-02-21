#!/usr/bin/env python
# -*- coding: utf-8 -*-

from tkinter import *
from tkinter import ttk, font, filedialog, Entry
from tkinter.messagebox import askokcancel, showinfo, WARNING
from PIL import ImageTk, Image
import csv
import tkcap
import numpy as np
import tensorflow as tf
import os  # Añade esta importación al inicio del archivo
import sys



# Resto de los imports
from src.data.read_img import read_dicom_file
from src.models.grad_cam import grad_cam
from src.interface.integrator import predict


tf.config.run_functions_eagerly(True)

class App:
    def __init__(self):
        self.root = Tk()
        self.root.title("Herramienta para la detección rápida de neumonía")

        #   BOLD FONT
        fonti = font.Font(weight="bold")

        self.root.geometry("815x560")
        self.root.resizable(0, 0)

        #   LABELS
        self.lab1 = ttk.Label(self.root, text="Imagen Radiográfica", font=fonti)
        self.lab2 = ttk.Label(self.root, text="Imagen con Heatmap", font=fonti)
        self.lab3 = ttk.Label(self.root, text="Resultado:", font=fonti)
        self.lab4 = ttk.Label(self.root, text="Cédula Paciente:", font=fonti)
        self.lab5 = ttk.Label(
            self.root,
            text="SOFTWARE PARA EL APOYO AL DIAGNÓSTICO MÉDICO DE NEUMONÍA",
            font=fonti,
        )
        self.lab6 = ttk.Label(self.root, text="Probabilidad:", font=fonti)

        #   STRING VARIABLES
        self.ID = StringVar()
        self.result = StringVar()

        #   INPUT BOXES
        self.text1 = ttk.Entry(self.root, textvariable=self.ID, width=10)
        self.ID_content = self.text1.get()

        #   IMAGE INPUT BOXES
        self.text_img1 = Text(self.root, width=31, height=15)
        self.text_img2 = Text(self.root, width=31, height=15)
        self.text2 = Text(self.root)
        self.text3 = Text(self.root)

        #   BUTTONS
        self.button1 = ttk.Button(
            self.root, text="Predecir", state="disabled", command=self.run_model
        )
        self.button2 = ttk.Button(
            self.root, text="Cargar Imagen", command=self.load_img_file
        )
        self.button3 = ttk.Button(self.root, text="Borrar", command=self.delete)
        self.button4 = ttk.Button(self.root, text="PDF", command=self.create_pdf)
        self.button6 = ttk.Button(
            self.root, text="Guardar", command=self.save_results_csv
        )

        #   WIDGETS POSITIONS
        self.lab1.place(x=110, y=65)
        self.lab2.place(x=545, y=65)
        self.lab3.place(x=500, y=350)
        self.lab4.place(x=65, y=350)
        self.lab5.place(x=122, y=25)
        self.lab6.place(x=500, y=400)
        self.button1.place(x=220, y=460)
        self.button2.place(x=70, y=460)
        self.button3.place(x=670, y=460)
        self.button4.place(x=520, y=460)
        self.button6.place(x=370, y=460)
        self.text1.place(x=200, y=350)
        self.text2.place(x=610, y=350, width=90, height=30)
        self.text3.place(x=610, y=400, width=90, height=30)
        self.text_img1.place(x=65, y=90)
        self.text_img2.place(x=500, y=90)

        self.text1.focus_set()
        self.array = None
        self.reportID = 0
        self.root.mainloop()

    def load_img_file(self):
           # Validar que haya un número de cédula
        cedula = self.text1.get().strip()  # strip() elimina espacios en blanco
        if not cedula:
            showinfo(title="Error", message="Por favor ingrese el número de cédula del paciente antes de guardar.", icon="error")
            return
        filepath = filedialog.askopenfilename(
            initialdir="/",
            title="Select image",
            filetypes=(
                ("DICOM", "*.dcm"),
                ("JPEG", "*.jpeg"),
                ("jpg files", "*.jpg"),
                ("png files", "*.png"),
            ),
        )
        if filepath:
            self.array, img2show = read_dicom_file(filepath)
            self.img1 = img2show.resize((250, 250), Image.Resampling.LANCZOS)
            self.img1 = ImageTk.PhotoImage(self.img1)
            self.text_img1.image_create(END, image=self.img1)
            self.button1["state"] = "enabled"

    def run_model(self):
        self.label, self.proba, self.heatmap = predict(self.array)
        self.img2 = Image.fromarray(self.heatmap)
        self.img2 = self.img2.resize((250, 250), Image.Resampling.LANCZOS)
        self.img2 = ImageTk.PhotoImage(self.img2)
        self.text_img2.image_create(END, image=self.img2)
        self.text2.insert(END, self.label)
        self.text3.insert(END, "{:.2f}".format(self.proba) + "%")

    def save_results_csv(self):
        # Validar que se haya hecho una predicción
        if not hasattr(self, 'label') or not hasattr(self, 'proba'):
            showinfo(title="Error", message="Por favor realice una predicción antes de guardar.", icon="error")
            return
        # Obtener la ruta raíz del proyecto (dos niveles arriba desde detector_neumonia.py)
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        
        # Crear la carpeta reports en la raíz si no existe
        reports_folder = os.path.join(project_root, "reports")
        if not os.path.exists(reports_folder):
            os.makedirs(reports_folder)
        
        # Definir la ruta del archivo CSV
        ruta_guardado = os.path.join(reports_folder, "historial.csv")
        
        # Guardar los resultados
        with open(ruta_guardado, "a") as csvfile:
            w = csv.writer(csvfile, delimiter="-")
            w.writerow([self.text1.get(), self.label, "{:.2f}".format(self.proba) + "%"])
            showinfo(title="Guardar", message=f"Los datos se guardaron con éxito en:\n{ruta_guardado}")

    def create_pdf(self):
        # Validar que se haya hecho una predicción
        if not hasattr(self, 'label') or not hasattr(self, 'proba'):
            showinfo(title="Error", message="Por favor realice una predicción antes de guardar.", icon="error")
            return
        # Obtener la ruta raíz del proyecto y la carpeta reports
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        reports_folder = os.path.join(project_root, "reports")
        
        # Crear la carpeta reports si no existe
        if not os.path.exists(reports_folder):
            os.makedirs(reports_folder)
        
        # Capturar la imagen
        cap = tkcap.CAP(self.root)
        
        # Crear las rutas para los archivos jpg y pdf
        jpg_filename = f"Reporte{self.reportID}.jpg"
        pdf_filename = f"Reporte{self.reportID}.pdf"
        jpg_path = os.path.join(reports_folder, jpg_filename)
        pdf_path = os.path.join(reports_folder, pdf_filename)
        
        # Capturar y procesar la imagen
        img = cap.capture(jpg_path)
        img = Image.open(jpg_path)
        img = img.convert("RGB")
        
        # Guardar como PDF
        img.save(pdf_path)
        self.reportID += 1
        
        # Eliminar el archivo JPG temporal
        os.remove(jpg_path)
        
        showinfo(title="PDF", message=f"El PDF fue generado con éxito en:\n{pdf_path}")

    def delete(self):
        answer = askokcancel(
            title="Confirmación", message="Se borrarán todos los datos.", icon=WARNING
        )
        if answer:
            self.text1.delete(0, "end")
            self.text2.delete(1.0, "end")
            self.text3.delete(1.0, "end")
            self.text_img1.delete(self.img1, "end")
            self.text_img2.delete(self.img2, "end")
            showinfo(title="Borrar", message="Los datos se borraron con éxito")

def main():
    my_app = App()
    return 0

if __name__ == "__main__":
    main()