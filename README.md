# Lil-Pump-Lyric-Generator
Este programa usa python 3.7.9

Entrena basándose en el input de 'lyrics.txt' asi que puedes cambiar las lyrics para que imiten a cualquier otro artiste, pero Lil Pump es mucho Lil Pump

Como instalar:
  1. Crea una carpeta donde quieras que vaya el Lil-Pump-Lyric-Generator
  2. Clona el repositorio a la carpeta
     ``` 
     git clone https://github.com/amsanabria/Lil-Pump-Lyric-Generator.git
     ```
  3. Instala pip en caso de no tenerlo 
  4. Instala los requerimientos con  
      ```
      pip install -r requirements.txt
      ```
      
 Como usar:
  1. Ejecuta LilPump.py:
     ```
     python LilPump.py
     ```
  2. Elige si quieres entrenar la RN o generar lyrics (las lyrics se guardarán en output.txt aparte de salir por consola).
  3. Se recomienda entrenar la RN por un mínimo de 70 epochs, ya que si no el output sera 'LKJSIUSAFNSAfkskfkk bitch asjfasdjpakdñ'
 
Descargar modelo pre-entrenado
  1. Descarga el modelo de Mega
     ``` 
     https://mega.nz/folder/6B8EiDTI#PhttcLJAFW6u1JL-bccgnQ
     ```
  2. Se descargarán 2 archivos pretrained.index y pretrained.data-00000-of-00001
  3. Dejalos en la raiz del proyecto
  4. En el menú elige 'Load Pretrained Model'
  5. El modelo ya está cargado, sigue entrenándolo o genera alguna lyric!
