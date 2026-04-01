import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# =====================================================================
# PARTE 1: PARÁMETROS DEL EXPERIMENTO (¡Los estudiantes modifican aquí!)
# =====================================================================
# INSTRUCCIÓN: Ingresen el error instrumental del micrómetro en metros.
error_micrometro = 0.00001  # REEMPLAZAR CON EL VALOR CORRECTO EN METROS

# Nombre del archivo de texto con los datos de las lentejas.
nombre_archivo = 'datos_lentejas.txt'

# =====================================================================
# FIX: Crear un archivo dummy si no existe para que el código pueda ejecutarse
# =====================================================================
import os
if not os.path.exists(nombre_archivo):
    print(f"Creando archivo dummy '{nombre_archivo}' para demostración...")
    # Generar 1000 números aleatorios siguiendo una distribución normal
    # con media 0.005 m y desviación estándar 0.0005 m.
    dummy_data = np.random.normal(loc=0.005, scale=0.0005, size=1000)
    np.savetxt(nombre_archivo, dummy_data, fmt='%.6f')
    print(f"Archivo '{nombre_archivo}' creado con {len(dummy_data)} datos dummy.")

# =====================================================================
# PARTE 2: LECTURA DE DATOS
# =====================================================================
try:
    datos_completos = np.loadtxt(nombre_archivo)
    N_total = len(datos_completos)
    print(f"¡Datos cargados exitosamente! Se leyeron {N_total} mediciones en total.")
except FileNotFoundError:
    print(f"Error: No se encontró el archivo '{nombre_archivo}'.")
    raise

# =====================================================================
# PARTE 3: CREACIÓN DE SUBCONJUNTOS (N/10, N/2, N)
# =====================================================================
# Usamos "//" para hacer una división entera, ya que los índices
# de un arreglo deben ser números enteros exactos.
datos_n10 = datos_completos[:N_total // 10]
datos_n2  = datos_completos[:N_total // 2]
datos_n   = datos_completos  # La muestra completa

# Guardamos los subconjuntos y sus títulos en una lista para graficarlos fácilmente
subconjuntos = [
    (f"Primeros N/10 datos (N={len(datos_n10)})", datos_n10),
    (f"Primeros N/2 datos (N={len(datos_n2)})", datos_n2),
    (f"Muestra Completa (N={len(datos_n)})", datos_n)
]

# =====================================================================
# PARTE 4: VISUALIZACIÓN COMPARATIVA (3 GRÁFICOS JUNTOS)
# =====================================================================
# Creamos una figura ancha que contendrá 3 gráficos en 1 fila (1, 3)
fig, arreglos_graficos = plt.subplots(1, 3, figsize=(18, 5))

# Iteramos sobre nuestros 3 subconjuntos para dibujar cada gráfico
for i in range(3):
    titulo, datos_actuales = subconjuntos[i]
    ax = arreglos_graficos[i] # Seleccionamos el gráfico específico (0, 1 o 2)

    # 4.1 Cálculos estadísticos del subconjunto actual
    media_actual = np.mean(datos_actuales)
    # Si hay muy pocos datos (ej. N=1), la desviación estándar da error.
    # Usamos un condicional simple para evitar que el código colapse.
    if len(datos_actuales) > 1:
        std_actual = np.std(datos_actuales, ddof=1)
    else:
        std_actual = 0

    # 4.2 Crear el Histograma
    n_bins = int(np.sqrt(len(datos_actuales)))
    # Evitamos que n_bins sea 0 si hay muy pocos datos
    n_bins = max(1, n_bins)

    ax.hist(datos_actuales, bins=n_bins, density=True, alpha=0.6,
            color='royalblue', edgecolor='black', label='Datos')

    # 4.3 Crear la Curva Gaussiana Teórica (solo si hay más de 1 dato)
    if std_actual > 0:
        x_min, x_max = np.min(datos_actuales), np.max(datos_actuales)
        x_curva = np.linspace(x_min - std_actual, x_max + std_actual, 500)
        y_gaussiana = norm.pdf(x_curva, media_actual, std_actual)

        ax.plot(x_curva, y_gaussiana, 'r-', linewidth=2,
                label=f'$\mu={media_actual:.5f}$ m\n$\sigma={std_actual:.5f}$ m')

    # 4.4 Formato de Ejes (Estándar DFIS)
    ax.set_title(titulo, fontsize=12)
    ax.set_xlabel(f'(Espesor $\pm$ {error_micrometro}) [m]', fontsize=10)

    # Solo le ponemos la etiqueta "Y" al primer gráfico para no saturar la imagen
    if i == 0:
        ax.set_ylabel('Densidad de Probabilidad', fontsize=10)

    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, linestyle='--', alpha=0.5)

# Ajustamos el espaciado para que no se superpongan los textos
plt.tight_layout()

# Título general para toda la figura
fig.suptitle('Evolución de la Distribución Estadística del Espesor', fontsize=16, y=1.05)

plt.show()
