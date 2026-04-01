import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from matplotlib.ticker import FormatStrFormatter

# 1. Definir la función de ajuste (seno amortiguado)
def seno_amortiguado(t, A, gamma, omega, phi):
    """
    Ecuación: y(t) = A * exp(-gamma * t) * sin(omega * t + phi)
    """
    return A * np.exp(-gamma * t) * np.sin(omega * t + phi)

def analizar_oscilacion(archivo_txt, error_tiempo, error_posicion, nombre_imagen):
    # 2. Cargar los datos desde el archivo .txt
    try:
        datos = np.loadtxt(archivo_txt)
        t = datos[:, 0]
        y = datos[:, 1]
    except Exception as e:
        print(f"Error al cargar el archivo: {e}")
        return

    # 3. Centrar la oscilación
    y_centrado = y - np.mean(y)

    # 4. Estimación de parámetros iniciales para ayudar al ajuste
    A0 = (np.max(y_centrado) - np.min(y_centrado)) / 2  # Amplitud inicial
    gamma0 = 0.1                                        # Factor de amortiguamiento inicial
    
    # Estimar la frecuencia angular cruzando por cero
    cruces_cero = np.where(np.diff(np.sign(y_centrado)))[0]
    if len(cruces_cero) > 1:
        T0 = 2 * np.mean(np.diff(t[cruces_cero]))       # Periodo aproximado
        omega0 = 2 * np.pi / T0
    else:
        omega0 = 1.0
    
    parametros_iniciales = [A0, gamma0, omega0, 0.0]

    # 5. Realizar el ajuste de curva
    try:
        popt, pcov = curve_fit(seno_amortiguado, t, y_centrado, p0=parametros_iniciales)
        A, gamma, omega, phi = popt
    except RuntimeError:
        print("El ajuste no pudo converger. Intenta revisar los datos.")
        return

    # 6. Calcular el coeficiente de determinación R^2
    y_ajuste = seno_amortiguado(t, *popt)
    residuos = y_centrado - y_ajuste
    ss_res = np.sum(residuos**2)
    ss_tot = np.sum((y_centrado - np.mean(y_centrado))**2)
    r_cuadrado = 1 - (ss_res / ss_tot)

    # 7. Automatizar el cálculo de decimales para los ejes
    # Usamos logaritmo base 10 para identificar la cifra significativa del error
    dec_t = max(0, int(np.ceil(-np.log10(error_tiempo))))
    dec_y = max(0, int(np.ceil(-np.log10(error_posicion))))

    # 8. Crear la gráfica
    fig, ax = plt.subplots(figsize=(10, 6))

    # Graficar los datos y la curva de ajuste
    ax.plot(t, y_centrado, 'o', color='royalblue', markersize=4, label='Datos experimentales (centrados)')
    ax.plot(t, y_ajuste, 'r-', linewidth=2, label='Ajuste: Seno amortiguado')

    # Configurar los decimales dinámicamente
    ax.xaxis.set_major_formatter(FormatStrFormatter(f'%.{dec_t}f'))
    ax.yaxis.set_major_formatter(FormatStrFormatter(f'%.{dec_y}f'))

    # Etiquetas de los ejes dinámicas
    ax.set_xlabel(rf'Tiempo ($t \pm {error_tiempo}$) s', fontsize=12)
    ax.set_ylabel(rf'Posición vertical ($y \pm {error_posicion}$) m', fontsize=12)


    # Crear el cuadro de texto
    texto_ajuste = (
        r"Ecuación: $y(t) = A e^{-\gamma t} \sin(\omega t + \phi)$" + "\n"
        f"$A = {A:.4f}$ m\n"
        f"$\gamma = {gamma:.4f}$ s$^{{-1}}$\n"
        f"$\omega = {omega:.4f}$ rad/s\n"
        f"$\phi = {phi:.4f}$ rad\n"
        f"$R^2 = {r_cuadrado:.4f}$"
    )

    ax.text(0.95, 0.95, texto_ajuste, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray'))

    ax.legend(loc='lower left')
    ax.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()

    # 9. Exportar el gráfico en alta resolución
    plt.savefig(nombre_imagen, dpi=300, bbox_inches='tight')
    print(f"\n¡Éxito! El gráfico se ha guardado en tu carpeta como: '{nombre_imagen}'")

    plt.show()

if __name__ == '__main__':
    # ====================================================================
    # ZONA DE CONFIGURACIÓN PARA EL ESTUDIANTE
    # Modifica los siguientes valores según tu experimento:
    # ====================================================================
    
    NOMBRE_ARCHIVO = 'datos.txt'
    NOMBRE_IMAGEN = 'grafico_ajuste_resorte.png'  # Nombre con el que se guardará tu gráfico
    
    # Ingresa tus errores instrumentales redondeados a UNA cifra significativa:
    ERROR_TIEMPO = 0.03      # Ejemplo: 0.03
    ERROR_POSICION = 0.0001  # Ejemplo: 0.0001
    
    # ====================================================================
    # EJECUCIÓN DEL ANÁLISIS
    # ====================================================================
    
    analizar_oscilacion(NOMBRE_ARCHIVO, ERROR_TIEMPO, ERROR_POSICION, NOMBRE_IMAGEN)
