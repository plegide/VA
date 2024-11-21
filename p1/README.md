# VA

## Consideraciones importantes

Seguir las especificaciones tal y como están.

Hacer muchas pruebas para ver que funcionen los operadores.

Tener en cuenta que puede hacer falta normalizar valores negativos a intervalos entre [0,1]

Cuando lleguemos a la parte de CANI probar un circulo negro con fondo blanco y un fondo negro con un circulo blanco. Para saber si los bordes van bien en todas las orientaciones. Con imagenes tipo camaraman o lenna no va bien.

Pasan detector de plagio, con este y otros años.

No se puede usar np.histogram pero en algun punto hay que hacer un histograma.

Una imagen todo en negro con puntos blancos, se tiene que ver la gaussiana y si lo vuelves a pasar mini gaussianas. Para probar gaussiano.

DEFENSA:

Image 0: semillas: (0,0) y (25,25) vecinos 8 y vecinos 4 (4 configs)

Image 5: canny con dos umbrales bajos, dos altos y uno alto con otro bajo, sigma 3

Grid: filtro de medianas tamaño 3, 5 y 7

Eq0: ecualizacion -> adjustIntensity