# VA

## Consideraciones importantes

- El ojo es una esfera que ve al reves, la inversion que hace el telescopio lo hacemos en el cerebro. Las miodiopsias son los puntitos, cuando se joroba el humor vitreo hace densidades que hacen que no pase la luz limpia y ves las manchitas grises. Lo que pasa es que el ojo las calibra. La pupila en el fondo de ojo abre o cierra absorbiendo mas o menos luz, en zonas oscuras abre, en zonas claras cierra y lo proyecta en la zona foveal, donde hay muchisimos fotoreceptores, que absorven y procesan la luz. El nervio optico hace que todo funcione a la perfeccion con las venas y arterias.

- Las patologias afectan a la copa del nervio optico (el glaucoma es la mas tipica, falta riego en la zona periferica y perdemos angulo de vision, poco a poco), o a la fovea (son las peores) cuanto mas cerca del centro peor, de la agudeza visual.


- Imagenes de nervio optico. Hay otro dataset con mas imagenes. 

- Se diagnostica con la copa y el disco. Si tiene el glaucoma se inflama la copa y cambian los ratios entre uno y otro. Suele haber un umbral para indicar esto. 

- Detectar con un centroide el disco optico, es la zona mas brillante de la imagen, sobre todo las sanas. Primero se identifica ese circulo brillante.

- Se establece un tamaño aproximado para coger la ROI (a mano)

- Dentro de la ROI segmentamos copa y disco, todo el disco optico es mas claro que el entorno y el disco aun mas claro. Con las imagenes en blanco y negro (grand truth, que es el CDR ideal) se ve cual es cual.

- Por ultimo vamos a sacar su ratio con una formula por internet

- Se tiene que evaluar cuantitativamente lo que hacemos (Saber con numeros si va bien)

- El sistema puede ser malo identificando y buenisimo diagnosticando, dependiendo del bias.

- El problem a es la copa pq tiene el nervio optico en medio. Se hace asumiendo una forma en la segmentacion. Una elipse o un ovalo.

- Metricas de segmentacion IoU (interseccion sobre la union) y Factor dice.