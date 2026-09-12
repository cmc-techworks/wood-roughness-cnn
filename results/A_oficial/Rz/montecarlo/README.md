# Monte Carlo — 20 replicas bootstrap, arquitectura A, Rz

- Nivel de remuestreo: **jerarquico**. la identidad de la probeta explica el 94% de la brecha CV-test y la SD entre probetas es 5.6x la SD entre modelos, de modo que un remuestreo que ignore el agrupamiento subestima la banda
- Corrida: `montecarlo_incertidumbre.py --arch A --param Rz --n-replicas 20 --nivel jerarquico --seed 10`
- Cobertura empirica del intervalo del 90%: 33.2% (ancho medio 3.890)
- Los indices de cada replica estan en `indices_bootstrap/`: sin ellos no es reproducible
- Alimenta: R1.4 (intervalos de confianza para los mapas espaciales)
