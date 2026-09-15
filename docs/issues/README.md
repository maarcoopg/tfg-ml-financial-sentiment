# Issues de la revisión

Estos son borradores locales conservados con la plantilla de imprevistos. No tienen número de GitHub: su publicación fue rechazada por la política de aprobación de la sesión del 8 de septiembre.

| Borrador | Rama prevista | Contenido |
| --- | --- | --- |
| revision-01.md | feature/point-in-time-alignment | Horarios bursátiles, timestamp y purga de etiquetas |
| revision-02.md | feature/news-coverage-audit | Cobertura, fuentes, relevancia y errores de descarga |
| revision-03.md | feature/nested-temporal-evaluation | Selección interna, bloques externos y bootstrap pareado |
| revision-04.md | feature/reproducible-experiments | Artefactos aislados, pruebas y correcciones del pipeline |
| revision-05.md | feature/controlled-sentiment-ablations | Doce variantes controladas, cinco algoritmos y documentación |

Publicar el título de cada archivo sin el prefijo Markdown `#`, usar el resto como cuerpo y aplicar la etiqueta `imprevisto`. Verificar que no exista ya una issue equivalente antes de crearla. No cerrar una issue hasta que sus cambios estén verificados e integrados.

Los cambios comparten la rama de la issue #45 porque las restricciones iniciales impidieron separarlos. El 15 de septiembre el usuario solicitó publicar e integrar esa entrega acumulada. No se han creado las ramas de esta tabla: se conservan como planificación original. La implementación y los resultados se describen en `docs/revision_implementation.md` y `docs/resultados_revision_temporal.md`.

## Verificación local completada

- Revisión 01: alineación con cierres NASDAQ, timestamps y purga verificados con pruebas y con las fronteras de todos los entrenamientos.
- Revisión 02: informes de cobertura generados; comprobados chunks vacíos, reintentos, errores y saturación de un día. La cobertura del proveedor sigue sin estar certificada.
- Revisión 03: ejecutados 1.080 ajustes internos y 183 entrenamientos externos; generados 62 intervalos pareados. El histórico sigue siendo exploratorio.
- Revisión 04: 28 pruebas locales superadas, incluidas seis de organización de artefactos; modelos y hashes verificados; ejecuciones aisladas y CI preparada. El resultado de CI remoto se consulta en GitHub Actions.
- Revisión 05: comparadas doce variantes con cinco algoritmos y baseline; informe y figuras completados. No se demuestra mejora positiva concluyente en los intervalos calculados.

La finalización local no equivale al cierre de una issue en GitHub ni a su integración en main.
