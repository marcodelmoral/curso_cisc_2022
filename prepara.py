import argparse
import pathlib

from PIL import Image


def prepara_dataset(directorio_raiz, directorio_salida):
    for nombre_archivo in directorio_raiz.glob("*/*"):
        if "-d" in nombre_archivo.name:
            imagen = Image.open(nombre_archivo)
            dir_archivo = (
                nombre_archivo.parents[2]
                / directorio_salida
                / "multiclase"
                / "mascaras"
                / nombre_archivo.parent.name
            )
            dir_archivo.mkdir(parents=True, exist_ok=True)
            imagen.save(
                dir_archivo / (nombre_archivo.stem + ".png"), format="png"
            )
        else:
            imagen = Image.open(nombre_archivo)
            dir_archivo = (
                nombre_archivo.parents[2]
                / directorio_salida
                / "multiclase"
                / "imagenes"
                / nombre_archivo.parent.name
            )
            dir_archivo.mkdir(parents=True, exist_ok=True)
            imagen.save(
                dir_archivo / (nombre_archivo.stem + ".png"), format="png"
            )

    # Normal
    for nombre_archivo in directorio_raiz.glob("*/*"):
        if "-d" in nombre_archivo.name:
            if "normal" in nombre_archivo.parent.name:
                dir_archivo = (
                    nombre_archivo.parents[2]
                    / directorio_salida
                    / "binario"
                    / "mascaras"
                    / "normal"
                )
                dir_archivo.mkdir(parents=True, exist_ok=True)
            else:
                dir_archivo = (
                    nombre_archivo.parents[2]
                    / directorio_salida
                    / "binario"
                    / "mascaras"
                    / "anormal"
                )
                dir_archivo.mkdir(parents=True, exist_ok=True)
            imagen.save(
                dir_archivo / (nombre_archivo.stem + ".png"), format="png"
            )
        else:
            imagen = Image.open(nombre_archivo)
            if "normal" in nombre_archivo.parent.name:
                dir_archivo = (
                    nombre_archivo.parents[2]
                    / directorio_salida
                    / "binario"
                    / "imagenes"
                    / "normal"
                )
                dir_archivo.mkdir(parents=True, exist_ok=True)
            else:
                dir_archivo = (
                    nombre_archivo.parents[2]
                    / directorio_salida
                    / "binario"
                    / "imagenes"
                    / "anormal"
                )
                dir_archivo.mkdir(parents=True, exist_ok=True)
            imagen.save(
                dir_archivo / (nombre_archivo.stem + ".png"), format="png"
            )


if __name__ == "__main__":
    directorio_salida = "dataset"
    directorio_raiz = pathlib.Path("./smear2005/New database pictures/")

    parser = argparse.ArgumentParser(
        description="Just an example",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-r", "--raiz", help="archive mode")
    parser.add_argument("-s", "--salida", help="increase verbosity")
    args = parser.parse_args()
    config = vars(args)
    if config.get("raiz"):
        directorio_raiz = pathlib.Path(config["raiz"])
    if config.get("salida"):
        directorio_salida = config["salida"]

    prepara_dataset(directorio_raiz, directorio_salida)
