#!/usr/bin/env python

import re
import argparse

def estrai_throughput(file_path):
    valori = []

    try:
        with open(file_path, 'r') as file:
            for riga in file:
                match = re.search(r"throughput per GPU \(TFLOP/s/GPU\):\s*([0-9.]+)", riga)
                if match:
                    valore = float(match.group(1))
                    valori.append(valore)
    except FileNotFoundError:
        print(f"Errore: il file '{file_path}' non esiste.")
        return

    if not valori:
        print("⚠️  Nessun valore trovato.")
        return

    media = sum(valori) / len(valori)
    massimo = max(valori)

    print(f"Valori trovati: {valori}")
    print(f"Media: {media:.2f} TFLOP/s/GPU")
    print(f"Massimo: {massimo:.2f} TFLOP/s/GPU")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Estrae throughput da log e calcola media e massimo.")
    parser.add_argument("file", help="Percorso del file di log da analizzare.")
    args = parser.parse_args()

    estrai_throughput(args.file)
