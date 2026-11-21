#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Descargador de modelos GGUF para red-xonidu v3.0
Descarga SmolLM2-135M en Q4_K_M y Q8_0 desde Hugging Face
"""

import os
import sys
import subprocess
from pathlib import Path
from huggingface_hub import hf_hub_download

MODELS_DIR = Path(__file__).parent.parent / "models"
MODELS_DIR.mkdir(exist_ok=True)

REPO_ID = "bartowski/SmolLM2-135M-Instruct-GGUF"
FILES = {
    "Q4_K_M": "SmolLM2-135M-Instruct-Q4_K_M.gguf",
    "Q8_0": "SmolLM2-135M-Instruct-Q8_0.gguf",
}


def download_model(quant: str, force: bool = False) -> bool:
    """Descarga un modelo específico"""
    filename = FILES[quant]
    dest = MODELS_DIR / filename

    if dest.exists() and not force:
        print(f"✓ {filename} ya existe ({dest.stat().st_size / 1024 / 1024:.1f} MB)")
        return True

    print(f"⬇ Descargando {filename}...")
    try:
        hf_hub_download(
            repo_id=REPO_ID,
            filename=filename,
            local_dir=MODELS_DIR,
            local_dir_use_symlinks=False
        )
        print(f"✓ {filename} descargado ({dest.stat().st_size / 1024 / 1024:.1f} MB)")
        return True
    except Exception as e:
        print(f"✗ Error descargando {filename}: {e}")
        return False


def install_llama_cpp() -> bool:
    """Instala llama.cpp si no está disponible"""
    try:
        subprocess.run(['llama-cli', '--version'], capture_output=True, check=True)
        print("✓ llama.cpp ya instalado")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("⬇ Instalando llama.cpp...")
        try:
            # Intentar con pacman (Arch)
            subprocess.run(['sudo', 'pacman', '-S', '--noconfirm', 'llama.cpp'], check=True)
            print("✓ llama.cpp instalado via pacman")
            return True
        except subprocess.CalledProcessError:
            print("✗ No se pudo instalar llama.cpp automáticamente")
            print("  Instala manualmente: sudo pacman -S llama.cpp")
            return False


def check_ram() -> float:
    """Retorna RAM en GB"""
    try:
        import psutil
        return psutil.virtual_memory().total / (1024**3)
    except ImportError:
        # Fallback: leer /proc/meminfo
        with open('/proc/meminfo') as f:
            for line in f:
                if line.startswith('MemTotal:'):
                    kb = int(line.split()[1])
                    return kb / (1024**2)
    return 0


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Descargador de modelos red-xonidu v3.0")
    parser.add_argument('--quant', choices=['Q4_K_M', 'Q8_0', 'both'], default='both',
                       help='Cuantización a descargar')
    parser.add_argument('--force', action='store_true', help='Forzar re-descarga')
    parser.add_argument('--install-llama', action='store_true', help='Instalar llama.cpp')
    args = parser.parse_args()

    print("=" * 50)
    print("red-xonidu v3.0 - Model Downloader")
    print("=" * 50)

    # Verificar RAM
    ram_gb = check_ram()
    print(f"RAM detectada: {ram_gb:.1f} GB")

    if ram_gb < 3:
        print("→ Recomendado: Q4_K_M (para 1-2 GB RAM)")
    else:
        print("→ Recomendado: Q8_0 (para 4+ GB RAM)")

    # Instalar llama.cpp si se pide
    if args.install_llama:
        install_llama_cpp()

    # Descargar modelos
    success = True

    if args.quant in ['Q4_K_M', 'both']:
        success &= download_model('Q4_K_M', args.force)

    if args.quant in ['Q8_0', 'both']:
        success &= download_model('Q8_0', args.force)

    print("\n" + "=" * 50)
    if success:
        print("✓ Modelos listos para usar")
        print(f"  Ubicación: {MODELS_DIR}")
    else:
        print("✗ Hubo errores en la descarga")
        sys.exit(1)


if __name__ == "__main__":
    main()