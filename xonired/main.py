#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
red-xonidu v3.1 - CLI Principal
Agente autónomo ultraligero: SmolLM2 135M + APIs externas
"""

import sys
import argparse
import logging
from pathlib import Path

# Añadir core al path
sys.path.insert(0, str(Path(__file__).parent.parent / "core"))

from core.agent import create_agent, AgentResponse

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


def print_banner():
    print("""
╔══════════════════════════════════════════════════════════════╗
║                    red-xonidu v3.1                            ║
║         Agente Autónomo Ultraligero (135M params)             ║
║         Modelo local + APIs externas = Conocimiento real      ║
╚══════════════════════════════════════════════════════════════╝
""")


def print_help():
    print("""
USO:
  python -m red_xonidu.main [OPCIONES] [CONSULTA]

OPCIONES:
  -c, --consultar TEXT    Consulta directa (ej: "¿Qué es la IA?")
  -i, --interactivo       Modo chat interactivo
  -m, --modelos           Descargar/verificar modelos
  --install-llama         Instalar llama.cpp automáticamente
  -h, --help              Mostrar esta ayuda

EJEMPLOS:
  python -m red_xonidu.main --consultar "¿Qué es computación cuántica?"
  python -m red_xonidu.main --consultar "¿Va a llover en Cuautitlán?"
  python -m red_xonidu.main --consultar "Busca papers sobre transformers"
  python -m red_xonidu.main -i

CASOS DE USO PREDEFINIDOS:
  --tutor "tema"          Explicación socrática con Wikipedia
  --paper "tema"          Buscar papers en ArXiv
  --clima "ciudad"        Consultar clima actual
  --noticias "tema"       Verificar noticias
  --tramite "tema"        Info de trámites gubernamentales
""")


def run_interactive(agent):
    print("\n💬 Modo interactivo (escribe 'salir' para terminar)\n")
    while True:
        try:
            user_input = input("🧑 Tú: ").strip()
            if user_input.lower() in ('salir', 'exit', 'quit', 'q'):
                print("👋 ¡Hasta luego!")
                break
            if not user_input:
                continue

            print("🤖 Pensando...")
            response = agent.query(user_input)
            print(f"🤖: {response.final_answer}\n")

        except KeyboardInterrupt:
            print("\n👋 ¡Hasta luego!")
            break
        except EOFError:
            break


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('-c', '--consultar', type=str, help='Consulta directa')
    parser.add_argument('-i', '--interactivo', action='store_true', help='Modo interactivo')
    parser.add_argument('-m', '--modelos', action='store_true', help='Gestionar modelos')
    parser.add_argument('--install-llama', action='store_true', help='Instalar llama.cpp')
    parser.add_argument('--tutor', type=str, help='Modo tutor socrático')
    parser.add_argument('--paper', type=str, help='Buscar papers ArXiv')
    parser.add_argument('--clima', type=str, help='Consultar clima')
    parser.add_argument('--noticias', type=str, help='Buscar noticias')
    parser.add_argument('--tramite', type=str, help='Info trámites gobierno')
    parser.add_argument('-h', '--help', action='store_true', help='Ayuda')

    args = parser.parse_args()

    if args.help or (not args.consultar and not args.interactivo and not args.modelos
                     and not args.tutor and not args.paper and not args.clima
                     and not args.noticias and not args.tramite and not args.install_llama):
        print_banner()
        print_help()
        return

    print_banner()

    # Gestionar modelos
    if args.modelos or args.install_llama:
        from scripts.download_models import main as download_main
        sys.argv = ['download_models.py']
        if args.install_llama:
            sys.argv.append('--install-llama')
        if args.modelos:
            sys.argv.append('--quant=both')
        download_main()
        return

    # Crear agente
    try:
        agent = create_agent()
    except Exception as e:
        print(f"✗ Error inicializando agente: {e}")
        print("  Ejecuta: python -m red_xonidu.main --modelos --install-llama")
        return

    # Casos de uso predefinidos
    query = None
    if args.tutor:
        query = f"Explícame {args.tutor} de forma sencilla y hazme preguntas para guiarme"
    elif args.paper:
        query = f"Busca papers científicos recientes sobre {args.paper}"
    elif args.clima:
        query = f"¿Cómo está el clima en {args.clima}?"
    elif args.noticias:
        query = f"Busca noticias recientes sobre {args.noticias}"
    elif args.tramite:
        query = f"¿Cómo hago el trámite de {args.tramite} en México?"
    elif args.consultar:
        query = args.consultar

    if query:
        print(f"🔍 Consultando: {query}\n")
        response = agent.query(query)
        print(f"🤖: {response.final_answer}")
        if response.tool_calls:
            print(f"\n🔧 Herramientas usadas:")
            for tc in response.tool_calls:
                print(f"   • {tc.tool.value}: {tc.query}")
        return

    # Modo interactivo
    if args.interactivo:
        run_interactive(agent)


if __name__ == "__main__":
    main()