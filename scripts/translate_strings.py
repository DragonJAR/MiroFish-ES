#!/usr/bin/env python3
"""
translate_strings.py - Traduce automáticamente strings chinos al español

LEE:
  - .translation-pending.txt: archivo con strings pendientes
  - .env: credenciales del LLM

FORMATO DE ENTRADA (.translation-pending.txt):
  archivo:linea:"string en chino"

OUTPUT:
  - Reemplaza los strings en los archivos
  - Genera .translation-report.txt con el resumen

USO:
  python3 scripts/translate_strings.py --input .translation-pending.txt --env .env
"""

import os
import sys
import re
import argparse
from pathlib import Path
from typing import List, Tuple, Optional

# Agregar backend al path para usar las dependencias
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

try:
    from openai import OpenAI
except ImportError:
    print("Error:openai no está instalado. Ejecuta: pip install openai")
    sys.exit(1)


def parse_env_file(env_path: str) -> dict:
    """Parsea archivo .env y retorna dict con variables."""
    env_vars = {}
    if not os.path.exists(env_path):
        print(f"Error: No se encontró {env_path}")
        return env_vars

    with open(env_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                key, value = line.split("=", 1)
                # Remover comillas si las hay
                value = value.strip("\"'")
                env_vars[key.strip()] = value

    return env_vars


def parse_pending_file(pending_path: str) -> List[Tuple[str, int, str]]:
    """Parsea archivo de strings pendientes.

    Returns:
        List of (archivo, linea, string_original)
    """
    entries = []
    if not os.path.exists(pending_path):
        print(f"Error: No se encontró {pending_path}")
        return entries

    with open(pending_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # Formato: archivo:linea:"string"
            match = re.match(r'^([^:]+):(\d+):"(.+)"$', line)
            if match:
                archivo = match.group(1)
                linea = int(match.group(2))
                string_original = match.group(3)
                entries.append((archivo, linea, string_original))
            else:
                print(f"Warning: Línea no parseada: {line}")

    return entries


def translate_batch(client: OpenAI, model: str, strings: List[str]) -> List[str]:
    """Traduce un batch de strings chinos al español usando LLM.

    Returns:
        Lista de strings traducidos en el mismo orden
    """
    if not strings:
        return []

    # Prompt optimizado para traducción
    prompt = """Eres un traductor profesional de chino a español.

Traduce los siguientes strings al español. Sigue estas reglas:
1. Mantén el significado original exacto
2. Usa lenguaje natural en español neutro
3. Conserva cualquier formato especial (corchetes, comillas, etc)
4. Si hay variables como {variable} o %s, mantenlas exactamente igual
5. Devuelve SOLO las traducciones, una por línea, en el mismo orden

Strings a traducir:
"""

    for i, s in enumerate(strings, 1):
        prompt += f"\n{i}. {s}"

    prompt += "\n\nTraducciones:"

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "Eres un traductor profesional. Responde SOLO con las traducciones, sin explicaciones adicionales.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,  # Baja temperatura para traducciones consistentes
            max_tokens=2000,
        )

        translated_text = response.choices[0].message.content.strip()
        translations = [t.strip() for t in translated_text.split("\n") if t.strip()]

        # Verificar que tenemos el mismo número de traducciones que inputs
        if len(translations) != len(strings):
            print(
                f"Warning: Esperaba {len(strings)} traducciones, recibí {len(translations)}"
            )
            # Rellenar con strings originales si faltan
            while len(translations) < len(strings):
                translations.append(strings[len(translations)])

        return translations

    except Exception as e:
        print(f"Error en traducción batch: {e}")
        return strings  # Retornar originales en caso de error


def apply_translation(archivo: str, linea: int, original: str, traduccion: str) -> bool:
    """Aplica una traducción a un archivo reemplazando la línea específica."""
    if not os.path.exists(archivo):
        print(f"Error: No existe {archivo}")
        return False

    try:
        with open(archivo, "r", encoding="utf-8") as f:
            lines = f.readlines()

        if linea > len(lines) or linea < 1:
            print(f"Error: Línea {linea} fuera de rango en {archivo}")
            return False

        # Obtener línea actual
        current_line = lines[linea - 1]

        # Intentar reemplazar solo el string chino dentro de la línea
        # Buscar el string chino en la línea (puede estar en diferentes contextos)
        if original in current_line:
            # Reemplazo directo
            new_line = current_line.replace(original, traduccion)
        elif has_chinese(current_line):
            # Si la línea tiene chino pero no exactamente el string original,
            # intentar reemplazar cualquier secuencia de caracteres chinos
            chinese_pattern = re.compile(r"[一-龥]+")
            # Solo reemplazar si encontramos chino
            new_line = chinese_pattern.sub(traduccion, current_line, count=1)
        else:
            print(f"Warning: Línea {linea} en {archivo} no contiene chino, saltando")
            return False

        lines[linea - 1] = new_line

        with open(archivo, "w", encoding="utf-8") as f:
            f.writelines(lines)

        return True

    except Exception as e:
        print(f"Error aplicando traducción en {archivo}:{linea}: {e}")
        return False


def has_chinese(text: str) -> bool:
    """Verifica si un texto contiene caracteres chinos."""
    return bool(re.search(r"[一-龥]", text))


def main():
    parser = argparse.ArgumentParser(description="Traduce strings chinos al español")
    parser.add_argument(
        "--input", "-i", required=True, help="Archivo .translation-pending.txt"
    )
    parser.add_argument(
        "--env", "-e", required=True, help="Archivo .env con credenciales"
    )
    parser.add_argument(
        "--batch-size",
        "-b",
        type=int,
        default=10,
        help="Tamaño de batch para traducción",
    )
    parser.add_argument(
        "--dry-run", "-n", action="store_true", help="Solo mostrar qué se traduciría"
    )

    args = parser.parse_args()

    # 1. Parsear .env
    print(f"Leyendo configuración desde {args.env}...")
    env_vars = parse_env_file(args.env)

    api_key = env_vars.get("LLM_API_KEY")
    base_url = env_vars.get("LLM_BASE_URL", "https://api.openai.com/v1")
    model = env_vars.get("LLM_MODEL_NAME", "gpt-4o-mini")

    if not api_key:
        print("Error: LLM_API_KEY no encontrado en .env")
        sys.exit(1)

    print(f"  API Key: {'*' * 10}...{api_key[-4:]}")
    print(f"  Base URL: {base_url}")
    print(f"  Modelo: {model}")
    print()

    # 2. Parsear strings pendientes
    print(f"Leyendo strings desde {args.input}...")
    entries = parse_pending_file(args.input)

    if not entries:
        print("No hay strings pendientes de traducción")
        sys.exit(0)

    print(f"  Encontrados: {len(entries)} strings")
    print()

    # 3. Agrupar por archivo para reporte
    by_file = {}
    for archivo, linea, string in entries:
        if archivo not in by_file:
            by_file[archivo] = []
        by_file[archivo].append((linea, string))

    print("Archivos afectados:")
    for archivo, items in by_file.items():
        print(f"  {archivo}: {len(items)} strings")
    print()

    if args.dry_run:
        print("[DRY-RUN] Strings a traducir:")
        for archivo, linea, string in entries:
            print(f"  {archivo}:{linea}: {string[:50]}...")
        sys.exit(0)

    # 4. Traducir en batches
    print("Traduciendo...")
    client = OpenAI(api_key=api_key, base_url=base_url)

    results = []  # (archivo, linea, original, traduccion, success)

    for i in range(0, len(entries), args.batch_size):
        batch = entries[i : i + args.batch_size]
        strings_to_translate = [e[2] for e in batch]

        print(f"  Batch {i // args.batch_size + 1}: {len(batch)} strings...")

        translations = translate_batch(client, model, strings_to_translate)

        for j, (archivo, linea, original) in enumerate(batch):
            traduccion = translations[j] if j < len(translations) else original
            results.append((archivo, linea, original, traduccion))

    print()

    # 5. Aplicar traducciones
    print("Aplicando traducciones...")
    applied = 0
    failed = 0

    for archivo, linea, original, traduccion in results:
        if apply_translation(archivo, linea, original, traduccion):
            print(f"  ✓ {archivo}:{linea}")
            applied += 1
        else:
            print(f"  ✗ {archivo}:{linea}")
            failed += 1

    print()

    # 6. Generar reporte
    report_path = Path(args.input).parent / ".translation-report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("REPORTE DE TRADUCCIÓN\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Total strings: {len(entries)}\n")
        f.write(f"Aplicados: {applied}\n")
        f.write(f"Fallidos: {failed}\n\n")
        f.write("DETALLE:\n")
        f.write("-" * 60 + "\n")

        for archivo, linea, original, traduccion in results:
            f.write(f"\n{archivo}:{linea}\n")
            f.write(f"  Original: {original}\n")
            f.write(f"  Traducción: {traduccion}\n")

    print(f"Reporte generado: {report_path}")

    # 7. Limpiar archivo pendiente si todo salió bien
    if failed == 0:
        pending_path = Path(args.input)
        if pending_path.exists():
            pending_path.unlink()
            print(f"Archivo pendientes limpiado: {args.input}")
    else:
        print(f"Quedan traducciones fallidas, revisa {args.input}")

    print()
    print(f"✓ Completado: {applied} traducciones aplicadas")

    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
