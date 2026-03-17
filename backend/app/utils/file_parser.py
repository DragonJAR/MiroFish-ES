"""
Utilidades de parsing de archivos
Soporta extraccion de texto de archivos PDF, Markdown y TXT
"""

import os
from pathlib import Path
from typing import List, Optional


def _read_text_with_fallback(file_path: str) -> str:
    """
    Leer archivo de texto con deteccion automatica de codificacion.
    
    Estrategia multi-nivel:
    1. Primero intentar decodificacion UTF-8
    2. Usar charset_normalizer para detectar codificacion
    3. Recurrir a chardet para detectar codificacion
    4. Usar UTF-8 + errors='replace' como ultimo recurso
    
    Args:
        file_path: Ruta del archivo
        
    Returns:
        Contenido decodificado
    """
    data = Path(file_path).read_bytes()
    
    # Primero intentar UTF-8
    try:
        return data.decode('utf-8')
    except UnicodeDecodeError:
        pass
    
    # Intentar usar charset_normalizer para detectar codificacion
    encoding = None
    try:
        from charset_normalizer import from_bytes
        best = from_bytes(data).best()
        if best and best.encoding:
            encoding = best.encoding
    except Exception:
        pass
    
    # Recurrir a chardet
    if not encoding:
        try:
            import chardet
            result = chardet.detect(data)
            encoding = result.get('encoding') if result else None
        except Exception:
            pass
    
    # Ultimo recurso: usar UTF-8 + replace
    if not encoding:
        encoding = 'utf-8'
    
    return data.decode(encoding, errors='replace')


class FileParser:
    """Parser de archivos"""
    
    SUPPORTED_EXTENSIONS = {'.pdf', '.md', '.markdown', '.txt'}
    
    @classmethod
    def extract_text(cls, file_path: str) -> str:
        """
        Extraer texto de un archivo
        
        Args:
            file_path: Ruta del archivo
            
        Returns:
            Contenido de texto extraido
        """
        path = Path(file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"El archivo no existe: {file_path}")
        
        suffix = path.suffix.lower()
        
        if suffix not in cls.SUPPORTED_EXTENSIONS:
            raise ValueError(f"Formato de archivo no soportado: {suffix}")
        
        if suffix == '.pdf':
            return cls._extract_from_pdf(file_path)
        elif suffix in {'.md', '.markdown'}:
            return cls._extract_from_md(file_path)
        elif suffix == '.txt':
            return cls._extract_from_txt(file_path)
        
        raise ValueError(f"No se puede procesar el formato: {suffix}")
    
    @staticmethod
    def _extract_from_pdf(file_path: str) -> str:
        """Extraer texto de PDF"""
        try:
            import fitz  # PyMuPDF
        except ImportError:
            raise ImportError("Se requiere PyMuPDF: pip install PyMuPDF")
        
        text_parts = []
        with fitz.open(file_path) as doc:
            for page in doc:
                text = page.get_text()
                if text.strip():
                    text_parts.append(text)
        
        return "\n\n".join(text_parts)
    
    @staticmethod
    def _extract_from_md(file_path: str) -> str:
        """Extraer texto de Markdown con deteccion automatica de codificacion"""
        return _read_text_with_fallback(file_path)
    
    @staticmethod
    def _extract_from_txt(file_path: str) -> str:
        """Extraer texto de TXT con deteccion automatica de codificacion"""
        return _read_text_with_fallback(file_path)
    
    @classmethod
    def extract_from_multiple(cls, file_paths: List[str]) -> str:
        """
        Extraer texto de multiples archivos y combinar
        
        Args:
            file_paths: Lista de rutas de archivos
            
        Returns:
            Texto combinado
        """
        all_texts = []
        
        for i, file_path in enumerate(file_paths, 1):
            try:
                text = cls.extract_text(file_path)
                filename = Path(file_path).name
                all_texts.append(f"=== Documento {i}: {filename} ===\n{text}")
            except Exception as e:
                all_texts.append(f"=== Documento {i}: {file_path} (Error al extraer: {str(e)}) ===")
        
        return "\n\n".join(all_texts)


def split_text_into_chunks(
    text: str, 
    chunk_size: int = 500, 
    overlap: int = 50
) -> List[str]:
    """
    Dividir texto en chunks pequenos
    
    Args:
        text: Texto original
        chunk_size: Numero de caracteres por chunk
        overlap: Numero de caracteres de superposicion
        
    Returns:
        Lista de chunks de texto
    """
    if len(text) <= chunk_size:
        return [text] if text.strip() else []
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        
        # Intentar dividir en limites de oracion
        if end < len(text):
            # Buscar el separador de oracion mas cercano
            for sep in ['.', '!', '?', '.\n', '!\n', '?\n', '\n\n', '. ', '! ', '? ']:
                last_sep = text[start:end].rfind(sep)
                if last_sep != -1 and last_sep > chunk_size * 0.3:
                    end = start + last_sep + len(sep)
                    break
        
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        
        # El siguiente chunk empieza en la posicion de superposicion
        start = end - overlap if end < len(text) else len(text)
    
    return chunks
